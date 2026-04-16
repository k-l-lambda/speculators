"""Custom worker extension for hidden states capture."""

import logging
import types
from collections import defaultdict
from itertools import islice

import torch
from vllm.distributed import get_pp_group, get_tp_group
from vllm.sequence import IntermediateTensors

__all__ = ["HiddenStatesWorkerExtension"]

logger = logging.getLogger(__name__)


def _patched_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
    **_kwargs,
):
    """Patched forward pass that captures hidden states from specified layers.

    This function is bound to base_model instances via types.MethodType.
    It expects base_model to have an _extension attribute pointing to the
    HiddenStatesWorkerExtension instance.

    Args:
        deepstack_input_embeds: For multimodal models with deepstack (Qwen3VL)
    """
    if get_pp_group().is_first_rank:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = []
    extension = self._extension  # noqa: SLF001
    # Only capture on TP rank 0 to avoid duplicates
    should_capture = get_tp_group().rank_in_group == 0
    target_layers = extension._layer_ids if should_capture else frozenset()  # noqa: SLF001

    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(
            hidden_states=hidden_states, positions=positions, residual=residual
        )
        absolute_layer_idx = self.start_layer + idx

        # Capture intermediate layers (not the last) before norm
        if absolute_layer_idx in target_layers:
            aux_hidden_states.append((hidden_states + residual).detach().cpu())

    # Return early if not last PP rank
    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )

    hidden_states, _ = self.norm(hidden_states, residual)
    if should_capture and aux_hidden_states:
        # Replace the last captured layer with post-norm version if it was the final layer
        # vLLM runtime passes post-norm hidden_states to MTP, so training data must match
        last_layer_idx = self.end_layer - 1
        if last_layer_idx in target_layers:
            aux_hidden_states[-1] = hidden_states.detach().cpu()
        extension._store_captured_states(aux_hidden_states)  # noqa: SLF001

    return hidden_states


class HiddenStatesWorkerExtension:
    """Worker extension that adds hidden states capture functionality.

    This extension hooks into VLLM's Worker initialization by being specified
    in ParallelConfig.worker_extension_cls. It patches the model's forward pass
    to intercept and capture intermediate layer hidden states during inference.

    Key behaviors:
    - Only captures on tensor parallel (TP) rank 0 to avoid duplicate data when
      using tensor parallelism. All TP ranks compute the same hidden states, so
      capturing from rank 0 is sufficient.
    - Stores captured states in GPU memory during batch processing as lists of
      tensors, concatenating them only when retrieved via _get_captured_states().
    - Supports pipeline parallelism by handling IntermediateTensors correctly.

    Attributes:
        _layer_ids: Frozenset of layer indices for O(1) lookup during capture
        _captured_states: Accumulated hidden states per layer (GPU tensors)
        model_runner: Reference to the VLLM model runner
    """

    def _store_captured_states(self, aux_hidden_states):
        if self._captured_states is None:  # type: ignore[has-type]
            self._captured_states = [[h] for h in aux_hidden_states]
        else:
            for i, h in enumerate(aux_hidden_states):
                self._captured_states[i].append(h)

        metadata = getattr(self, "_current_request_metadata", None)
        if metadata is not None:
            # Sort by vLLM's actual batch position (vLLM reorders requests internally)
            input_batch = self.model_runner.input_batch  # type: ignore[attr-defined]
            sorted_metadata = sorted(
                metadata.items(),
                key=lambda item: input_batch.req_id_to_index.get(item[0], float("inf")),
            )
            self._request_metadata.append(sorted_metadata)  # type: ignore[has-type]

    def _setup_hidden_states_capture(self, layer_ids: list[int]):
        """Setup model to capture auxiliary hidden states from specific layers"""
        self._layer_ids = frozenset(layer_ids)  # Convert once for O(1) lookup
        self._captured_states = None  # type: ignore[assignment]

        model = self.model_runner.model  # type: ignore[attr-defined]

        # Vision-language models
        if hasattr(model, "get_language_model"):
            base_model = model.get_language_model().model
        # Text models
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            base_model = model.model
        else:
            attrs = [a for a in dir(model) if not a.startswith("_")]
            raise AttributeError(
                f"Could not find base model with 'layers' attribute. "
                f"Model type: {type(model).__name__}, "
                f"Available attributes: {attrs}"
            )

        base_model._extension = self  # noqa: SLF001
        base_model.forward = types.MethodType(_patched_forward, base_model)
        logger.info(f"Hidden states capture setup complete for layers {layer_ids}")

    def _set_request_metadata(self, request_metadata: dict[str, int]):
        """Set request metadata for the next forward pass.

        Args:
            request_metadata: Dict mapping request_id -> num_prefill_tokens
        """
        self._current_request_metadata = request_metadata  # type: ignore[assignment]

    def _reset_capture(self):
        """Reset captured states before starting a new batch"""
        if not hasattr(self, "_layer_ids"):
            raise RuntimeError(
                "Must call _setup_hidden_states_capture before capturing states"
            )
        self._captured_states = None  # type: ignore[assignment]
        self._request_metadata = []  # type: ignore[assignment]
        self._current_request_metadata = None  # type: ignore[assignment]

    def _get_captured_states(self):
        """Get the captured hidden states organized by request ID.

        Returns:
            Dict mapping request_id to list of CPU tensors (one per layer),
            or None if no states captured.

        Tensors are already on CPU (moved during _patched_forward capture),
        so all operations here are pure CPU with no CUDA sync required.
        """
        import time as _time
        import sys as _sys
        t0 = _time.monotonic()
        rank = get_tp_group().rank_in_group
        print(f"[custom_worker] _get_captured_states rank={rank} enter", flush=True, file=_sys.stderr)

        if self._captured_states is None:
            print(f"[custom_worker] _get_captured_states rank={rank} no states, returning None", flush=True, file=_sys.stderr)
            return None

        n_layers = len(self._captured_states)
        n_iters = len(self._captured_states[0])
        total_toks = sum(t.shape[0] for t in self._captured_states[0])
        print(f"[custom_worker] _get_captured_states rank={rank} {n_layers} layers, {n_iters} iters, {total_toks} toks", flush=True, file=_sys.stderr)

        # Concatenate captured states from all scheduler iterations (already CPU tensors)
        t1 = _time.monotonic()
        concatenated_layers = [
            torch.cat(layer_tensors, dim=0) for layer_tensors in self._captured_states
        ]
        print(f"[custom_worker] _get_captured_states rank={rank} cat done in {_time.monotonic()-t1:.3f}s shape={concatenated_layers[0].shape}", flush=True, file=_sys.stderr)

        # Slice and group by request
        t2 = _time.monotonic()
        request_chunks: defaultdict[str, list[list[torch.Tensor]]] = defaultdict(
            lambda: [[] for _ in range(len(concatenated_layers))]
        )
        current_idx = 0

        for metadata in self._request_metadata:  # type: ignore[has-type]
            for req_id, num_tok in metadata:
                for layer_idx, layer_tensor in enumerate(concatenated_layers):
                    chunk = layer_tensor[current_idx : current_idx + num_tok].clone()
                    request_chunks[req_id][layer_idx].append(chunk)
                current_idx += num_tok
        print(f"[custom_worker] _get_captured_states rank={rank} slicing done in {_time.monotonic()-t2:.3f}s", flush=True, file=_sys.stderr)

        # Concatenate chunks — tensors already on CPU, no .cpu() needed
        t3 = _time.monotonic()
        result: dict[str, list[torch.Tensor]] = {
            req_id: [torch.cat(chunks, dim=0) for chunks in layer_chunks]
            for req_id, layer_chunks in request_chunks.items()
        }
        print(f"[custom_worker] _get_captured_states rank={rank} assemble done in {_time.monotonic()-t3:.3f}s {len(result)} reqs total={_time.monotonic()-t0:.3f}s", flush=True, file=_sys.stderr)

        # Clear intermediate storage
        self._captured_states = None  # type: ignore[assignment]
        self._request_metadata = []  # type: ignore[assignment]
        return result
