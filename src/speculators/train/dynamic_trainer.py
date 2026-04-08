"""Dynamic hidden states training for Eagle3.

Generates hidden states on-the-fly via VllmHiddenStatesGenerator during the
training loop, eliminating the need for pre-extracted .pt files on disk.

Requires TP=8 vLLM (uses GPUs 1-7 for K2.5 shards), Eagle3 trains on GPU 0.
"""

import logging
import queue
import threading
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning  # noqa: F401
from tqdm.rich import tqdm

from speculators.data_generation.vllm_hidden_states_generator import (
    VllmHiddenStatesGenerator,
)
from speculators.model import SpeculatorModel
from speculators.train.data import (
    BatchType,
    StandardizeFnSig,
    create_collate_fn,
    process_generated_sample,
    standardize_data_v1,
)
from speculators.train.noise_transforms import TransformTensors
from speculators.train.trainer import Trainer, TrainerConfig

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")

# Maximum consecutive generation failures before aborting
MAX_CONSECUTIVE_FAILURES = 5


class DynamicBatchPrefetcher:
    """Background thread that pre-generates training batches with hidden states.

    Runs _generate_batch() in a background thread, buffering ready batches so
    the training loop doesn't block waiting for vLLM generation.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        generate_fn,
        buffer_size: int = 2,
    ):
        self.dataloader = dataloader
        self.generate_fn = generate_fn
        self.buffer_size = buffer_size
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._epoch: int = 0
        self._error: BaseException | None = None
        self._skipped: int = 0
        self._total: int = 0

    def start(self, epoch: int):
        """Start prefetching for an epoch."""
        self._epoch = epoch
        self._stop_event.clear()
        self._error = None
        self._skipped = 0
        self._total = 0
        # Drain any leftover items from previous epoch
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the producer to stop and wait for it."""
        self._stop_event.set()
        # Drain queue so producer can unblock from put()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def _put(self, item):
        """Put item into queue, respecting stop event to avoid deadlock."""
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _producer(self):
        """Background thread: iterate dataloader, generate batches, enqueue."""
        try:
            if hasattr(self.dataloader.batch_sampler, "set_epoch"):
                self.dataloader.batch_sampler.set_epoch(self._epoch)

            for batch in self.dataloader:
                if self._stop_event.is_set():
                    break
                try:
                    gpu_batch = self.generate_fn(batch)
                    if not self._put(gpu_batch):
                        break
                except Exception as e:
                    root_logger.error(f"Generation failed in prefetcher: {e}")
                    if not self._put(None):
                        break
        except Exception as e:
            self._error = e
        finally:
            # Sentinel to signal end of epoch — must always be sent
            try:
                self._queue.put(StopIteration, timeout=5.0)
            except queue.Full:
                pass

    def __iter__(self):
        return self

    def __next__(self) -> BatchType:
        if self._error is not None:
            raise self._error
        try:
            item = self._queue.get(timeout=300)
        except queue.Empty:
            raise RuntimeError("Prefetcher timed out waiting for batch (300s)")
        if item is StopIteration:
            if self._skipped > 0:
                root_logger.warning(
                    f"Epoch {self._epoch}: skipped {self._skipped}/{self._total} "
                    f"batches ({self._skipped / max(self._total, 1) * 100:.1f}%)"
                )
            raise StopIteration
        self._total += 1
        if item is None:
            # Generation failed for this batch, skip
            self._skipped += 1
            if self._total >= 50 and self._skipped / self._total > 0.5:
                raise RuntimeError(
                    f"Too many batch failures: {self._skipped}/{self._total} "
                    f"({self._skipped / self._total * 100:.0f}%) — aborting epoch"
                )
            return self.__next__()
        return item

    def __len__(self):
        return len(self.dataloader)


class DynamicTrainer(Trainer):
    """Trainer that generates hidden states on-the-fly via VllmHiddenStatesGenerator.

    Overrides train_epoch/val_epoch to insert generation before model.forward().
    Uses a background prefetcher to overlap generation with training.
    """

    def __init__(
        self,
        model: SpeculatorModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        generator: VllmHiddenStatesGenerator,
        noise_transform: TransformTensors | None = None,
        standardize_fn: StandardizeFnSig = standardize_data_v1,
        max_len: int = 8192,
        hidden_states_dtype: torch.dtype = torch.float,
        epoch_samples: int = 0,
    ):
        assert not config.is_distributed, (
            "DynamicTrainer only supports single-GPU training "
            "(vLLM uses TP for other GPUs)"
        )
        self.generator = generator
        self.noise_transform = noise_transform
        self.standardize_fn = standardize_fn
        self.max_len = max_len
        self.hidden_states_dtype = hidden_states_dtype
        self._collate_fn = create_collate_fn(max_len)
        self._epoch_samples = epoch_samples
        self._val_epoch_samples = val_epoch_samples
        self._global_train_iter = None
        self._full_passes = 0
        super().__init__(model, config, train_loader, val_loader)

    def setup_model(self):
        """Skip FSDP — always single-GPU on local_rank."""
        SpeculatorModel.verify_training_compatible(self.model)
        self.model.to(self.local_rank)
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_model_state_dict(self.model)

    def _generate_batch(
        self,
        lightweight_batch: dict[str, Any],
        apply_noise: bool = True,
    ) -> BatchType:
        """Convert lightweight batch (input_ids+loss_mask) → full training batch.

        Steps:
        1. Unpack packed batch into per-sample sequences using lengths
        2. Generate hidden states via VllmHiddenStatesGenerator
        3. Standardize + shift each sample (shared with offline path)
        4. Re-collate into training batch
        """
        # 1. Unpack packed batch into per-sample sequences
        input_ids_packed = lightweight_batch["input_ids"].squeeze(0)  # [max_len]
        loss_mask_packed = lightweight_batch["loss_mask"].squeeze(0)
        lengths = lightweight_batch["lengths"]

        samples: list[dict[str, torch.Tensor]] = []
        offset = 0
        for length in lengths:
            l = length.item()
            samples.append(
                {
                    "input_ids": input_ids_packed[offset : offset + l],
                    "loss_mask": loss_mask_packed[offset : offset + l],
                }
            )
            offset += l

        # 2. Generate hidden states
        token_ids_list = [s["input_ids"].tolist() for s in samples]
        input_lens = [len(ids) for ids in token_ids_list]
        results = self.generator.generate(token_ids_list)
        result_lens = [len(r["input_ids"]) for r in results]

        # 3. Process each sample through shared pipeline
        transform = self.noise_transform if apply_noise else None
        processed: list[BatchType] = []
        consecutive_failures = 0

        for i, (result, sample) in enumerate(zip(results, samples)):
            # Validate token count (allow ±1 for BOS/EOS handling differences)
            result_len = len(result["input_ids"])
            expected_len = len(sample["input_ids"])
            if abs(result_len - expected_len) > 1:
                raise ValueError(
                    f"Sample {i}: vLLM returned {result_len} tokens, "
                    f"expected {expected_len} (diff > 1). "
                    f"Batch: {len(samples)} samples, input_lens={input_lens}, "
                    f"result_lens={result_lens}"
                )
            # Use the shorter length to align input_ids and loss_mask
            seq_len = min(result_len, expected_len)
            if result_len != expected_len:
                result["input_ids"] = result["input_ids"][:seq_len]
                result["hidden_states"] = [h[:seq_len] for h in result["hidden_states"]]
                sample["loss_mask"] = sample["loss_mask"][:seq_len]
            for j, h in enumerate(result["hidden_states"]):
                assert h.shape[0] == seq_len, (
                    f"Sample {i} layer {j}: hidden_states length {h.shape[0]} "
                    f"!= aligned length {seq_len}"
                )

            item = process_generated_sample(
                raw_data=result,
                loss_mask=sample["loss_mask"],
                standardize_fn=self.standardize_fn,
                transform=transform,
                hidden_states_dtype=self.hidden_states_dtype,
            )

            # Check for NaNs
            for key in ("hidden_states", "verifier_last_hidden_states"):
                if key in item and torch.isnan(item[key]).any():
                    root_logger.warning(
                        f"NaN detected in {key} for sample {i}, skipping"
                    )
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"{MAX_CONSECUTIVE_FAILURES} consecutive NaN failures"
                        )
                    continue

            consecutive_failures = 0
            processed.append(item)

        if not processed:
            raise RuntimeError("All samples in batch failed processing")

        # 4. Re-collate into training batch
        return self._collate_fn(processed)

    def train_epoch(self, epoch: int):
        self.model.train()

        if self._epoch_samples > 0:
            self._run_chunked_epoch(epoch)
            return

        # Create prefetcher with noise enabled
        prefetcher = DynamicBatchPrefetcher(
            dataloader=self.train_loader,
            generate_fn=lambda batch: self._generate_batch(batch, apply_noise=True),
            buffer_size=2,
        )
        prefetcher.start(epoch)

        train_iter = prefetcher
        if self.local_rank == 0:
            train_iter = tqdm(
                prefetcher,
                total=len(self.train_loader),
                desc=f"Epoch {epoch}",
            )

        try:
            for gpu_batch in train_iter:
                gpu_batch = {
                    k: v.to(self.local_rank, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in gpu_batch.items()
                }

                _draft_tokens, loss, metrics = self.model(
                    **gpu_batch, **self.config.train_call_kwargs
                )

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                current_lr = self.opt.param_groups[0]["lr"]
                if self.scheduler is not None:
                    self.scheduler.step()

                metrics = {k: v.item() for k, v in metrics.items()}
                metric_logger.info(
                    {"train": metrics, "epoch": epoch, "lr": current_lr},
                    extra={"step": self.global_step},
                )
                self.global_step += 1
        finally:
            prefetcher.stop()

    def _run_chunked_epoch(self, epoch: int) -> None:
        """Epoch-chunked: consume epoch_samples training samples from persistent iterator."""
        if self._global_train_iter is None:
            if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                self.train_loader.batch_sampler.set_epoch(self._full_passes)
            self._global_train_iter = iter(self.train_loader)
            root_logger.info("[chunked] Starting full-dataset pass #1")

        samples_done = 0
        batches_done = 0

        while samples_done < self._epoch_samples:
            try:
                lbatch = next(self._global_train_iter)
            except StopIteration:
                self._full_passes += 1
                root_logger.info(
                    "[chunked] Pass %d complete (epoch %d, %d samples so far). Restarting.",
                    self._full_passes, epoch, samples_done)
                if hasattr(self.train_loader.batch_sampler, "set_epoch"):
                    self.train_loader.batch_sampler.set_epoch(self._full_passes)
                self._global_train_iter = iter(self.train_loader)
                lbatch = next(self._global_train_iter)

            try:
                gpu_batch = self._generate_batch(lbatch, apply_noise=True)
            except Exception as exc:
                root_logger.warning("Chunked batch gen failed: %s, skipping", exc)
                continue

            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in gpu_batch.items()
            }
            _draft_tokens, loss, metrics = self.model(**gpu_batch, **self.config.train_call_kwargs)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.opt.param_groups[0]["lr"]
            metrics = {k: v.item() for k, v in metrics.items()}
            metric_logger.info(
                {"train": metrics, "epoch": epoch, "lr": current_lr},
                extra={"step": self.global_step},
            )
            self.global_step += 1

            n = lbatch["lengths"].shape[0]
            samples_done += n
            batches_done += 1
            if batches_done % 500 == 0:
                pct = 100.0 * samples_done / self._epoch_samples
                root_logger.info(
                    "[chunked] Epoch %d: %d/%d samples (%.1f%%), pass #%d, lr=%.2e",
                    epoch, samples_done, self._epoch_samples, pct,
                    self._full_passes + 1, current_lr)

        root_logger.info(
            "[chunked] Epoch %d done: %d samples, %d batches, pass #%d",
            epoch, samples_done, batches_done, self._full_passes + 1)

    @torch.no_grad()
    def val_epoch(self, epoch: int):
        if self.val_loader is None:
            return
        self.model.eval()

        # Validation: no noise, no prefetching (simpler, less memory)
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)

        val_loader = self.val_loader
        if self.local_rank == 0:
            val_loader = tqdm(val_loader, desc=f"Val {epoch}")

        val_metrics: dict[str, float] = {}
        num_batches = 0
        val_samples_done = 0

        for batch in val_loader:
            try:
                gpu_batch = self._generate_batch(batch, apply_noise=False)
            except Exception as e:
                root_logger.warning(f"Val batch generation failed: {e}, skipping")
                continue

            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in gpu_batch.items()
            }

            _draft_tokens, _loss, metrics = self.model(
                **gpu_batch, **self.config.val_call_kwargs
            )

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v.item()
            num_batches += 1
            val_samples_done += batch["lengths"].shape[0] if "lengths" in batch else 1
            if self._val_epoch_samples > 0 and val_samples_done >= self._val_epoch_samples:
                break

        if num_batches > 0:
            val_metrics = {
                f"{k}_epoch": v / num_batches for k, v in val_metrics.items()
            }
            metric_logger.info(
                {"val": val_metrics, "epoch": epoch},
                extra={"step": self.global_step},
            )
