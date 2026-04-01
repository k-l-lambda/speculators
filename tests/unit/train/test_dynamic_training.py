"""Unit tests for dynamic hidden states training components.

Tests cover:
- process_generated_sample() preprocessing parity
- DynamicEagle3Dataset contract
- create_dynamic_collate_fn() packing behavior
- DynamicBatchPrefetcher thread/queue lifecycle
- _generate_batch() validation and skip logic
"""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from speculators.train.data import (
    DynamicEagle3Dataset,
    create_collate_fn,
    create_dynamic_collate_fn,
    process_generated_sample,
    shift_batch,
    standardize_data_v1,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def make_raw_sample(seq_len: int, hidden_size: int = 8, n_layers: int = 4):
    """Create a synthetic raw sample matching VllmHiddenStatesGenerator output."""
    return {
        "input_ids": torch.arange(seq_len, dtype=torch.long),
        "hidden_states": [
            torch.randn(seq_len, hidden_size) for _ in range(n_layers)
        ],
        "loss_mask": None,
    }


def make_loss_mask(seq_len: int):
    """Create a simple loss mask with first token masked out."""
    mask = torch.ones(seq_len, dtype=torch.long)
    mask[0] = 0
    return mask


# ── process_generated_sample() tests ─────────────────────────────────────

class TestProcessGeneratedSample:
    """Tests for the shared preprocessing function."""

    def test_basic_output_format(self):
        """Verify output has expected keys and shapes after processing."""
        seq_len, H = 6, 8
        raw = make_raw_sample(seq_len, H, n_layers=4)
        loss_mask = make_loss_mask(seq_len)

        result = process_generated_sample(
            raw_data=raw,
            loss_mask=loss_mask,
            standardize_fn=standardize_data_v1,
        )

        # After shift_batch: seq_len - 1
        expected_len = seq_len - 1
        assert result["input_ids"].shape == (expected_len,)
        assert result["hidden_states"].shape == (expected_len, 3 * H)
        assert result["verifier_last_hidden_states"].shape == (expected_len, H)
        assert result["loss_mask"].shape == (expected_len,)
        assert result["lengths"].shape == (1,)
        assert result["lengths"].item() == expected_len
        assert result["position_ids"].shape == (expected_len,)

    def test_matches_offline_path(self):
        """Verify process_generated_sample matches the offline Eagle3SampleFileDataset path."""
        seq_len, H = 10, 4
        raw = make_raw_sample(seq_len, H, n_layers=4)
        loss_mask = make_loss_mask(seq_len)

        # Dynamic path
        dynamic_result = process_generated_sample(
            raw_data=raw,
            loss_mask=loss_mask,
            standardize_fn=standardize_data_v1,
            hidden_states_dtype=torch.float,
        )

        # Manual offline path (same as Eagle3SampleFileDataset.__getitem__)
        data = {
            "input_ids": raw["input_ids"].clone(),
            "hidden_states": [h.clone() for h in raw["hidden_states"]],
            "loss_mask": loss_mask.clone(),
        }
        data = standardize_data_v1(data)
        data = {
            k: v.to(torch.float) if "hidden_states" in k else v
            for k, v in data.items()
        }
        data["lengths"] = torch.tensor([data["input_ids"].shape[0]], dtype=torch.long)
        data["position_ids"] = torch.arange(data["input_ids"].shape[0], dtype=torch.long)
        offline_result = shift_batch(data)

        # Compare all keys
        for key in offline_result:
            assert key in dynamic_result, f"Missing key: {key}"
            torch.testing.assert_close(
                dynamic_result[key], offline_result[key],
                msg=f"Mismatch on key '{key}'",
            )

    def test_loss_mask_truncation(self):
        """Loss mask longer than input_ids is truncated to input_ids length."""
        seq_len, H = 5, 4
        raw = make_raw_sample(seq_len, H, n_layers=4)
        long_mask = torch.ones(seq_len + 10, dtype=torch.long)

        result = process_generated_sample(
            raw_data=raw, loss_mask=long_mask,
        )
        # After shift: seq_len - 1
        assert result["loss_mask"].shape[0] == seq_len - 1

    def test_all_zero_loss_mask(self):
        """All-zero loss mask should not crash."""
        seq_len, H = 5, 4
        raw = make_raw_sample(seq_len, H, n_layers=4)
        zero_mask = torch.zeros(seq_len, dtype=torch.long)

        result = process_generated_sample(
            raw_data=raw, loss_mask=zero_mask,
        )
        assert (result["loss_mask"] == 0).all()

    def test_single_token_sample(self):
        """Single token sample: after shift, seq_len becomes 0 — should not crash."""
        raw = make_raw_sample(1, 4, n_layers=4)
        loss_mask = torch.ones(1, dtype=torch.long)

        result = process_generated_sample(
            raw_data=raw, loss_mask=loss_mask,
        )
        # After shift_batch on seq_len=1 → seq_len=0
        assert result["input_ids"].shape[0] == 0

    def test_noise_transform_applied(self):
        """Noise transform modifies hidden states."""
        seq_len, H = 6, 8
        raw = make_raw_sample(seq_len, H, n_layers=4)
        loss_mask = make_loss_mask(seq_len)

        # Without noise
        result_clean = process_generated_sample(
            raw_data={
                "input_ids": raw["input_ids"].clone(),
                "hidden_states": [h.clone() for h in raw["hidden_states"]],
                "loss_mask": None,
            },
            loss_mask=loss_mask.clone(),
            transform=None,
        )

        # With noise (mock transform that adds 1.0)
        class AddOne:
            def __call__(self, data):
                data["hidden_states"] = data["hidden_states"] + 1.0
                return data

        result_noisy = process_generated_sample(
            raw_data={
                "input_ids": raw["input_ids"].clone(),
                "hidden_states": [h.clone() for h in raw["hidden_states"]],
                "loss_mask": None,
            },
            loss_mask=loss_mask.clone(),
            transform=AddOne(),
        )

        diff = (result_noisy["hidden_states"] - result_clean["hidden_states"]).abs()
        assert diff.mean().item() > 0.5  # Transform was applied


# ── DynamicEagle3Dataset tests ───────────────────────────────────────────

class TestDynamicEagle3Dataset:
    """Tests for the lightweight dynamic dataset."""

    def _make_hf_dataset(self, samples):
        """Create a minimal HF-like dataset from list of dicts."""
        from datasets import Dataset
        return Dataset.from_dict({
            "input_ids": [s["input_ids"] for s in samples],
            "loss_mask": [s["loss_mask"] for s in samples],
        })

    def test_returns_lightweight_fields_only(self):
        """Dataset returns only input_ids, loss_mask, lengths — no hidden states."""
        hf = self._make_hf_dataset([
            {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]},
        ])
        ds = DynamicEagle3Dataset(hf, max_len=100)

        item = ds[0]
        assert set(item.keys()) == {"input_ids", "loss_mask", "lengths"}
        assert item["input_ids"].dtype == torch.long
        assert item["loss_mask"].dtype == torch.long
        assert item["lengths"].item() == 4

    def test_truncates_to_max_len(self):
        """Samples longer than max_len are truncated."""
        hf = self._make_hf_dataset([
            {"input_ids": list(range(20)), "loss_mask": [1] * 20},
        ])
        ds = DynamicEagle3Dataset(hf, max_len=5)

        item = ds[0]
        assert item["input_ids"].shape[0] == 5
        assert item["loss_mask"].shape[0] == 5
        assert item["lengths"].item() == 5

    def test_approx_lengths(self):
        """approx_lengths reflects capped sequence lengths."""
        hf = self._make_hf_dataset([
            {"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1]},
            {"input_ids": list(range(100)), "loss_mask": [1] * 100},
        ])
        ds = DynamicEagle3Dataset(hf, max_len=10)

        assert ds.approx_lengths == [3, 10]


# ── create_dynamic_collate_fn() tests ────────────────────────────────────

class TestDynamicCollateFn:
    """Tests for lightweight collation."""

    def test_packs_variable_length_samples(self):
        """Multiple samples of different lengths are packed and padded."""
        max_len = 10
        collate = create_dynamic_collate_fn(max_len)

        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "loss_mask": torch.tensor([0, 1, 1], dtype=torch.long),
                "lengths": torch.tensor([3], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([4, 5], dtype=torch.long),
                "loss_mask": torch.tensor([1, 1], dtype=torch.long),
                "lengths": torch.tensor([2], dtype=torch.long),
            },
        ]

        result = collate(batch)

        # Packed and padded to [1, max_len]
        assert result["input_ids"].shape == (1, max_len)
        assert result["loss_mask"].shape == (1, max_len)
        # First 5 tokens should be the concatenated samples
        assert result["input_ids"][0, :5].tolist() == [1, 2, 3, 4, 5]
        # Lengths should be preserved
        assert result["lengths"].tolist() == [3, 2]

    def test_truncates_when_exceeding_max_len(self):
        """When total tokens exceed max_len, lengths are truncated."""
        max_len = 5
        collate = create_dynamic_collate_fn(max_len)

        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
                "loss_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "lengths": torch.tensor([3], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7], dtype=torch.long),
                "loss_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                "lengths": torch.tensor([4], dtype=torch.long),
            },
        ]

        result = collate(batch)
        # Total is 7 > max_len=5, so second sample truncated to 2
        assert result["lengths"].sum().item() <= max_len
        assert result["lengths"].tolist() == [3, 2]

    def test_single_sample_batch(self):
        """Single sample batch should work without squeeze issues."""
        max_len = 8
        collate = create_dynamic_collate_fn(max_len)

        batch = [
            {
                "input_ids": torch.tensor([10, 20, 30], dtype=torch.long),
                "loss_mask": torch.tensor([1, 0, 1], dtype=torch.long),
                "lengths": torch.tensor([3], dtype=torch.long),
            },
        ]

        result = collate(batch)
        assert result["input_ids"].shape == (1, max_len)
        assert result["lengths"].tolist() == [3]


# ── DynamicBatchPrefetcher tests ─────────────────────────────────────────

class TestDynamicBatchPrefetcher:
    """Tests for the background prefetch thread."""

    def test_yields_batches_in_order(self):
        """Prefetcher yields batches in the order produced by the dataloader."""
        from speculators.train.dynamic_trainer import DynamicBatchPrefetcher

        # Fake dataloader: list of batches
        batches = [{"id": i} for i in range(5)]
        fake_loader = batches
        fake_loader_obj = MagicMock()
        fake_loader_obj.__iter__ = MagicMock(return_value=iter(batches))
        fake_loader_obj.__len__ = MagicMock(return_value=len(batches))
        fake_loader_obj.batch_sampler = MagicMock(spec=[])  # no set_epoch

        # generate_fn is identity
        prefetcher = DynamicBatchPrefetcher(
            dataloader=fake_loader_obj,
            generate_fn=lambda b: b,
            buffer_size=2,
        )
        prefetcher.start(epoch=0)

        results = list(prefetcher)
        prefetcher.stop()

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r["id"] == i

    def test_handles_generator_exception(self):
        """Prefetcher skips batches where generate_fn raises."""
        from speculators.train.dynamic_trainer import DynamicBatchPrefetcher

        batches = [{"id": i} for i in range(4)]
        fake_loader = MagicMock()
        fake_loader.__iter__ = MagicMock(return_value=iter(batches))
        fake_loader.__len__ = MagicMock(return_value=4)
        fake_loader.batch_sampler = MagicMock(spec=[])

        call_count = 0

        def flaky_generate(batch):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated generation failure")
            return batch

        prefetcher = DynamicBatchPrefetcher(
            dataloader=fake_loader,
            generate_fn=flaky_generate,
            buffer_size=2,
        )
        prefetcher.start(epoch=0)

        results = list(prefetcher)
        prefetcher.stop()

        # 4 batches, 1 failed → 3 valid results
        assert len(results) == 3
        assert prefetcher._skipped == 1

    def test_stop_does_not_deadlock(self):
        """Calling stop() while producer is running should not deadlock."""
        from speculators.train.dynamic_trainer import DynamicBatchPrefetcher

        # Slow generator to ensure producer is still running when we stop
        def slow_generate(batch):
            time.sleep(0.1)
            return batch

        batches = [{"id": i} for i in range(20)]
        fake_loader = MagicMock()
        fake_loader.__iter__ = MagicMock(return_value=iter(batches))
        fake_loader.__len__ = MagicMock(return_value=20)
        fake_loader.batch_sampler = MagicMock(spec=[])

        prefetcher = DynamicBatchPrefetcher(
            dataloader=fake_loader,
            generate_fn=slow_generate,
            buffer_size=1,
        )
        prefetcher.start(epoch=0)

        # Consume just 1 batch then stop
        next(prefetcher)
        prefetcher.stop()

        # Should reach here without deadlock
        assert prefetcher._thread is None

    def test_timeout_on_hung_producer(self):
        """Consumer raises RuntimeError if queue is empty for too long."""
        from speculators.train.dynamic_trainer import DynamicBatchPrefetcher

        # Empty dataloader that never produces anything
        fake_loader = MagicMock()
        fake_loader.__iter__ = MagicMock(return_value=iter([]))
        fake_loader.__len__ = MagicMock(return_value=0)
        fake_loader.batch_sampler = MagicMock(spec=[])

        prefetcher = DynamicBatchPrefetcher(
            dataloader=fake_loader,
            generate_fn=lambda b: b,
            buffer_size=2,
        )
        prefetcher.start(epoch=0)

        # Should get StopIteration (empty dataloader → sentinel immediately)
        results = list(prefetcher)
        prefetcher.stop()
        assert results == []

    def test_skip_rate_abort(self):
        """Prefetcher aborts if >50% of batches fail after 10+ attempts."""
        from speculators.train.dynamic_trainer import DynamicBatchPrefetcher

        batches = [{"id": i} for i in range(20)]
        fake_loader = MagicMock()
        fake_loader.__iter__ = MagicMock(return_value=iter(batches))
        fake_loader.__len__ = MagicMock(return_value=20)
        fake_loader.batch_sampler = MagicMock(spec=[])

        # Fail 80% of batches
        def mostly_failing(batch):
            if batch["id"] % 5 == 0:
                return batch
            raise RuntimeError("fail")

        prefetcher = DynamicBatchPrefetcher(
            dataloader=fake_loader,
            generate_fn=mostly_failing,
            buffer_size=2,
        )
        prefetcher.start(epoch=0)

        with pytest.raises(RuntimeError, match="Too many batch failures"):
            list(prefetcher)
        prefetcher.stop()


# ── Collate parity test ──────────────────────────────────────────────────

class TestCollateParity:
    """Test that dynamic collate + generate aligns with offline collate."""

    def test_collated_batch_keys_match_offline(self):
        """Dynamic pipeline produces same batch keys as offline pipeline."""
        max_len = 16
        H = 4
        seq_len = 6

        # Simulate what _generate_batch produces after process_generated_sample
        raw = make_raw_sample(seq_len, H, n_layers=4)
        loss_mask = make_loss_mask(seq_len)

        processed = process_generated_sample(
            raw_data=raw, loss_mask=loss_mask,
            standardize_fn=standardize_data_v1,
        )

        # Collate with create_collate_fn (same as used by DynamicTrainer)
        collate = create_collate_fn(max_len)
        batch = collate([processed])

        expected_keys = {
            "hidden_states", "input_ids", "verifier_last_hidden_states",
            "loss_mask", "lengths", "position_ids",
        }
        assert set(batch.keys()) == expected_keys
        # Batch dimension added
        assert batch["input_ids"].dim() == 2
        assert batch["hidden_states"].dim() == 3
