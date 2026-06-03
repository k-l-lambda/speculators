from speculators.train.checkpointer import SingleGPUCheckpointer


def test_previous_epoch_ignores_incomplete_numeric_checkpoint_dirs(tmp_path):
    (tmp_path / "0").mkdir()
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "model.safetensors").write_bytes(b"")

    checkpointer = SingleGPUCheckpointer(tmp_path)

    assert checkpointer.previous_epoch == 1


def test_previous_epoch_returns_minus_one_for_only_incomplete_dirs(tmp_path):
    (tmp_path / "0").mkdir()

    checkpointer = SingleGPUCheckpointer(tmp_path)

    assert checkpointer.previous_epoch == -1
