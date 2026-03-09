import sys
from pathlib import Path

# Add scripts directory to path so we can import the run_e2e function.
scripts_path = Path(__file__).absolute().parent.parent.parent / "scripts"
sys.path.append(str(scripts_path))

from gen_and_train import (
    DataGenArgs,
    TrainArgs,
    VocabMappingArgs,
    run_e2e,
)

### Quick test run for Qwen3-8B on 1k samples from ShareGPT ###
# Reduced from the full example for faster iteration

if __name__ == "__main__":
    VERIFIER_NAME_OR_PATH = "Qwen/Qwen3-8B"
    OUTPUT_PATH = "./output/qwen3_8b_sharegpt_1k_test"
    TOTAL_SEQ_LEN = 4096  # Reduced from 8192 for speed

    # Data Generation - just 1000 samples
    data_gen_args_sharegpt = DataGenArgs(
        train_data_path="sharegpt",
        seq_length=TOTAL_SEQ_LEN,
        max_samples=1000,
        turn_dropout=True,
    )

    # Vocab Mapping
    vocab_mapping_args = VocabMappingArgs(
        draft_vocab_size=32000,
        target_vocab_size=151936,  # Qwen3-8B vocab size
    )

    # Training - fewer epochs for test
    train_args = TrainArgs(
        logger="tensorboard",
        lr=3e-5,
        total_seq_len=TOTAL_SEQ_LEN,
        run_name="qwen3_8b_sharegpt_1k_test",
        epochs=3,
    )

    run_e2e(
        verifier_name_or_path=VERIFIER_NAME_OR_PATH,
        output_path=OUTPUT_PATH,
        data_gen_args=data_gen_args_sharegpt,
        vocab_mapping_args=vocab_mapping_args,
        train_args=train_args,
    )
