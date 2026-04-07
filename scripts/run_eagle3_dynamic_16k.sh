#!/bin/bash
# Eagle3 dynamic training on export_3c6b3075_16k_hf dataset (276K samples, 16K context)
# Pretrain weights: eagle3_v2_apilog/7 (epoch 7, val acc 71.4%, 32K draft vocab)
# NOTE: NVIDIA Eagle3 weights are incompatible (163840 vs 32000 draft vocab)
# Virtual epoch size: 5000 samples; keep latest 3 checkpoints
# Run inside dynamic-train-test container on youyun.37
# Restarted: 2026-04-07 CST

PYTORCH_ALLOC_CONF=expandable_segments:True python3 -u /data/speculators-dev/scripts/train.py     --dynamic     --verifier-name-or-path /data/models/Kimi-K2.5     --target-model-path /data/models/Kimi-K2.5     --train-data-path /data/datasets/export_3c6b3075_16k_hf/train     --seq-length 16384 --total-seq-len 16384     --gpu-memory-utilization 0.3 --tensor-parallel-size 8     --layer-ids 2 30 58 60 --epochs 10 --lr 1e-4     --draft-arch kimi_k2     --d2t-path /data/datasets/apilog_k25_eagle3/vocab_mapping/d2t.npy     --t2d-path /data/datasets/apilog_k25_eagle3/vocab_mapping/t2d.npy     --save-path /data/training/eagle3_dynamic_16k     --noise-std 0.05 --norm-before-residual     --pretrain-weights /data/training/eagle3_v2_apilog/7/model.safetensors     --no-resume-from-checkpoint     --ttt-steps 1 --scheduler-type cosine --scheduler-warmup-steps 200     --epoch-samples 1000 --max-checkpoints 3 --enforce-eager --loss-type ce     "$@"
