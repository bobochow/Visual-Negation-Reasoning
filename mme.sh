#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

CUDA_LAUNCH_BLOCKING=1 python -m llava_mme_eval_cot_decoding_batch \
    --model-path llava-hf/llava-1.5-7b-hf \
    --question-file llava_eval/MME/llava_mme.jsonl \
    --image-folder data/MME_Benchmark_release_version \
    --answers-file llava_eval/MME/answers/llava-1.5-7b-hf-cot-decode.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --batch_size 4 \
    --max_new_tokens 128

# cd llava_eval/MME

# python convert_answer_to_mme.py --experiment llava-1.5-7b-hf

# cd eval_tool

# python calculation.py --results_dir answers/llava-1.5-7b-hf