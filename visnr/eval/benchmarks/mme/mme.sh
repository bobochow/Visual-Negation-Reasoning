#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

model_path=llava-hf/llava-1.5-7b-hf
question_file=llava_eval/MME/llava_mme_gt.jsonl
experiment=llava-1.5-7b-hf_greedy_t0
answers_file=llava_eval/MME/answers/${experiment}.jsonl

python -m llava_hf_mme_eval \
    --model-path $model_path \
    --question-file $question_file \
    --image-folder data/MME_Benchmark_release_version \
    --answers-file $answers_file \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --batch_size 16 \


cd llava_eval/MME

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

python calculation.py --results_dir answers/${experiment}