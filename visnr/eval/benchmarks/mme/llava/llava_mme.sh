export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/Visual-Negation-Reasoning
seed=${1:-42}

# model_name=llava-1.5-7b-hf
model_name=llava-1.5-13b-hf

model_path=llava-hf/${model_name}

image_folder=data/MME_Benchmark_release_version

temperature=0

neg=false

if [[ $neg == false ]]; then
    question_file=visnr/eval/results/mme/llava_mme_gt.jsonl
    experiment=${model_name}-greedy-t${temperature}-seed${seed}
else
    question_file=visnr/eval/results/mme/llava_mme_neg.jsonl
    experiment=NEG-${model_name}-greedy-t${temperature}-seed${seed}
fi

answers_file=visnr/eval/results/mme/answers/${experiment}.jsonl

echo "MME Experiment: $experiment"

python visnr/eval/model/llava/llava_hf_mme_eval.py \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --image-folder ${image_folder} \
    --answers-file  ${answers_file} \
    --seed ${seed} \
    --temperature ${temperature} \
    --batch_size 8

cd visnr/eval/results/mme

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

python calculation.py --results_dir answers/${experiment}
