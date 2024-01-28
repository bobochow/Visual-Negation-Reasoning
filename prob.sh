export CUDA_VISIBLE_DEVICES=2

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances 64 --output_dir ./outputs/llava_1.5_7b_hf/test  --extra_info test --cot_type DNeg --device cuda:2

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/zeroshot_cot_SG  --extra_info zeroshot_cot_SG --cot_type SG 

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/zeroshot_cot_DNeg  --extra_info zeroshot_cot_DNeg --cot_type DNeg 

python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/zeroshot_cot_hint  --extra_info zeroshot_cot_hint --cot_type hint

# python blip2_hf_infer_batched.py --model-path Salesforce/blip2-flan-t5-xxl --model_name blip2 --batch_size 16 --max_instances -1 --output_dir ./outputs/blip2-flan-t5-xxl-hf/zeroshot --extra_info zeroshot 

# CUDA_VISIBLE_DEVICES=1 python blip2_hf_infer_batched.py --model-path Salesforce/blip2-flan-t5-xxl --model_name blip2 --batch_size 8 --max_instances -1 --output_dir ./outputs/blip2-flan-t5-xxl-hf/zeroshot_cot_content --extra_info zeroshot_cot_content --cot_type cot 

