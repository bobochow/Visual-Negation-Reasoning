export CUDA_VISIBLE_DEVICES=1

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances 64 --output_dir ./outputs/llava_1.5_7b_hf/test  --extra_info test --cot_type DNeg 

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/zeroshot_cot_SG  --extra_info zeroshot_cot_SG --cot_type SG 

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/zeroshot_cot_DNeg  --extra_info zeroshot_cot_DNeg --cot_type DNeg 

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/zeroshot_cot_hint  --extra_info zeroshot_cot_hint --cot_type hint

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 16 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/zeroshot  --extra_info zeroshot

# python blip2_hf_infer_batched.py --model-path Salesforce/blip2-flan-t5-xxl --model_name blip2 --batch_size 16 --max_instances -1 --output_dir ./outputs/blip2-flan-t5-xxl-hf/zeroshot --extra_info zeroshot --seed 2

# python blip2_hf_infer_batched.py --model-path Salesforce/blip2-flan-t5-xxl --model_name blip2 --batch_size 8 --max_instances -1 --output_dir ./outputs/blip2-flan-t5-xxl-hf/zeroshot_cot_content --extra_info zeroshot_cot_content --cot_type cot 

# python blip_hf_infer_batched.py --model-path Salesforce/blip-vqa-capfilt-large --model_name blip --batch_size 16 --max_instances -1 --output_dir ./outputs/blip-vqa-capfilt-large/zeroshot --extra_info zeroshot

# python instructblip_hf_infer_batched.py --model-path Salesforce/instructblip-flan-t5-xxl --model_name instructblip --batch_size 16 --max_instances -1 --output_dir .outputs/instructblip-flan-t5-xxl/zeroshot_cot_content --extra_info zeroshot_cot_content --cot_type cot

# python instructblip_hf_infer_batched.py --model-path Salesforce/instructblip-flan-t5-xxl --model_name instructblip --batch_size 16 --max_instances -1 --output_dir .outputs/instructblip-flan-t5-xxl/zeroshot --extra_info zeroshot

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-13b-hf --batch_size 8 --max_instances -1 --output_dir ./outputs/llava-1.5-13b-hf/zeroshot  --extra_info zeroshot;

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-13b-hf --batch_size 4 --max_instances -1 --output_dir ./outputs/llava-1.5-13b-hf/zeroshot_cot_content  --extra_info zeroshot_cot_content --cot_type cot 

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-13b-hf --batch_size 4 --max_instances -1 --output_dir ./outputs/llava-1.5-13b-hf/zeroshot_cot_SG  --extra_info zeroshot_cot_SG --cot_type SG

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 16 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/subclausal/zeroshot  --extra_info subclausal_zeroshot --subclausal

# python llava_hf_infer_batched.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances -1 --output_dir ./outputs/llava_1.5_7b_hf/subclausal/zeroshot_cot  --extra_info subclausal_zeroshot_cot --subclausal --cot_type cot


# CUDA_VISIBLE_DEVICES=2 python llava_cot_decoding_batch.py --model-path llava-hf/llava-1.5-7b-hf --batch_size 8 --max_instances 32 --output_dir ./outputs/llava_1.5_7b_hf/zeroshot_cot_decoding  --extra_info subclausal_zeroshot_cot --num_branches 10

# python instructblip_hf_infer_batched.py --model-path Salesforce/instructblip-flan-t5-xxl --model_name instructblip --batch_size 16 --max_instances -1 --output_dir ./outputs/instructblip-flan-t5-xxl/subclausal/zeroshot --extra_info subclausal_zeroshot --subclausal;

# python instructblip_hf_infer_batched.py --model-path Salesforce/instructblip-flan-t5-xxl --model_name instructblip --batch_size 16 --max_instances -1 --output_dir ./outputs/instructblip-flan-t5-xxl/subclausal/zeroshot_cot_content --extra_info subclausal_zeroshot_cot --cot_type cot --subclausal;

# python blip2_hf_infer_batched.py --model-path Salesforce/blip2-flan-t5-xxl --model_name blip2 --batch_size 16 --max_instances -1 --output_dir ./outputs/blip2-flan-t5-xxl-hf/subclausal/zeroshot --extra_info subclausal_zeroshot --subclausal;

# python blip2_hf_infer_batched.py --model-path Salesforce/blip2-flan-t5-xxl --model_name blip2 --batch_size 8 --max_instances -1 --output_dir ./outputs/blip2-flan-t5-xxl-hf/subclausal/zeroshot_cot_content --extra_info subclausal_zeroshot_cot --cot_type cot --subclausal;

python blip_hf_infer_batched.py --model-path Salesforce/blip-vqa-capfilt-large --model_name blip --batch_size 16 --max_instances -1 --output_dir ./outputs/blip-vqa-capfilt-large/subclausal_zeroshot --extra_info subclausal_zeroshot --subclausal


