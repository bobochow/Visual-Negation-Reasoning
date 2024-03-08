import argparse
import copy
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass, field

from torch.utils.data import Dataset,DataLoader
import torch

from einops import rearrange

from visnr.datasets import get_dataset
from visnr import set_seed, save_scores, datasets

from tqdm import tqdm

from transformers import AutoProcessor, BlipForQuestionAnswering


@dataclass
class DataCollatorForVisualTextGeneration(object):

    def __call__(self, batch):
        images_list=[]
        pos_caption_list=[]
        neg_caption_list=[]
        
        for item in batch:
            images_list.append(item['image_options'][0])
            pos_caption_list.append(item['caption_options'][0])
            neg_caption_list.append(item['caption_options'][1])
        
        # images = np.array(images_list)
        
        return images_list, [pos_caption_list, neg_caption_list]



def main(args):
    set_seed(args.seed)
    
    datasets.COCO_ROOT = os.path.join(args.data_path, "coco")
    datasets.FLICKR_ROOT = os.path.join(args.data_path, "flickr30k")
    datasets.CASSP_ROOT = os.path.join(args.data_path, "prerelease_bow")

    model = BlipForQuestionAnswering.from_pretrained(args.model_path, torch_dtype=torch.float16).to(args.device)
    # model.config.text_config.pad_token_id = 0
    # model.config.pad_token_id = 0
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download,max_instances=args.max_instances,subclausal=args.subclausal)

    collator = DataCollatorForVisualTextGeneration()
    data_loader = DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    if args.extra_info is None:
        conv_output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name}_seed-{args.seed}.txt")
    else:
        conv_output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name}_seed-{args.seed}_{args.extra_info}.txt")
    os.makedirs(args.output_dir) if not os.path.exists(args.output_dir) else None
    
    conv_output = open(conv_output_file, 'w')
    conv_output.close()
    conv_output = open(conv_output_file, "a")
    
    scores=[]
    
    for images_list, captions_list in tqdm(data_loader, desc=f"{args.model_name} Negation logic Evaluating"):
        
        batch_scores = []
        for captions in captions_list:
            score=[]
            prompts=[]
            for opt in captions:
                qs_ =  f'\nQuestion: {opt} \nAnswer:'
                prompts.append(qs_)
            
            inputs = processor(text=prompts, images=images_list, return_tensors="pt", padding=True).to(dtype=torch.float16, device=args.device)
            
            with torch.inference_mode():
                output_ids = model.generate(**inputs)
            
            outputs = processor.batch_decode(output_ids, skip_special_tokens=True)
            
            for output, prompt in zip(outputs,prompts):
                output = output.lower().strip()
                # print(f'{prompt}\n')
                # print(f'{output}\n\n\n')
                conv_output.write(f'{prompt}\n'  )
                conv_output.write("\n" + output )
                
                if "yes" in output :
                    score.append(1)
                elif "no" in output:
                    score.append(0)
                else:
                    score.append(0)
                    print(f"There are not \"Yes\" or \"No\" in answer. \n The answer is: {output}\n")
            batch_scores.append(score)
        
        
        batch_scores_flip=[list(item) for item in zip(*batch_scores)]
        scores.append(batch_scores_flip)
    
    conv_output.close()
    
    all_scores = np.concatenate(scores, axis=0)
    result_records = dataset.evaluate_vllm_scores(all_scores)
    
    for record in result_records:
        record.update({"Model": args.model_name, "Dataset": args.dataset, "Seed": args.seed})
    if args.extra_info is None:
        output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name}_seed-{args.seed}.csv")
    else:
        output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name}_seed-{args.seed}_{args.extra_info}.csv")
    df = pd.DataFrame(result_records)
    
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)

    else:
        df.to_csv(output_file)

    if args.save_scores:
        save_scores(scores, args)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--device_map", default="auto", type=str)
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-path", type=str, default="Salesforce/blip-vqa-capfilt-large")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_name", default="blip", choices=["blip","instructblip","blip2", "llava"], type=str)
    parser.add_argument("--dataset", default="Negation_Logic", type=str,
                        choices=["Attribute_Ownership", "Subordination_Relationship",
                                 "Spatial_Relationship", "Negation_Logic","Negation_Logic_Batched",
                                 "COCO_Semantic_Structure", "Flickr30k_Semantic_Structure",
                                 "VG_Relation", "VG_Attribution", "COCO_Order", "Flickr30k_Order"])

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--text_perturb_fn", default=None, type=str,
                        help="Perturbation function to apply to the text.")
    parser.add_argument("--image_perturb_fn", default=None, type=str,
                        help="Perturbation function to apply to the images.")

    parser.add_argument("--download", action="store_true",
                        help="Download the datasets_zoo if it doesn't exist. (Default: False)")
    parser.add_argument("--save_scores", action="store_false",
                        help="Save the scores for the retrieval. (Default: True)")
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--extra_info", default=None, type=str)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_instances", type=int, default=16)
    parser.add_argument("--cot_type", type=str, default=None)
    parser.add_argument("--subclausal", action="store_true",default=False)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = config()
    main(args)
