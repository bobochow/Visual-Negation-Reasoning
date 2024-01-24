import argparse
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass, field

from torch.utils.data import Dataset,DataLoader
import torch

from einops import rearrange

from snare.models import get_model
from snare.datasets_zoo import  get_dataset
from snare import set_seed, save_scores, datasets_zoo

from tqdm import tqdm
from typing import List, Tuple

import math

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration



@dataclass
class DataCollatorForVisualTextGeneration(object):

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        images, prompt_list = zip(*batch)

        prompt= [list(x) for x in zip(*prompt_list)]
        
        return images, prompt



def main(args):
    set_seed(args.seed)
    
    datasets_zoo.COCO_ROOT = os.path.join(args.data_path, "coco")
    datasets_zoo.FLICKR_ROOT = os.path.join(args.data_path, "flickr30k")
    datasets_zoo.CASSP_ROOT = os.path.join(args.data_path, "prerelease_bow")

    model = LlavaForConditionalGeneration.from_pretrained(args.model_path,device_map=args.device)

    processor = AutoProcessor.from_pretrained(args.model_path, pad_token="<pad>")
    
    dataset = get_dataset(args.dataset, image_preprocess=None, download=args.download,max_instances=args.max_instances)

    collator = DataCollatorForVisualTextGeneration()
    data_loader = DataLoader(dataset, collate_fn=collator, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    scores=[]
    
    if args.extra_info is None:
        conv_output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name}_seed-{args.seed}.txt")
    else:
        conv_output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_name}_seed-{args.seed}_{args.extra_info}.txt")
    os.makedirs(args.output_dir) if not os.path.exists(args.output_dir) else None
    conv_output = open(conv_output_file, "a")
    
    for images, prompts in tqdm(data_loader):
        
        prompts=rearrange(prompts,'(b n) l -> n b l',n=2)
        
        batch_scores = []
        for prompt in prompts:
            score=[]
            
            inputs = processor(prompt, images=images, return_tensors="pt", padding=True)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True)
            
            input_token_len = output_ids.shape[1]
            
            outputs = processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            
            for output in outputs:
                output = output.lower().strip()
                # print(f'{output}\n')
                
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
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--data_path", default="/home/SNARE/data", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_name", default="llava", choices=["blip2", "llava"], type=str)
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--max_instances", type=int, default=16)
    parser.add_argument("--cot_type", type=str, default='cot')
    return parser.parse_args()

if __name__ == "__main__":
    args = config()
    main(args)
