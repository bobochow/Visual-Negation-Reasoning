import os
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import numpy as np
from dataclasses import dataclass, field
import argparse
import copy
import pandas as pd

from torch.utils.data import Dataset,DataLoader

from visnr.datasets import get_dataset
from visnr import set_seed, save_scores, datasets
from visnr.conversation import conv_templates
from visnr.constants import DEFAULT_IMAGE_TOKEN


from tqdm import tqdm
from typing import List, Tuple

from PIL import Image

# Get our initial top k tokens
def get_topk_tokens(model, inputs, num_branches=10):
        
    # Generate logits for the next token after the prompt 
    with torch.no_grad():
        outputs = model(**inputs,return_dict=True)
        next_token_logits = outputs.logits[:, -1, :] # batch, seq_len, vocab_size
    
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(next_token_logits, dim=-1)

    # Get the top k tokens and their probabilities
    topk_values, topk_indicies = torch.topk(probabilities, num_branches) # batch, k

    return topk_values, topk_indicies


# Generate a full response from the model and log the difference in probabilities between the top two tokens
def generate_response(model, processor, inputs, max_length=1024, batch=1):

    # Create variables to store our response and each token's probabilities
    # response = []
    response = [[] for _ in range(batch)]
    # response_probs = []
    response_probs = [[] for _ in range(batch)]
    
    unfinished_sequences = torch.ones(inputs['input_ids'].shape[0], dtype=torch.long, device=inputs['input_ids'].device)
    
    pad_token_id = processor.tokenizer.pad_token_id
    
    eos_token_id_tensor = torch.tensor([processor.tokenizer.eos_token_id]).to(inputs['input_ids'].device)
    
    # Loop through the max length of the response
    for i in range(max_length):

        # Generate the logits for the next token
        topk_values, topk_indices = get_topk_tokens(model, inputs, num_branches=2)

        # Get the difference in probabilities between the top two tokens
        prob_diff = topk_values[:, 0] - topk_values[:, 1]
        
        # response_probs.append(prob_diff.item())  # Convert tensor to scalar

        next_tokens = topk_indices[:, 0] * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        for i, p in enumerate(prob_diff.cpu().numpy().tolist()):
            response_probs[i].append(p)
        
        # Append the most likely token to the response
        # response.append(topk_indices[:, 0])
        for i, indices in enumerate(topk_indices):
            response[i].append(indices[0])

        # Stop if this token is the end of sequence token        
        unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
        # stop when each sentence is finished
        if unfinished_sequences.max() == 0:
            break

        # Add the token to the input for the next iteration
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_tokens[:, None]], dim=1)
        inputs['attention_mask'] = torch.cat(
                    [inputs['attention_mask'], inputs['attention_mask'].new_ones((inputs['attention_mask'].shape[0], 1))], dim=-1
                )

    return inputs['input_ids'], response_probs

# Generate all branching responses
def generate_branching_responses(model, processor, inputs, num_branches=10, max_length=500, batch=1):

    # First we tokenize the prompt
    # inputs = tokenizer(prompt, return_tensors="pt")
    input_token_len = inputs['input_ids'].shape[1]

    # Get our initial top k tokens
    _, topk_indices = get_topk_tokens(model, inputs, num_branches) # batch, k

    # Create a list to store our responses and each token's probabilities
    # responses = []
    responses = [[] for _ in range(batch)]
    # response_probs = []
    response_probs = [[] for _ in range(batch)]
    for k in tqdm(range(num_branches)):
        
        # Add the kth most likely token to this new branch
        new_input_ids = inputs.copy()
        topk_token= topk_indices[:, k].unsqueeze(-1)
        new_input_ids['input_ids'] = torch.cat([inputs['input_ids'], topk_token], dim=1)
        new_input_ids['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones(batch,1).to('cuda',dtype=torch.int64)], dim=1)
        # Generate a response and log the difference in probabilities between the top two tokens
        response, probs = generate_response(model, processor, new_input_ids, max_length, batch)
        
        # Append the response to our list
        # responses.append(processor.batch_decode(response))
        # responses.append(processor.batch_decode(response[:, input_token_len:]))
        outputs = processor.batch_decode(response[:, input_token_len:], skip_special_tokens=True)
        for i, r in enumerate(outputs):
            responses[i].append(r)

        # Determine the average difference in probabilities for this response
        # response_probs.append(sum(probs) / len(probs))
        for i, p in enumerate(probs):
            response_probs[i].append(sum(p) / len(p))

    return responses, response_probs

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
        
        return images_list, [pos_caption_list, neg_caption_list]



def main(args):
    set_seed(args.seed)
    
    datasets.COCO_ROOT = os.path.join(args.data_path, "coco")
    datasets.FLICKR_ROOT = os.path.join(args.data_path, "flickr30k")
    datasets.CASSP_ROOT = os.path.join(args.data_path, "prerelease_bow")

    model = LlavaForConditionalGeneration.from_pretrained(args.model_path,torch_dtype=torch.float16,device_map=args.device_map)

    processor = AutoProcessor.from_pretrained(args.model_path, pad_token="<pad>")

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
    
    conv=conv_templates[args.conv_mode].copy()
    
    cot = None
    if args.cot_type == 'cot':
        cot=f"Describe the image.\n"
    elif args.cot_type == 'SG':
        cot=f"Let's think step by step. For the provided image and its associated question, generate a scene graph in JSON format that includes the following:\n" \
                f'1.Objects that are relevant to answering the question.\n'    \
                f'2.Object attributes that are relevant to answering the question\n'  \
                f'3.Object relationships that are relevant to answering the question.\n'  \
                f'\nScene Graph:\n'
    elif args.cot_type == 'DNeg':
        cot=f"Let's think step by step based on the logic of double negation.\n" \
        f"Firstly, let's think if the negation form of the question is consistent with the content in the image.\n" \
        f"Then, let's think if the double negation of the question is consistent with the content in the image.\n" \
        f"Finally, let's think if the question itself is correct.\n" 
    elif args.cot_type == 'hint':
        cot=f"Note that if there is a negation in the question, you should answer the question with the opposite result of the affirmative form.\n"
    
    
    scores=[]
    for images_list, captions_list in tqdm(data_loader, desc="LLaVA Negation logic Evaluating"):
        batch_scores = []
        for captions in captions_list:
            score=[]
            cot_outputs=[]
            
            prompts=[]
            for i, opt in enumerate(captions):
                
                if args.cot_type == None:
                    qs_ =  DEFAULT_IMAGE_TOKEN + f"\n{opt}\nAnswer the question with 'yes' or 'no'.\n"
                    sys_conv = copy.deepcopy(conv)
                    sys_conv.append_message(conv.roles[0], qs_)
                    sys_conv.append_message(conv.roles[1], None)
                    qs_ = sys_conv.get_prompt()
                elif args.cot_type == 'hint':
                    qs_ =  DEFAULT_IMAGE_TOKEN + f"\n{opt} {cot}"
                    sys_conv = copy.deepcopy(conv)
                    sys_conv.append_message(conv.roles[0], qs_)
                    sys_conv.append_message(conv.roles[1], None)
                    qs_ = sys_conv.get_prompt()
                else:
                    qs_ =  cot_prompts[i] + cot_outputs[i] + f"\n{conv.roles[0]}: {opt} \nAnswer the question with 'yes' or 'no'.\n{conv.roles[1]}:"
                prompts.append(qs_)
                
            
            inputs = processor(prompts, images=images_list, return_tensors="pt", padding=True).to(dtype=torch.float16, device=args.device, non_blocking=True)
            
            responses, response_probs = generate_branching_responses(model, processor, inputs, num_branches=args.num_branches, max_length=args.max_new_tokens, batch=len(captions))
            
            
            for i in range(len(captions)):
                # print('Prompt:', prompts[i])
                pos_score = 0.0
                neg_score = 0.0
                conv_output.write(f'\nPrompt: {prompts[i]}\n')
                for k, response, prob in zip(range(len(responses[i])), responses[i], response_probs[i]):
                    
                    # print(f'\nResponse k={k}:\n\n', response)
                    # print('\nScore:', prob)
                    if 'yes' in response.lower().strip():
                        pos_score += prob
                    elif 'no' in response.lower().strip() and 'not' not in response.lower().strip():
                        neg_score += prob
                    conv_output.write(f'\nResponse k={k}:\n\n{response}')
                    conv_output.write(f'\nScore: {prob}')
                
                if pos_score > neg_score:
                    score.append(1)
                    
                else:
                    score.append(0)
                    
            
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


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--device_map", default="auto", type=str)
    parser.add_argument("--data_path", default="./data", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
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
    
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--extra_info", default=None, type=str)
    
    
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--max_instances", type=int, default=16)
    parser.add_argument("--cot_type", type=str, default=None)
    parser.add_argument("--subclausal", action="store_true",default=False)
    parser.add_argument("--num_branches", type=int, default=10)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = config()
    main(args)