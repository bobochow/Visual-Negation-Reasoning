import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import numpy as np
from dataclasses import dataclass, field
import argparse
import copy
import re
from torch.utils.data import Dataset,DataLoader

from visnr.datasets import get_dataset
from visnr import set_seed, save_scores, datasets
from visnr.conversation import conv_templates
from visnr.constants import DEFAULT_IMAGE_TOKEN


from tqdm import tqdm
from typing import List, Tuple
import math
from PIL import Image
import json
import shortuuid

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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

class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        idx = line["question_id"]
        gt = line["GT"]
        
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        
        return idx, qs, image, gt

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    idx, qs, image = zip(*batch)
    
    return list(idx), list(qs), list(image), list(gt)

# DataLoader
def create_data_loader(questions, image_folder, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    
    
    model_name = get_model_name_from_path(args.model_path)
    
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path,torch_dtype=torch.float16,device_map=args.device_map)

    processor = AutoProcessor.from_pretrained(args.model_path, pad_token="<pad>")
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
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
    
    
    ans_file = open(answers_file, "w")
    
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, batch_size=args.batch_size)

    for (idx_list, qs_list, images_list, gt_list) in tqdm(data_loader, desc="LLaVA MME Benchmark Evaluating"):
        cot_outputs=[]
        cot_prompts=[]
        
        prompts=[]
        for i, opt in enumerate(qs_list):
            
            if args.cot_type == None:
                qs_ =  DEFAULT_IMAGE_TOKEN + f"\n{opt}\n"
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
                qs_ =  cot_prompts[i] + cot_outputs[i] + f"\n{conv.roles[0]}: {opt} \n{conv.roles[1]}:"
            prompts.append(qs_)
            
        
        inputs = processor(prompts, images=images_list, return_tensors="pt", padding=True).to(dtype=torch.float16, device=args.device, non_blocking=True)
        
        responses, response_probs = generate_branching_responses(model, processor, inputs, num_branches=args.num_branches, max_length=args.max_new_tokens, batch=len(qs_list))
        
        
        for i,(idx, cur_prompt, output, gt) in enumerate(zip(idx_list, qs_list, responses, gt_list)):
            # print('Prompt:', prompts[i])
            pos_score = 0.0
            neg_score = 0.0
            ans_id = shortuuid.uuid()
            for k, response, prob in zip(range(len(responses[i])), responses[i], response_probs[i]):
                
                # print(f'\nResponse k={k}:\n\n', response)
                # print('\nScore:', prob)
                if 'yes' in response.lower().strip():
                    pos_score += prob
                elif re.search(r'\bno\b', response, re.IGNORECASE):
                    neg_score += prob
            if pos_score > neg_score:
                ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": 'Yes',
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "GT": gt,
                                   "metadata": {}}) + "\n")
            else:
                ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": 'No',
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "GT": gt,
                                   "metadata": {}}) + "\n")
    
    ans_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image-folder", type=str, default="data/MME_Benchmark_release_version")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--device_map", default="auto", type=str)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    
    parser.add_argument("--question-file", type=str, default="llava_eval/MME/llava_mme_gt.jsonl")
    parser.add_argument("--answers-file", type=str, default="llava_eval/MME/answers/llava-1.5-7b-hf-cot-decoding.jsonl")
    
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    
    parser.add_argument("--cot_type", type=str, default=None)
    
    parser.add_argument("--num_branches", type=int, default=5)
    
    args = parser.parse_args()

    eval_model(args)