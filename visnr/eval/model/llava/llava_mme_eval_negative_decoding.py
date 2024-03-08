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
import re
from torch.utils.data import Dataset,DataLoader

from visnr.datasets import get_dataset
from visnr import set_seed, save_scores, datasets
from visnr.conversation import conv_templates
from visnr.constants import DEFAULT_IMAGE_TOKEN


from tqdm import tqdm
from typing import List, Tuple, Optional
import math
from PIL import Image
import json
import shortuuid

from transformers import (
            LogitsProcessorList,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TemperatureLogitsWarper,
            StoppingCriteriaList,
        )

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def ncd_sample(model, processor, inputs, neg_inputs, use_ncd=False ,cd_alpha=0.5, cd_beta=0.1,max_length=500, batch=1,
                logits_processor: Optional[LogitsProcessorList] = None,
                stopping_criteria: Optional[StoppingCriteriaList] = None,
                logits_warper: Optional[LogitsProcessorList] = None,):
    
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    
    unfinished_sequences = torch.ones(inputs['input_ids'].shape[0], dtype=torch.long, device=inputs['input_ids'].device)
    
    unfinished_sequences_cd = torch.ones(neg_inputs['input_ids'].shape[0], dtype=torch.long, device=neg_inputs['input_ids'].device)
    
    pad_token_id = processor.tokenizer.pad_token_id
    
    eos_token_id_tensor = torch.tensor([processor.tokenizer.eos_token_id]).to(inputs['input_ids'].device)
    
    response = [[] for _ in range(batch)]
    
    neg_response = [[] for _ in range(batch)]
    
    # scores = ()
    
    # neg_scores = ()
    
    for i in range(max_length):
        
        with torch.no_grad():
            outputs = model(**inputs,return_dict=True)
            next_token_logits = outputs.logits[:, -1, :] # batch, seq_len, vocab_size
        
        with torch.no_grad():
                neg_outputs = model(**neg_inputs,return_dict=True)
                next_token_logits_cd = neg_outputs.logits[:, -1, :] # batch, seq_len, vocab_size
        
        if use_ncd:
            
            next_token_scores_cd = logits_processor(neg_inputs['input_ids'], next_token_logits_cd)
            next_token_scores_cd = logits_warper(neg_inputs['input_ids'], next_token_scores_cd)
            
            # neg_scores += (next_token_scores_cd,)
            
            probs_cd = nn.functional.softmax(next_token_scores_cd, dim=-1)
            next_tokens_cd = torch.multinomial(probs_cd, num_samples=1).squeeze(1)
            
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            # cd_logits = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(inputs['input_ids'], cd_logits)
            cd_logits = logits_warper(inputs['input_ids'], cd_logits)

            next_token_scores = cd_logits
            
            # scores += (next_token_scores,)
            
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
            
            
        else:
            next_token_scores = logits_processor(inputs['input_ids'], next_token_logits)
            next_token_scores = logits_warper(inputs['input_ids'], next_token_scores)
            # scores += (next_token_scores,)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            next_token_scores_cd = logits_processor(neg_inputs['input_ids'], next_token_logits_cd)
            next_token_scores_cd = logits_warper(neg_inputs['input_ids'], next_token_scores_cd)
            # neg_scores += (next_token_scores_cd,)
            probs_cd = nn.functional.softmax(next_token_scores_cd, dim=-1)
            next_tokens_cd = torch.multinomial(probs_cd, num_samples=1).squeeze(1)

        # finished sentences should have their next token be a padding token
        next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        next_tokens_cd = next_tokens_cd * unfinished_sequences_cd + pad_token_id * (1 - unfinished_sequences_cd)
        
        
        # update generated ids, model inputs, and length for next step
        inputs['input_ids'] = torch.cat([inputs['input_ids'], next_tokens[:, None]], dim=1)
        inputs['attention_mask'] = torch.cat(
                    [inputs['attention_mask'], inputs['attention_mask'].new_ones((inputs['attention_mask'].shape[0], 1))], dim=-1
                )
        
        # if use_ncd:
        neg_inputs['input_ids'] = torch.cat([neg_inputs['input_ids'], next_tokens_cd[:, None]], dim=1)
        neg_inputs['attention_mask'] = torch.cat(
                [neg_inputs['attention_mask'], neg_inputs['attention_mask'].new_ones((neg_inputs['attention_mask'].shape[0], 1))], dim=-1
            )
            
        
        for i, token in enumerate(next_tokens.tolist()):
            if unfinished_sequences[i] == 1:
                response[i].append(token)
        
        for i, token in enumerate(next_tokens_cd.tolist()):
            if unfinished_sequences_cd[i] == 1:
                neg_response[i].append(token)
        
        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
        
        unfinished_sequences_cd = unfinished_sequences_cd.mul(
                    next_tokens_cd.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )
        
        # stop when each sentence is finished
        if unfinished_sequences.max() == 0 and unfinished_sequences_cd.max() == 0:
            break
        
    # return response, neg_response, scores, neg_scores
    return response, neg_response

class CustomDataset(Dataset):
    def __init__(self, questions, neg_questions,image_folder):
        self.questions = questions
        self.neg_questions = neg_questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        idx = line["question_id"]
        gt = line["answer"]
        
        neg_qs=self.neg_questions[index]["text"]
        
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        
        return idx, qs, neg_qs, image

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    idx, qs, neg_qs, image, gt = zip(*batch)
    
    return list(idx), list(qs), list(neg_qs), list(image), list(gt)

# DataLoader
def create_data_loader(questions, neg_questions, image_folder, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, neg_questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def eval_model(args):
    
    
    model_name = get_model_name_from_path(args.model_path)
    
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path,torch_dtype=torch.float16,device_map=args.device_map)

    processor = AutoProcessor.from_pretrained(args.model_path, pad_token="<pad>")
    
    logits_warper = LogitsProcessorList(
            [
                TopPLogitsWarper(args.top_p),
                TemperatureLogitsWarper(args.temperature),
            ]
        )
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    neg_questions = [json.loads(q) for q in open(os.path.expanduser(args.neg_question_file), "r")]
    
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
    
    data_loader = create_data_loader(questions, neg_questions, args.image_folder, batch_size=args.batch_size)

    for (idx_list, qs_list, neg_qs_list, images_list, gt_list) in tqdm(data_loader, desc="LLaVA MME Benchmark Evaluating"):
        cot_outputs=[]
        cot_prompts=[]
        
        prompts=[]
        neg_prompts=[]
        for i, opt in enumerate(qs_list):
            
            if args.cot_type == None:
                qs_ =  DEFAULT_IMAGE_TOKEN + f"\n{opt}\n"
                sys_conv = copy.deepcopy(conv)
                sys_conv.append_message(conv.roles[0], qs_)
                sys_conv.append_message(conv.roles[1], None)
                qs_ = sys_conv.get_prompt()
                
                neg_qs_ =  DEFAULT_IMAGE_TOKEN + f"\n{neg_qs_list[i]}\n"
                neg_sys_conv = copy.deepcopy(conv)
                neg_sys_conv.append_message(conv.roles[0], neg_qs_)
                neg_sys_conv.append_message(conv.roles[1], None)
                neg_qs_ = neg_sys_conv.get_prompt()
                
            elif args.cot_type == 'hint':
                qs_ =  DEFAULT_IMAGE_TOKEN + f"\n{opt} {cot}"
                sys_conv = copy.deepcopy(conv)
                sys_conv.append_message(conv.roles[0], qs_)
                sys_conv.append_message(conv.roles[1], None)
                qs_ = sys_conv.get_prompt()
            else:
                qs_ =  cot_prompts[i] + cot_outputs[i] + f"\n{conv.roles[0]}: {opt} \n{conv.roles[1]}:"
            prompts.append(qs_)
            neg_prompts.append(neg_qs_)
            
        
        inputs = processor(prompts, images=images_list, return_tensors="pt", padding=True).to(dtype=torch.float16, device=args.device, non_blocking=True)
        neg_inputs = processor(neg_prompts, images=images_list, return_tensors="pt", padding=True).to(dtype=torch.float16, device=args.device, non_blocking=True)
        responses, neg_responses = ncd_sample(model, processor, inputs, neg_inputs, use_ncd=args.use_ncd ,cd_alpha=args.cd_alpha, cd_beta=args.cd_beta,max_length=args.max_new_tokens, batch=args.batch, logits_warper=logits_warper)

        outputs = processor.batch_decode(responses, skip_special_tokens=True)
        
        for i,(idx, cur_prompt, output, gt) in enumerate(zip(idx_list, qs_list, outputs, gt_list)):
            # print('Prompt:', prompts[i])
            ans_id = shortuuid.uuid()
            
            ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "text": output,
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
    parser.add_argument("--neg-question-file", type=str, default="llava_eval/MME/llava_mme_neg.jsonl")
    parser.add_argument("--answers-file", type=str, default="llava_eval/MME/answers/llava-1.5-7b-hf-cot-decoding.jsonl")
    
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    
    parser.add_argument("--cot_type", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    
    parser.add_argument("--use_ncd", type=bool, default=True)
    parser.add_argument("--cd_alpha", type=float, default=0.5)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    
    args = parser.parse_args()

    eval_model(args)