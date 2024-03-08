import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import numpy as np
from typing import Optional

from transformers import (
            LogitsProcessorList,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TemperatureLogitsWarper,
            StoppingCriteriaList,
        )

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


model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# # instantiate logits processors
# logits_processor = LogitsProcessorList(
#             [
#                 MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
#             ]
#         )

# # instantiate logits processors
logits_warper = LogitsProcessorList(
            [
                TopPLogitsWarper(1.0),
                TemperatureLogitsWarper(0.5),
            ]
)

# stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
torch.manual_seed(0)

prompts = [
            "USER: <image>\nIs the door open and the man crouched? \nASSISTANT:",
            "USER: <image>\nIs the banana unpeeled and the light colored plate? \nASSISTANT:",
            "USER: <image>\nIs the umbrella blue and the grass green? \nASSISTANT:",
        ]

neg_prompts = [
            "USER: <image>\nIs the door not open and the man not crouched? \nASSISTANT:",
            "USER: <image>\nIs the banana not unpeeled and the light not colored plate? \nASSISTANT:",
            "USER: <image>\nIs the umbrella not blue and the grass not green? \nASSISTANT:",
        ]

image1 = Image.open('data/prerelease_bow/images/2410049.jpg')
image2 = Image.open('data/prerelease_bow/images/2375361.jpg')
image3 = Image.open('data/prerelease_bow/images/2358252.jpg')


inputs = processor(prompts, images=[image1, image2, image3], return_tensors="pt", padding=True).to('cuda', torch.float16)
neg_inputs = processor(neg_prompts, images=[image1, image2, image3], return_tensors="pt", padding=True).to('cuda', torch.float16)
outputs, neg_outputs = ncd_sample(model, processor, inputs, neg_inputs, use_ncd=False ,cd_alpha=0.9, cd_beta=0.1,max_length=500, batch=3, logits_warper=logits_warper)


responses = processor.batch_decode(outputs, skip_special_tokens=True)

neg_responses = processor.batch_decode(neg_outputs, skip_special_tokens=True)

for prompt, rep in zip(prompts,responses):
    print(f"{prompt}\n")
    print(f"{rep}\n")
    # print(f"score: {scores}\n")
    print("\n------------\n\n")

for prompt, rep in zip(neg_prompts,neg_responses):
    print(f"{prompt}\n")
    print(f"{rep}\n")
    # print(f"score: {scores}\n")
    print("\n------------\n\n")

