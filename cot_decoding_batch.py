import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
from PIL import Image
import requests
import numpy as np

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

model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompts = [
            "USER: <image>\nIs the door not open and the not man crouched? Answer yes or no.\nASSISTANT:",
            "USER: <image>\nIs the banana unpeeled and the light colored plate? Answer yes or no.\nASSISTANT:",
        ]

image1 = Image.open('data/prerelease_bow/images/2375361.jpg')
image2 = Image.open('data/prerelease_bow/images/2410049.jpg')

inputs = processor(prompts, images=[image1, image2], return_tensors="pt", padding=True).to('cuda', torch.float16)

# prompt = "USER: <image>\nIs the door not open and the not man crouched? Answer yes or no.\nASSISTANT:"
# raw_image = Image.open('data/prerelease_bow/images/2410049.jpg')
# inputs = processor(prompt, raw_image, return_tensors="pt").to('cuda', torch.float16)
batch=2
# Generate branching responses
responses, response_probs = generate_branching_responses(model, processor, inputs, num_branches=10, max_length=1024, batch=batch)

# Print responses and scores
# print('Prompt:', prompt)

for i in range(batch):
    print('Prompt:', prompts[i])
    for k, response, prob in zip(range(len(responses[i])), responses[i], response_probs[i]):
        
        print(f'\nResponse k={k}:\n\n', response)
        print('\nScore:', prob)
    