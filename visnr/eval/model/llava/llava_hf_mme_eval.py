import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import copy

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init
# from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from transformers import AutoProcessor, LlavaForConditionalGeneration
from visnr.conversation import conv_templates
from visnr.constants import DEFAULT_IMAGE_TOKEN

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


# Custom dataset class
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
    idx, qs, image, gt = zip(*batch)
    
    return list(idx), list(qs), list(image), list(gt)


# DataLoader
def create_data_loader(questions, image_folder, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
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
        if args.cot_type in {'cot','DNeg','SG'}:
            cot_prompts=[]
            for i in range(len(qs_list)):
                
                if args.cot_type == 'cot':
                    qs_ =  DEFAULT_IMAGE_TOKEN + '\n' +  cot
                elif args.cot_type in{'DNeg','SG'}:
                    qs_ =  DEFAULT_IMAGE_TOKEN + '\n' + qs_list[i] + '\n' + cot
                cot_conv = copy.deepcopy(conv)
                cot_conv.append_message(conv.roles[0], qs_)
                cot_conv.append_message(conv.roles[1], None)
                cot_prompt = cot_conv.get_prompt()
                cot_prompts.append(cot_prompt)
            
            with torch.inference_mode():
                cot_inputs = processor(cot_prompts, images=images_list, return_tensors="pt", padding=True).to(dtype=torch.float16, device=args.device, non_blocking=True)
                cot_output_ids = model.generate(
                    **cot_inputs,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
            
            input_token_len = cot_inputs['input_ids'].shape[1]
            cot_outputs = processor.batch_decode(cot_output_ids[:, input_token_len:], skip_special_tokens=True)
        
        prompts=[]
        for i, opt in enumerate(qs_list):
            
            if args.cot_type == None:
                qs_ =  DEFAULT_IMAGE_TOKEN + f"\n{opt}\n"
                sys_conv = copy.deepcopy(conv)
                sys_conv.append_message(conv.roles[0], qs_)
                sys_conv.append_message(conv.roles[1], None)
                qs_ = sys_conv.get_prompt()
            elif args.cot_type == 'hint':
                qs_ =  DEFAULT_IMAGE_TOKEN + f"\n{opt} {cot}\n"
                sys_conv = copy.deepcopy(conv)
                sys_conv.append_message(conv.roles[0], qs_)
                sys_conv.append_message(conv.roles[1], None)
                qs_ = sys_conv.get_prompt()
            else:
                qs_ =  cot_prompts[i] + cot_outputs[i] + f"\n{conv.roles[0]}: {opt} \n{conv.roles[1]}:"
            prompts.append(qs_)
        
        assert len(prompts) == len(images_list)
        
        inputs = processor(prompts, images=images_list, return_tensors="pt", padding=True).to(dtype=torch.float16, device=args.device, non_blocking=True)
        
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        
        input_token_len = inputs['input_ids'].shape[1]
        
        outputs = processor.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        
        for idx, cur_prompt, output, gt in zip(idx_list, qs_list, outputs, gt_list):
            output = output.strip()
            # print(f'{prompt}\n')
            # print(f'{output}\n\n\n')
            
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": output,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "GT": gt,
                                   "metadata": {}}) + "\n")
        #     conv_output.write(f'{prompt}\n')
        #     conv_output.write(f'{output}\n\n')

        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image-folder", type=str, default="data/MME_Benchmark_release_version")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--device_map",  type=str, default="auto")
    parser.add_argument("--torch_dtype",  type=torch.dtype, default=torch.float16)
    parser.add_argument("--question-file", type=str, default="llava_eval/MME/llava_mme.jsonl")
    parser.add_argument("--answers-file", type=str, default="llava_eval/MME/answers/test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--cot_type", type=str, default=None)
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()

    eval_model(args)