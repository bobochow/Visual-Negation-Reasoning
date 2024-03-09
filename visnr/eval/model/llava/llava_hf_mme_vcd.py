import argparse
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from tqdm import tqdm
import shortuuid
import copy

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from transformers import AutoProcessor, LlavaForConditionalGeneration, set_seed

from visnr.conversation import conv_templates
from visnr.constants import DEFAULT_IMAGE_TOKEN
from visnr.decoding_utils.vcd_decoding import add_diffusion_noise

# from visnr import set_seed
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
        
        prompt = DEFAULT_IMAGE_TOKEN + '\n' + qs 
        conv=conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        
        return idx, prompt, image, gt, qs

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    idx, prompt, image, gt, qs = zip(*batch)
    return list(idx), list(prompt), list(image), list(gt), list(qs)


# DataLoader
def create_data_loader(questions, image_folder, batch_size=1, num_workers=4):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path,torch_dtype=torch.float16,device_map=args.device_map)

    processor = AutoProcessor.from_pretrained(args.model_path, pad_token="<pad>")
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, batch_size=args.batch_size)

    for (idx_list, prompt_list, images_list, gt_list, qs_list) in tqdm(data_loader, desc="LLaVA MME Benchmark Evaluating"):
        
        
        inputs = processor(prompt_list, images=images_list, return_tensors="pt", padding=True).to(dtype=torch.float16, device=args.device, non_blocking=True)

        images_cd = inputs['pixel_values'].contiguous()

        for i in range(len(images_cd)):
            images_cd[i] = add_diffusion_noise(images_cd[i], args.noise_step)
        
        
        
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                images_cd=images_cd,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                
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
                                    "model_id": 'llava',
                                    "GT": gt,
                                    "metadata": {}}) + "\n")

        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--image-folder", type=str, default="data/MME_Benchmark_release_version")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--device_map",  type=str, default="auto")
    parser.add_argument("--question-file", type=str, default="visnr/eval/results/mme/llava_mme_gt.jsonl")
    parser.add_argument("--answers-file", type=str, default="visnr/eval/results/mme/answers/test.jsonl")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", default=8, type=int)
    
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)