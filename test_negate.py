# from negate import Negator
# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# # Use default model (en_core_web_md):
# # negator = Negator()

# # Use a Transformer model (en_core_web_trf):
# # GPU can also be used (if available):
# negator = Negator(use_transformers=True, use_gpu=True)

# sentence = "The image shows a python code. Is the output of the code '7'?\nAnswer the question using a single word or phrase."

# negated_sentence = negator.negate_sentence(sentence)

# print(negated_sentence)  # "An apple a day, doesn't keep the doctor away."

# from nltk.corpus import wordnet as wn

# print(wn.synsets('dog'))

import json
import os

path = '/home/SNARE/data/POPE/coco/coco_pope_random_neg1.json'
questions = [json.loads(q) for q in open(os.path.expanduser(path), "r")]

# ans_file = open('llava_eval/MME/llava_mme_gt.jsonl', "w")



for i in range(len(questions)):
    
    questions[i]["label"] = 'yes' if questions[i]["label"] == 'no' else 'no'


# 更新Python对象的值


# 将更新后的Python对象转换为JSON字符串
json_data = json.dumps(questions)

# 将JSON字符串写回文件（如果需要）
with open(path, 'w') as file:
    file.write(json_data)
    
# import re

# pattern = r'\bno\b'

# test_string = "No, not now."

# matches = re.search(pattern, test_string, re.IGNORECASE)

# print(matches)


