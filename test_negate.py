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

questions = [json.loads(q) for q in open(os.path.expanduser('llava_eval/MME/llava_mme.jsonl'), "r")]

ans_file = open('llava_eval/MME/llava_mme_gt.jsonl', "a")



for i in range(len(questions)):
    
    ans_file.write(json.dumps({"question_id": questions[i]["question_id"],
                               "image": questions[i]["image"],
                                   "text": questions[i]["text"],
                                   "category": questions[i]["category"],
                                   "GT": 'Yes' if i%2 == 0 else 'No',
                                   }) + "\n")