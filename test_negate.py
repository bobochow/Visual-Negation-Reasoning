from negate import Negator
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Use default model (en_core_web_md):
# negator = Negator()

# Use a Transformer model (en_core_web_trf):
# GPU can also be used (if available):
negator = Negator(use_transformers=True, use_gpu=True)

sentence = "The image shows a python code. Is the output of the code '7'?\nAnswer the question using a single word or phrase."

negated_sentence = negator.negate_sentence(sentence)

print(negated_sentence)  # "An apple a day, doesn't keep the doctor away."
