import os
import pickle
import collections
import re
import random

import numpy as np
import torch

string_classes = str
import PIL


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def save_scores(scores, args):
	"""
	Utility function to save retrieval scores.
	"""
	if args.extra_info is None:
		si2t_file = os.path.join(args.output_dir, "scores",
								 f"{args.dataset}--{args.text_perturb_fn}--{args.image_perturb_fn}--{args.model_name}--{args.seed}--si2t.pkl")
		st2i_file = os.path.join(args.output_dir, "scores",
								 f"{args.dataset}--{args.text_perturb_fn}--{args.image_perturb_fn}--{args.model_name}--{args.seed}--st2i.pkl")
	else:
		si2t_file = os.path.join(args.output_dir, "scores",
								 f"{args.dataset}--{args.text_perturb_fn}--{args.image_perturb_fn}--{args.model_name}--{args.seed}--{args.extra_info}--si2t.pkl")
		st2i_file = os.path.join(args.output_dir, "scores",
								 f"{args.dataset}--{args.text_perturb_fn}--{args.image_perturb_fn}--{args.model_name}--{args.seed}--{args.extra_info}--st2i.pkl")

	if isinstance(scores, tuple):
		scores_i2t = scores[0]
		scores_t2i = scores[1].T  # Make it N_ims x N_text
	else:
		scores_t2i = scores
		scores_i2t = scores

	os.makedirs(os.path.dirname(si2t_file), exist_ok=True)
	pickle.dump(scores_i2t, open(si2t_file, "wb"))
	pickle.dump(scores_t2i, open(st2i_file, "wb"))
