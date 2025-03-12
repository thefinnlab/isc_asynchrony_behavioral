import sys
import os
import glob
import json

DATASET_CONFIG = {
	'lrs3': {
		'ckpt_path': "token-fusion_wav2vec-data2vec_representation-loss/checkpoints/epoch_012.ckpt",
		# MSE loss
		#"token-fusion_wav2vec-data2vec/checkpoints/epoch_013.ckpt",
		},
}

def attempt_makedirs(d):

	if not os.path.exists(d):
		try:
			os.makedirs(d)
		except Exception:
			pass