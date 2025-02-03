import sys, os
import glob
import argparse
import subprocess

from config import *
import utils

PARTITION='preemptable'
TIME = '24:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
GPU_INFO = ''

TIME = '2-12:00:00'
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
PARTITION = 'a100'
GPU_INFO = '--gres=gpu:1'
NODE_LIST = ''#--nodelist=a03,a04'
ACCOUNT = 'test_a100' #dbic

DATASET_INFO = {
	'gigaspeech-m': [
		'data.dataset_name=gigaspeech-m',
		'data.data_dir=\${paths.data_dir}/gigaspeech/m',
		'data.cache_dir=\${paths.cache_dir}/nlp-datasets/gigaspeech/m',
	],
	'libritts-r': [
		'data.dataset_name=libritts-r',
		'data.data_dir=\${paths.data_dir}/libritts-r',
		'data.cache_dir=\${paths.cache_dir}/nlp-datasets/libritts-r',
	],
	'tedlium': [
		'data.dataset_name=tedlium',
		'data.data_dir=\${paths.data_dir}/tedlium',
		'data.cache_dir=\${paths.cache_dir}/nlp-datasets/tedlium',
	],
	'peoples-speech': [
		'data.dataset_name=peoples-speech',
		'data.data_dir=\${paths.data_dir}/peoples-speech',
		'data.cache_dir=\${paths.cache_dir}/nlp-datasets/peoples-speech',
	]
}

if __name__ == "__main__":

	EXPERIMENT_NAME = 'careful_whisper'
	MODEL_CONFIG_NAME = '' #'_batch-32_lr-'
 
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', type=str)
	# parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()

	MODEL_CONFIGS = {
	
		# General GPT2-esque model
		'careful-whisper_no-xattn': [
			f"model.config.cross_attention=False",
			f"model.config.use_causal_cross_attention=False",
		],

		# Whisper w/ CLM integration
		'careful-whisper_causal-xattn': [
			f"model.config.cross_attention=True",
			f"model.config.use_causal_cross_attention=True",

			# Add in dropout and position embedding
			f"model.config.context_embed_dropout=0.1",
			f"model.config.context_pos_embed=True",
		],

		# # Whisper w/ CLM integration
		# 'careful-whisper_bi-xattn': [
		# 	f"model.config.bidirectional_cross_attention=True",
		# 	f"model.config.use_causal_cross_attention=False",
		# ],

		# Whisper w/ CLM integration
		'prosody-whisper_causal-xattn': [
			f"model.config.cross_attention=True",
			f"model.config.use_causal_cross_attention=True",

			# Prosody embedding information
			f"model.config.context_type=prominence",
			f"model.config.context_dim=1",
			f"model.config.context_embed_dropout=0.1",
			f"model.config.context_pos_embed=True",

		],


		# # Whisper w/ CLM integration
		# 'careful-whisper_no-xattn_parameter-control': [
		# 	f"model.config.num_layers=18",
		# ],

		# # Whisper w/ CLM integration
		# 'careful-whisper_causal-bi-xattn': [
		# 	f"model.config.bidirectional_cross_attention=True",
		# 	f"model.config.use_causal_cross_attention=True",
		# ],

		# # Matched architecture to GPT2 (same embed dim + num_heads)
		# 'careful-whisper_gpt2-control': [
		# 	f"model.config.embed_dim=768",
		# 	f"model.config.num_heads=12",
		# 	f"model.config.cross_attention=False",
		# 	f"model.config.use_causal_cross_attention=False",
		# ],

		# # Whisper w/ CLM integration --> Double-attend to text / no audio
		# 'careful-whisper_causal-xattn_text-control': [
		# 	f"model.config.cross_attention=True",
		# 	f"model.config.use_causal_cross_attention=True",
		# 	f"model.config.use_text_control=True"
		# ],

		# # Whisper-type model (e.g., attending ahead)
		# 'careful-whisper_xattn': [
		# 	f"model.config.cross_attention=True",
		# 	f"model.config.use_causal_cross_attention=False",
		# ],

	}

	# make directories
	dsq_dir = os.path.join(SUBMIT_DIR, 'dsq')
	joblist_dir = os.path.join(SUBMIT_DIR, 'joblists')
	logs_dir = os.path.join(LOGS_DIR)

	utils.attempt_makedirs(dsq_dir)
	utils.attempt_makedirs(joblist_dir)
	utils.attempt_makedirs(logs_dir)

	all_cmds = []
	script_fn = os.path.join(BASE_DIR, 'src/train.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'
	job_num = 0

	# Dataset overrides --> set up dataset
	dataset_config = ' '.join(DATASET_INFO[p.dataset])

	for model_name, model_config in MODEL_CONFIGS.items():

		# Model config --> change model configurations
		model_name += MODEL_CONFIG_NAME
		model_config = ' '.join(model_config)
		
		# Logger overrides --> change the name based on the dataset
		wandb_config = (
			f"logger.wandb.project={p.dataset}-audio "
			f"logger.wandb.name={model_name}"
		)

		hydra_config = (
			"hydra.run.dir=\${paths.log_dir}/\${task_name}/careful-whisper/"
			f"{p.dataset}/"
			f"{model_name}/"
		)

		cmd = (
			f"{job_string} "
			f"experiment={EXPERIMENT_NAME}.yaml "
			f"{model_config} "
			f"{dataset_config} "
			f"{wandb_config} "
			f"{hydra_config} "
		)

		all_cmds.append(cmd)
		job_num += 1

		# break

	if not all_cmds:
		print (f'No model needing extraction - overwrite if you want to redo extraction', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(joblist_dir, f'{p.dataset}_careful_whisper_experiments.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_{p.dataset}_careful_whisper_experiments'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(job_num)) 

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --account={ACCOUNT} --nodes={N_NODES} {GPU_INFO} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
