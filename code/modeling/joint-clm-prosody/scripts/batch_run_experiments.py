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

TIME = '1-00:00:00'
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
PARTITION = 'gpuq'
GPU_INFO = '--gres=gpu:1'
NODE_LIST = ''#--nodelist=a03,a04'
ACCOUNT = 'dbic'

if __name__ == "__main__":

	experiment_name = 'helsinki_prosody' #helsinki_prosody
 
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-d', '--dataset', type=str)
	# parser.add_argument('-o', '--overwrite', type=int, default=0)
	# p = parser.parse_args()

	experiments = {
	
		# # no prosody embeddings
		# 'scratch-gpt2_clm-loss_no-prosody-embed': [
		# 	f"model.loss_mode=clm",
		# 	f"model.pretrained=False",
		# 	f"model.use_prosody_embeddings=False"
		# ],

		# # prosody embeddings
		# 'scratch-gpt2_clm-loss_prosody-embed': [
		# 	f"model.loss_mode=clm",
		# 	f"model.pretrained=False",
		# 	f"model.use_prosody_embeddings=True"
		# ],

		# # joint prosody embeddings
		# 'scratch-gpt2_joint-loss_prosody-embed': [
		# 	f"model.loss_mode=joint",
		# 	f"model.pretrained=False",
		# 	f"model.use_prosody_embeddings=True",
		# 	f"model.loss_kwargs.w_prosody=0.3"
		# ],

		# ####### Yoke the shuffling to the batch ##########

		# # joint prosody embeddings
		# 'yoked-shuffle_scratch-gpt2_joint-loss_prosody-embed': [
		# 	f"model.loss_mode=joint",
		# 	f"model.pretrained=False",
		# 	f"model.use_prosody_embeddings=True",
		# 	f"model.loss_kwargs.w_prosody=0.3",
		# 	f"data.shuffle_labels_yoked=True"
		# ],


		# # joint prosody embeddings
		# 'yoked-shuffle_scratch-gpt2_clm-loss_prosody-embed': [
		# 	f"model.loss_mode=clm",
		# 	f"model.pretrained=False",
		# 	f"model.use_prosody_embeddings=True",
		# 	f"data.shuffle_labels_yoked=True"
		# ],
		
		# ####### Shuffle labels randomly each batch ##########

		# # joint prosody embeddings
		# 'label-shuffle_scratch-gpt2_joint-loss_prosody-embed': [
		# 	f"model.loss_mode=joint",
		# 	f"model.pretrained=False",
		# 	f"model.use_prosody_embeddings=True",
		# 	f"model.loss_kwargs.w_prosody=0.3",
		# 	f"model.shuffle_prosody=True"
		# ],

		# 'label-shuffle_scratch-gpt2_clm-loss_prosody-embed': [
		# 	f"model.loss_mode=clm",
		# 	f"model.pretrained=False",
		# 	f"model.shuffle_prosody=True"
		# ],

		####### Random number draw each batch ##########

		# joint prosody embeddings
		'random-uniform-label_scratch-gpt2_joint-loss_prosody-embed': [
			f"model.loss_mode=joint",
			f"model.pretrained=False",
			f"model.use_prosody_embeddings=True",
			f"model.loss_kwargs.w_prosody=0.3",
			f"model.random_prosody=True"
		],

		'random-uniform-label_scratch-gpt2_clm-loss_prosody-embed': [
			f"model.loss_mode=clm",
			f"model.pretrained=False",
			f"model.random_prosody=True"
		],

		# # no prosody embeddings
		# 'finetune-gpt2_clm-loss_no-prosody-embed': [
		# 	f"model.loss_mode=clm",
		# 	f"model.pretrained=True",
		# 	f"model.use_prosody_embeddings=False"
		# ],

		# # add prosody embeddings
		# 'finetune-gpt2_clm-loss': [
		# 	f"model.loss_mode=clm",
		# 	f"model.pretrained=True",
		# 	f"model.use_prosody_embeddings=True"
		# ],

		# # joint loss without prosody embeddings
		# 'finetune-gpt2_shifted-joint-gamma-loss_no-prosody-embed_wprosody-0.3': [
		# 	f"model.loss_mode=joint",
		# 	f"model.pretrained=True",
		# 	f"model.use_prosody_embeddings=False",
		# 	f"model.loss_kwargs.w_prosody=0.3"
		# ], 

		# # joint loss with prosody embeddings
		# 'finetune-gpt2_shifted-joint-gamma-loss_wprosody-0.3': [
		# 	f"model.loss_mode=joint",
		# 	f"model.pretrained=True",
		# 	f"model.use_prosody_embeddings=True",
		# 	f"model.loss_kwargs.w_prosody=0.3"
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

	model = 'gpt2'
	model_name = "gpt2" #meta-llama/Llama-2-7b-hf"

	wait_secs = 20

	for i, (name, experiment) in enumerate(experiments.items()):

		name = name.replace('gpt2', model)

		experiment += [
			f"model.model_name={model_name}",
			f"data.model_name={model_name}"
		]

		cmd = f"sleep {i*wait_secs}; {job_string} experiment={experiment_name}.yaml {' '.join(experiment)} logger.wandb.name={name}"
		all_cmds.append(cmd)
		job_num += 1

		# break

	if not all_cmds:
		print (f'No model needing extraction - overwrite if you want to redo extraction', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(joblist_dir, f'prosody_experiments.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_run_experiments'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(job_num))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --nodes={N_NODES} {GPU_INFO} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
