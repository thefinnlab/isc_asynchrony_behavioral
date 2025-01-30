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
PARTITION = 'gpuq'
GPU_INFO = '--gres=gpu:1'
NODE_LIST = ''#--nodelist=a03,a04'
ACCOUNT = 'dbic'

if __name__ == "__main__":

	experiment_name = 'gigaspeech_audio' #helsinki_prosody
 
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-d', '--dataset', type=str)
	# parser.add_argument('-o', '--overwrite', type=int, default=0)
	# p = parser.parse_args()

	experiments = {
	
		# # General GPT2-esque model
		# 'careful-whisper_no-xattn': [
		# 	f"model.config.cross_attention=False",
		# 	f"model.config.use_causal_cross_attention=False",
		# ],

		# # Whisper w/ CLM integration
		# 'careful-whisper_causal-xattn': [
		# 	f"model.config.cross_attention=True",
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

		# Whisper-type model (e.g., attending ahead)
		'careful-whisper_xattn': [
			f"model.config.cross_attention=True",
			f"model.config.use_causal_cross_attention=False",
		],

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

	wait_secs = 20

	for i, (name, experiment) in enumerate(experiments.items()):

		cmd = f"sleep {i*wait_secs}; {job_string} experiment={experiment_name}.yaml {' '.join(experiment)} logger.wandb.name={name}"
		all_cmds.append(cmd)
		job_num += 1

		# break

	if not all_cmds:
		print (f'No model needing extraction - overwrite if you want to redo extraction', flush=True)
		sys.exit(0)

	joblist_fn = os.path.join(joblist_dir, f'careful_whisper_experiments.txt')

	with open(joblist_fn, 'w') as f:
		for cmd in all_cmds:
			f.write(f"{cmd}\n")
	
	dsq_base_string = f'dsq_careful_whisper_experiments'
	dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
	dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
	array_fmt_width = len(str(job_num))

	if not os.path.exists(dsq_out_dir):
		os.makedirs(dsq_out_dir)
	
	subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
		f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
		f"--time={TIME} --nodes={N_NODES} {GPU_INFO} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
