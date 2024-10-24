import os, sys

sys.path.append('../utils/')

from config import *

from tommy_utils import nlp
from preproc_utils import load_model_results, divide_nwp_dataframe
import analysis_utils as utils

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# type of analysis we're running --> linked to the name of the regressors
	parser.add_argument('-task_list', '--task_list', type=str, nargs='+')
	parser.add_argument('-word_model', '--word_model', type=str, default='fasttext')
	parser.add_argument('-m', '--model_name', type=str, default='gpt2-xl')
	parser.add_argument('-window', '--window_size', type=int, default=25)
	parser.add_argument('-o', '--overwrite', type=int, default=0)
	p = parser.parse_args()
    
    ###################################
    ### Load results from task list ###
    ###################################

    # Load model names
    MLM_MODELS = list(nlp.MLM_MODELS_DICT.keys())[1:]
    CLM_MODELS = list(nlp.CLM_MODELS_DICT.keys()) 
    model_names = CLM_MODELS + MLM_MODELS

    print (f'Loading the following models')
    print (f'MLM models: {MLM_MODELS}')
    print (f'CLM models: {CLM_MODELS}')

    ###################################
    ### Load results from task list ###
    ###################################

    df_results = []

    for task in p.task_list:
        df_task = pd.read_csv(os.path.join(results_dir, f'task-{task}_group-analyzed-behavior_window-size-{window_size}_human-model-lemmatized.csv'))
        df_task['task'] = task
        df_results.append(df_task)

    # concatenate into one dataframe --> write to file for posterity 
    df_results = pd.concat(df_results).reset_index(drop=True)
    df_results.to_csv(os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{window_size}_human-model-lemmatized.csv'), index=False)

    # use the accuracy within these results to get the order to plot the models
    ordered_accuracy = utils.get_ordered_accuracy(df_results)

    # always put humans first (audio, text) then CLM models, then MLM models
    human_conditions = ['audio', 'text']
    group_order = [item for item in ordered_accuracy if item not in ['audio', 'text', *MLM_MODELS]]
    group_order = human_conditions + group_order + MLM_MODELS

    #################################################
    ### Plot 1: continuous accuracy for each task ###
    #################################################

    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    axes = axes.flatten()

    variable = f'fasttext_avg_accuracy'

    for ax, (task, df) in zip(axes, df_results.groupby('task')):
        plt.sca(ax)
        ax = utils.plot_bar_results(df, x='modality', y=variable, hue=None, cmap=cmap, figsize=None, add_points=False, order=group_order)
        ax.set_ylim([0, 1])
        ax.set_title(f'Task - {task}')
        plt.xticks(rotation=45, ha='right')

    #################################################
    ####### Plot 2: quadrant plot of accuracy #######
    #################################################

    

    #################################################
    #### Plot 3: KL Divergence of model & humans ####
    #################################################
    