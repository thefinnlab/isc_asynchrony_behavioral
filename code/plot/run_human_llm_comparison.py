import os, sys
import argparse
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

sys.path.append('../utils/')

from config import *

from tommy_utils import nlp
from dataset_utils import attempt_makedirs
import plotting_utils as utils

FIGSIZE = (6,5)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-task_list', '--task_list', type=str, nargs='+')
    parser.add_argument('-word_model', '--word_model', type=str, default='fasttext')
    parser.add_argument('-m', '--model_name', type=str, default='gpt2-xl')
    parser.add_argument('-window', '--window_size', type=int, default=25)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    preproc_dir = os.path.join(BASE_DIR, 'stimuli/preprocessed')
    models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions')

    results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
    plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final/human-llm-comparison/')

    attempt_makedirs(plots_dir)

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
        df_task = pd.read_csv(os.path.join(results_dir, f'task-{task}_group-analyzed-behavior_window-size-{p.window_size}_human-model-lemmatized.csv'))
        df_task['task'] = task
        df_results.append(df_task)

    # concatenate into one dataframe --> write to file for posterity 
    df_results = pd.concat(df_results).reset_index(drop=True)
    df_results.to_csv(os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{p.window_size}_human-model-lemmatized.csv'), index=False)

    # use the accuracy within these results to get the order to plot the models
    ordered_accuracy = utils.get_ordered_accuracy(df_results)

    # always put humans first (audio, text) then CLM models, then MLM models
    human_conditions = ['audio', 'text']

    models_order = [item for item in ordered_accuracy if item not in ['audio', 'text', *MLM_MODELS]]
    models_order = models_order + MLM_MODELS

    human_models_order = human_conditions + models_order

    #################################################
    ### Plot 1a: binary accuracy for each task ###
    #################################################

    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    axes = axes.flatten()

    cmap = utils.create_spoken_written_cmap(continuous=False)
    variable = f'accuracy'

    for ax, (task, df) in zip(axes, df_results.groupby('task')):
        plt.sca(ax)
        ax = utils.plot_bar_results(df, x='modality', y=variable, hue=None, order=human_models_order, cmap=cmap, figsize=None, add_points=False)
        ax.set_ylim([0, 1])
        ax.set_title(f'Task - {task}')
        plt.xticks(rotation=45, ha='right')

        ax.set_xlabel('Modality/Model')
        ax.set_ylabel('Accuracy (Percent Correct)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "individual-task_human-llm-comparison_binary-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #######################################################
    ### Plot 1b: binary accuracy collapsed across tasks ###
    #######################################################

    # plot top word accuracy for humans
    cmap = utils.create_spoken_written_cmap(continuous=False)
    ax = utils.plot_bar_results(df_results, x='modality', y='accuracy', hue=None, order=human_models_order, cmap=cmap, figsize=FIGSIZE, add_points=False)
    
    plt.xlabel('Modality/Model')
    plt.xticks(rotation=45, ha='right')
    
    plt.ylim([0, 1])
    plt.ylabel('Accuracy (Percent Correct)')
    plt.title(f'All task - binary accuracy')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_binary-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    ### Plot 2a: continuous accuracy for each task ###
    #################################################

    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    axes = axes.flatten()

    variable = f'{p.word_model}_top_word_accuracy'

    for ax, (task, df) in zip(axes, df_results.groupby('task')):
        plt.sca(ax)
        ax = utils.plot_bar_results(df, x='modality', y=variable, hue=None, order=human_models_order, cmap=cmap, figsize=None, add_points=False)
        ax.set_ylim([0, 1])
        ax.set_title(f'Task - {task}')
        plt.xticks(rotation=45, ha='right')

        ax.set_xlabel('Modality/Model')
        ax.set_ylabel('Cosine similarity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_continuous-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    ###########################################################
    ### Plot 2b: continuous accuracy collapsed across tasks ###
    ###########################################################

    # plot top word accuracy for humans
    variable = f'{p.word_model}_top_word_accuracy'

    cmap = utils.create_spoken_written_cmap(continuous=False)
    ax = utils.plot_bar_results(df_results, x='modality', y=variable, hue=None, order=human_models_order, cmap=cmap, figsize=FIGSIZE, add_points=False)

    plt.xlabel('Task')
    plt.xticks(rotation=45, ha='right')

    plt.ylim([0, 1.0])
    plt.ylabel('Cosine similarity')
    
    plt.title(f'All task - continuous accuracy')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "individual-task_human-llm-comparison_continuous-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    ####### Plot 3: quadrant plot of accuracy #######
    #################################################

    # currently plotting over tasks
    df_all_tasks = []

    for task in p.task_list:
        model_quadrants = utils.load_model_quadrant_info(preproc_dir, models_dir, task=task, model_name='gpt2-xl')
        df_task = df_results[df_results['task'] == task]

        df_all_tasks.append((df_task, model_quadrants))

    df_tasks, df_quadrants = [pd.concat(df).reset_index(drop=True) for df in zip(*df_all_tasks)]
    utils.create_joint_density_plot(df_tasks, df_quadrants, word_model_name=p.word_model, weight_type='human>model', bw_adjust=0.65)

    plt.xlabel('Cosine Similarity')
    plt.xticks(rotation=45, ha='right')

    plt.ylabel('Entropy')
    plt.title(f'All task - human-LLM difference')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_quadrant-plot.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    #### Plot 4: KL Divergence of model & humans ####
    #################################################
    
    df_kl_div = []

    for task in p.task_list:
        df = pd.read_csv(os.path.join(results_dir,  f'task-{task}_group-analyzed-behavior_window-size-{p.window_size}_human-model-distributions-lemmatized.csv'))
        df['task'] = task
        df_kl_div.append(df)

    df_kl_div = pd.concat(df_kl_div).reset_index(drop=True)
    
    df_kl_div.to_csv(os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{p.window_size}_human-model-distributions-lemmatized.csv'), index=False)

    # cmap = create_spoken_written_cmap(continuous=False)
    sns.set(style='white', rc={'figure.figsize':(8,5)})

    cmap = utils.create_spoken_written_cmap(continuous=False)
    ax = utils.plot_bar_results(df, x='model_name', y="kl_divergence", hue="modality", order=models_order, cmap=cmap, figsize=(7,5), add_points=False)

    plt.xticks(rotation=45, ha='right')

    plt.ylim([0, 6.5])
    plt.ylabel('KL Divergence')
    
    plt.title(f'All task - model fit to human distribution')
    plt.gca().get_legend().remove()
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_kl-divergence.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    # sns.barplot(data=df_kl_div, x='model_name', y=variable, hue='modality', palette=cmap, order=ordered_modalities)
    # plt.ylim([0, 4.5])

    #################################################
    ###### Plot 5: Difference of KL Divergence ######
    #################################################

    # Define gradient colors
    cmap = utils.create_spoken_written_cmap()

    df_kl_difference = []

    for (task, model_name), df in df_kl_div.groupby(['task', 'model_name']):
        df_audio = df[df['modality'] == 'audio'].reset_index(drop=True)
        df_text = df[df['modality'] == 'text'].reset_index(drop=True)

        kl_diff = df_text['kl_divergence'] - df_audio['kl_divergence'] 

        df_diff = pd.DataFrame(kl_diff, columns=['kl_divergence'])
        df_diff['task'] = task
        df_diff['model_name'] = model_name

        df_kl_difference.append(df_diff)

    df_kl_difference = pd.concat(df_kl_difference).reset_index(drop=True)

    ax = sns.pointplot(df_kl_difference, x="kl_divergence", y="model_name",
        color="black", markersize=4, linestyles="none", order=models_order,
        errorbar='se',
    )

    plt.xlim([-1, 1])

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Create gradient rectangle
    gradient_rect = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        transform=ax.transData,
        color="white", alpha=0.2
    )

    ax.add_patch(gradient_rect)

    # Use the gradient as an image in the background
    ax.imshow(
        np.linspace(0, 1, 256).reshape(1, -1),  # Create 1D gradient
        aspect="auto", extent=(xmin, xmax, ymin, ymax),
        origin="lower", cmap=cmap, alpha=0.75
    )

    ax.vlines(0, ymin=ymin, ymax=ymax, linestyle='--', color='0.5', linewidth=2)

    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_kl-divergence-difference.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')