import os, sys
import argparse
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append('../utils/')

from config import *
from dataset_utils import attempt_makedirs
import plotting_utils as utils

FIG_SIZE = (3,5)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-task_list', '--task_list', type=str, nargs='+')
    parser.add_argument('-word_model', '--word_model', type=str, default='fasttext')
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
    plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final/behavioral/')

    attempt_makedirs(plots_dir)

    ###################################
    ###### Subject-wise analysis ######
    ###################################

    # load results of overall subjects
    df_subject_results = []

    for task in p.task_list:
        df_task = pd.read_csv(os.path.join(results_dir, f'task-{task}_group-cleaned-behavior_lemmatized.csv'))
        df_task['task'] = task

        df_subject_results.append(df_task)

    # concatenate into one dataframe
    df_subject_results = pd.concat(df_subject_results).reset_index(drop=True)
    df_subject_results.to_csv(os.path.join(results_dir, 'all-task_subject-behavior_lemmatized.csv'), index=False)

    #########################################
    ##### Plot 1: Subject-wise accuracy #####
    #########################################

    # average within accuracy by subject within task and modality
    df_subject_accuracy = df_subject_results.groupby(['task', 'modality', 'subject'])['accuracy'].mean().reset_index()

    cmap = utils.create_colormap(continuous=False)
    ax = utils.plot_bar_results(df_subject_accuracy, x='task', y='accuracy', hue='modality', order=p.task_list, cmap=cmap, figsize=FIG_SIZE)

    plt.xlabel('Task')
    plt.ylabel('Accuracy (Percent Correct)')
    plt.title(f'All task - Subject-wise accuracy')
    plt.ylim(0, 0.75)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.gca().get_legend().remove()

    plt.savefig(os.path.join(plots_dir, "all-task_subject-prediction_accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    ###################################
    ### Load results from task list ###
    ###################################

    df_results = []

    for task in p.task_list:
        df_task = pd.read_csv(os.path.join(results_dir, f'task-{task}_group-analyzed-behavior_human-lemmatized.csv'))
        df_task['task'] = task
        df_results.append(df_task)

    # concatenate into one dataframe --> write to file for posterity 
    df_results = pd.concat(df_results).reset_index(drop=True)

    # Calculate here for now, but move to analysis scripts
    df_results['weighted_accuracy'] = (df_results[f'{p.word_model}_top_word_accuracy'] * df_results['top_prob'])

    df_results.to_csv(os.path.join(results_dir, 'all-task_group-analyzed-behavior_human-lemmatized.csv'), index=False)

    ###################################
    ##### Plot 2: Binary accuracy #####
    ###################################

    # no points for binary since it's 0 or 1
    cmap = utils.create_colormap(continuous=False)
    ax = utils.plot_bar_results(df_results, x='task', y='accuracy', hue='modality', order=p.task_list, cmap=cmap, figsize=FIG_SIZE, add_points=False)
    
    plt.xlabel('Task')
    plt.ylabel('Accuracy (Percent Correct)')
    plt.title(f'All task - Binary accuracy')
    plt.ylim(0, 0.85)
    plt.gca().get_legend().remove()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-behavior_binary-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #######################################
    ##### Plot 3: Continuous accuracy #####
    #######################################

    # plot top word accuracy for humans
    cmap = utils.create_colormap(continuous=False)
    ax = utils.plot_bar_results(df_results, x='task', y=f'{p.word_model}_top_word_accuracy', hue='modality', order=p.task_list, cmap=cmap, figsize=FIG_SIZE)

    plt.xlabel('Task')
    plt.ylabel('Cosine similarity')
    plt.title(f'All task - Continuous accuracy')
    plt.gca().get_legend().remove()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-behavior_continuous-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #######################################
    ###### Plot 4: Weighted accuracy ######
    #######################################

    # plot top word accuracy for humans
    cmap = utils.create_colormap(continuous=False)
    ax = utils.plot_bar_results(df_results, x='task', y=f'weighted_accuracy', hue='modality', order=p.task_list, cmap=cmap, figsize=FIG_SIZE)

    plt.xlabel('Task')
    plt.ylabel('Cosine similarity')
    plt.title(f'All task - Weighted accuracy')
    plt.gca().get_legend().remove()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-behavior_weighted-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    ###################################
    ######### Plot 5: Entropy #########
    ###################################

    # plot entropy of human distributions
    cmap = utils.create_colormap(continuous=False)
    ax = utils.plot_bar_results(df_results, x='task', y='entropy', hue='modality', order=p.task_list, cmap=cmap, figsize=FIG_SIZE)

    plt.xlabel('Task')
    plt.ylabel('Entropy')
    plt.title(f'All task - Distribution entropy')
    plt.gca().get_legend().remove()
    plt.ylim([0, 4.5])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-behavior_distribution-entropy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    ##########################################
    ##### Plot 6: Normalized entropy #########
    ##########################################

    # plot entropy of human distributions
    cmap = utils.create_colormap(continuous=False)
    ax = utils.plot_bar_results(df_results, x='task', y='normalized_entropy', hue='modality', order=p.task_list, cmap=cmap, figsize=FIG_SIZE)

    plt.xlabel('Task')
    plt.ylabel('Entropy')
    plt.title(f'All task - Normalized entropy')
    plt.gca().get_legend().remove()
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-behavior_normalized-entropy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    ##########################################
    ########## Plot 7: Predictability ########
    ##########################################

    # predictability is the percentage of participants predicting the correct word
    # create colors of the plot based on how different the scores were between modalities 
    df_predictability = pd.pivot(df_results, index=['task', 'word_index'], columns='modality', values='predictability')
    colors = df_predictability['audio'] - df_predictability['text']
    colors = utils.linear_norm(colors.to_numpy(), -1, 1)

    df_predictability['colors'] = colors

    # get the continuous colormap
    cmap = utils.create_colormap()
    sns.set(style='white', rc={'figure.figsize':(5,5)})

    # set the color of the individual lines for each task
    line_cmap = sns.color_palette("gist_earth", 3)
    fig, ax = plt.subplots(1,1)

    for i, (task, df) in enumerate(df_predictability.groupby('task')):
        # lines of color to average skew = cmap(df['colors'].mean())
        ax = sns.scatterplot(data=df, x='audio', y='text', color=cmap(df['colors']), alpha=0.75, s=25, ax=ax, legend=False)
        ax = sns.regplot(data=df, x='audio', y='text', color=line_cmap[i], scatter=False, label=task, ax=ax)

    plt.plot((0,1), (0, 1), 'k--')
    plt.legend(title='Tasks')

    plt.ylabel('Written predictability')
    plt.xlabel('Spoken predictability') 
    plt.title(f'All task - human predictability relationship')
    plt.tight_layout()
    sns.despine()

    plt.savefig(os.path.join(plots_dir, "all-task_human-behavior_predictability.pdf"), bbox_inches='tight', dpi=600)