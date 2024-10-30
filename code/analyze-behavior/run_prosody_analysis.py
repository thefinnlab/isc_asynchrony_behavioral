import os, sys
import glob
import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt
import seaborn as sns

sys.path.append('../utils/')

from config import *
from dataset_utils import attempt_makedirs
import analysis_utils as utils

def process_prosody_data(results_dir, task_list, word_model, past_n_words=5, remove_words=None):
    """
    Process prosody data for multiple tasks by combining transcripts, prosody metrics, and human results.
    
    Args:
        task_list (list): List of tasks to process
        results_dir (str): Directory containing human results
        past_n_words (int): Number of previous words to consider for prosody calculations
        remove_words (list): List of words to remove from analysis
    
    Returns:
        tuple: (processed_data, processed_results) DataFrames containing combined analysis
    """
    # Define column names for prosody data
    prosody_columns = ['stim', 'start', 'end', 'word', 'prominence', 'boundary']
    
    # Initialize lists to store processed DataFrames
    processed_data = []
    processed_results = []
    
    for task in task_list:
        # Load data files
        transcript_df = pd.read_csv(os.path.join(BASE_DIR, 'stimuli/preprocessed/', task, f'{task}_transcript-selected.csv'))
        prosody_df = pd.read_csv(os.path.join(BASE_DIR, 'stimuli/prosody/', f'{task}.prom'), sep='\t', names=prosody_columns)

        # Read results from behavioral tasks and sort by modality and word index --> allows us to add the prosody data easily
        human_results_df = pd.read_csv(os.path.join(results_dir, f'task-{task}_group-analyzed-behavior_human-lemmatized.csv'))
        human_results_df = human_results_df.sort_values(by=['modality', 'word_index'])
        human_results_df['task'] = task
        
        # Process prosody metrics and filter data
        prosody_df = utils.calculate_prosody_metrics(prosody_df, n_prev=past_n_words, remove_characters=remove_words)
        filtered_prosody = prosody_df[transcript_df['NWP_Candidate']]
        filtered_transcript = transcript_df[transcript_df['NWP_Candidate']]
        
        # Add task-specific columns
        filtered_transcript['entropy_accuracy_group'] = (filtered_transcript['entropy_group'] + 
                                                       '_' + filtered_transcript['accuracy_group'])
        filtered_transcript['stim'] = task
        
        # Map prosody features to human results (duplicate for paired data)
        prosody_features = {
            'boundary': filtered_prosody['boundary'],
            'prosody_raw': filtered_prosody['prominence'],
            'prosody_mean': filtered_prosody['prosody_mean'],
            'prosody_std': filtered_prosody['prosody_std'],
            'prosody_slope': filtered_prosody['prosody_slope'],
            'relative_prosody': filtered_prosody['relative_prosody'],
            'relative_norm': filtered_prosody['relative_norm'],
            'boundary_mean': filtered_prosody['boundary_mean'],
            'boundary_std': filtered_prosody['boundary_std']
        }
        
        # Apply features to human results DataFrame
        for feature, values in prosody_features.items():
            human_results_df[feature] = np.tile(values, 2)
        
        # Append processed DataFrames
        processed_results.append(human_results_df)
    
    # Combine all processed data
    final_results = pd.concat(processed_results).reset_index(drop=True)
    
    # Calculate weighted accuracy
    final_results['weighted_accuracy'] = (final_results[f'{word_model}_top_word_accuracy'] * final_results['top_prob'])
    
    return final_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-task_list', '--task_list', type=str, nargs='+')
    parser.add_argument('-word_model', '--word_model', type=str, default='fasttext')
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
    prosody_dir = os.path.join(BASE_DIR, 'derivatives/joint-prosody-clm/')
    plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final/prosody/')

    attempt_makedirs(plots_dir)

    ###################################
    ### Load results from task list ###
    ###################################

    df_prosody = process_prosody_data(results_dir, p.task_list, p.word_model, remove_words=utils.REMOVE_WORDS)

    # Melt the dataframe for easy plotting of prosody by metric
    prosody_vars = ['prosody_raw', 'prosody_mean', 'prosody_std', 'boundary',  'boundary_mean', 'boundary_std', 'prosody_slope',  'relative_prosody']
    # df_prosody = pd.melt(df_prosody, id_vars=['modality', 'weighted_accuracy', 'fasttext_top_word_accuracy'], 
    #                 value_vars=prosody_vars, var_name='prosody_metric', value_name='value')

    #################################################
    ### Plot 1: Average prosody over last n_words ###
    #################################################

    cmap = utils.create_spoken_written_cmap(continuous=False)
    ax = sns.lmplot(df_prosody, x='prosody_mean', y=f"{p.word_model}_top_word_accuracy", hue='modality', palette=cmap)

    plt.xlabel('Average Prosodic Prominence')
    plt.ylabel('Cosine Similarity')

    plt.title(f'All task - average prosody accuracy relationship')
    # plt.gca().get_legend().remove()
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_avg-prosody-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    ##### Plot 2: STD prosody over last n_words #####
    #################################################

    cmap = utils.create_spoken_written_cmap(continuous=False)
    ax = sns.lmplot(df_prosody, x='prosody_std', y=f"{p.word_model}_top_word_accuracy", hue='modality', palette=cmap)

    plt.xlabel('Variability of Prosodic Prominence')
    plt.ylabel('Cosine Similarity')

    plt.title(f'All task - std prosody accuracy relationship')
    # plt.gca().get_legend().remove()
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_std-prosody-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    ##### Plot 3: LLM w/ prosody access v. none  ####
    #################################################

    results_fns = glob.glob(os.path.join(prosody_dir, f'*test-prominence.csv'))
    df_results = pd.concat([pd.read_csv(fn) for fn in results_fns]).reset_index(drop=True)
    df_results['accuracy'] = df_results['accuracy'] * 100

    ax = utils.plot_bar_results(df_results, x='model_name', y="accuracy", hue="model_name", cmap='rocket', figsize=(4,5), add_points=False)

    plt.xlabel('Model')
    plt.ylabel('Accuracy (Percent Correct)')

    plt.title(f'ProsodyCLM – Helsinki')
    # plt.gca().get_legend().remove()
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "joint-prosody-clm_model-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

