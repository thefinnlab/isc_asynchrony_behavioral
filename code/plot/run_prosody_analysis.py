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

def batch_average(df, batch_size, columns):
    """
    Average a DataFrame into batches of specified size.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame to be batched
    batch_size (int): Size of each batch
    
    Returns:
    pandas.DataFrame: DataFrame with averaged batches
    """
    # Calculate number of complete batches
    n_batches = len(df) // batch_size
    
    # Handle case where DataFrame length isn't divisible by batch_size
    remainder = len(df) % batch_size
    
    # If there's no perfect division, we'll need one more batch
    if remainder > 0:
        n_batches += 1
    
    # Create list to store batch averages
    batch_averages = []
    
    # Process complete batches
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))  # Use min to handle last batch
        
        # Calculate average for current batch
        batch_avg = df.iloc[start_idx:end_idx]
        batch_avg = batch_avg[columns].mean()
        batch_avg['batch_number'] = i  # Add batch number for reference
        batch_averages.append(batch_avg)
    
    # Combine all batch averages into a new DataFrame
    return pd.DataFrame(batch_averages)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
    prosody_dir = os.path.join(BASE_DIR, 'derivatives/joint-prosody-clm/')
    plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final/prosody/')

    attempt_makedirs(plots_dir)

    model_name_mapping = {
        f'{p.dataset}-prosody_scratch-gpt2_joint-loss_prosody-embed': 'ProsodyPrediction',
        f'{p.dataset}-prosody_scratch-gpt2_clm-loss_prosody-embed': 'ProsodyAccess',
        f'{p.dataset}-prosody_scratch-gpt2_clm-loss_no-prosody-embed': 'ProsodyDeprived'
    }

    # Load results for all models except the yoked models
    results_fns = glob.glob(os.path.join(BASE_DIR, f'derivatives/joint-prosody-clm/*{p.dataset}*'))
    df_results = pd.concat([pd.read_csv(fn) for fn in results_fns if 'yoked' not in fn]).reset_index(drop=True)

    df_stack = []

    # Go through each set of results and average based on batch size
    for i, df in df_results.groupby('model_name'):
        df = batch_average(df, batch_size=p.batch_size, columns=['loss', 'clm_loss', 'accuracy'])
        df['model_name'] = i
        df['perplexity'] = np.exp(df['clm_loss'])
        df['batch'] = i
        df_stack.append(df)

    df_results = pd.concat(df_stack).reset_index(drop=True)

    # Get order of models by binary accuracy
    ordered_accuracy = df_results.loc[:,['model_name', 'accuracy', 'perplexity']] \
        .groupby(['model_name']) \
        .mean()

    # get max chance of null models
    null_models = ordered_accuracy.index.str.contains('shuffle')
    accuracy_chance = ordered_accuracy.loc[null_models, 'accuracy'].max()
    perplexity_chance = ordered_accuracy.loc[null_models, 'perplexity'].min()

    # order models by accuracy
    ordered_accuracy = ordered_accuracy[~null_models]
    ordered_models = ordered_accuracy.sort_values(by=f'accuracy').index[::-1]

    # remove the null models
    df_results = df_results[~df_results['model_name'].str.contains('shuffle')]
    df_results['model_name'] = df_results['model_name'].apply(lambda x: model_name_mapping[x])
    ordered_models = [model_name_mapping[model] for model in ordered_models]

    # Save to a csv file
    out_fn = os.path.join(results_dir, f'{p.dataset}-prosodyllm_all-results_batch-size-{p.batch_size}.csv')
    df_results.to_csv(out_fn, index=False)

    #################################################
    ########## Plot 1: Plot model accuracy  #########
    #################################################

    plt.figure(figsize=(4,5))
    sns.set(style='white')
    
    ax = sns.barplot(data=df_results, x="model_name", y="accuracy", hue="model_name",  # Add hue parameter
        palette="rocket", alpha=0.8, order=ordered_models, legend=False)

    plt.xlabel('Model')
    plt.ylabel('Accuracy (Percent Correct)')

    plt.title(f'ProsodyLLM – {p.dataset.capitalize()}')
    plt.xticks(rotation=45, ha='right')

    if p.dataset == 'helsinki':
        plt.ylim([0.2, 0.26])
    elif p.dataset == 'gigaspeech':
        plt.ylim([0.15, 0.21])

    plt.axhline(y=accuracy_chance, color='k', linestyle='--')
    sns.despine()

    # Add text labels on top of each bar
    for patch, acc in zip(ax.patches, ordered_accuracy['accuracy']):  # Changed variable name from p to patch
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2.,
            height + 0.008,
            f'{acc:.3f}',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{p.dataset}-prosody-llm_accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    ########## Plot 2: Plot model perplexity  #########
    #################################################

    plt.figure(figsize=(4,5))
    sns.set(style='white')
    
    ax = sns.barplot(data=df_results, x="model_name", y="perplexity", hue="model_name",
        palette="rocket", alpha=0.8, order=ordered_models, legend=False)

    plt.xlabel('Model')
    plt.ylabel('Perplexity')

    plt.title(f'ProsodyLLM – {p.dataset.capitalize()}')
    plt.xticks(rotation=45, ha='right')

    if p.dataset == 'helsinki':
        plt.ylim([0, 350])
    elif p.dataset == 'gigaspeech':
        plt.ylim([0, 350])

    plt.axhline(y=perplexity_chance, color='k', linestyle='--')
    sns.despine()

    # Add text labels on top of each bar
    for patch, acc in zip(ax.patches, ordered_accuracy['perplexity']):  # Changed variable name from p to patch
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2.,
            height + 15,
            f'{acc:.2f}',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{p.dataset}-prosody-llm_perplexity.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')
