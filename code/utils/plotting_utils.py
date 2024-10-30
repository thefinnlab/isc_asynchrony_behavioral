import os, sys
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

from preproc_utils import load_model_results, divide_nwp_dataframe

def linear_norm(x, min, max):
    return (x - min) / (max - min)

def get_ordered_accuracy(df_human_models, word_model_name='fasttext'):
    
    # Get order of models by binary accuracy
    ordered_accuracy = df_human_models.loc[:,['modality', f'{word_model_name}_avg_accuracy']] \
        .groupby(['modality']) \
        .mean() \
        .sort_values(by=f'{word_model_name}_avg_accuracy').index[::-1]

    return ordered_accuracy


def plot_comparison_identity(ds_a, ds_b, lim):
    # plt.switch_backend('agg')

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)

    ax.plot(lim, lim, 'k--')
    ax.scatter(ds_a, ds_b, s=10, linewidth=0, alpha=0.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    return fig

def create_spoken_written_cmap(continuous=True):

    if continuous:
        spoken_written_cmap = clr.LinearSegmentedColormap.from_list('spoken-written', ['#005208', '#72D16B', '#808080', '#E4B266', '#623800'], N=256)
        spoken_written_cmap = spoken_written_cmap.reversed()
    else:
        spoken_written_cmap = sns.color_palette('BuPu', n_colors=9)
        spoken_written_cmap.insert(0, '#82C564')
        spoken_written_cmap.insert(1, '#F7CD84')
    return spoken_written_cmap

###############################
####### Bar plot utility ######
###############################

def plot_bar_results(df, x, y, hue, cmap, alpha=0.75, figsize=(6,5), order=None, add_points=True):

    if figsize:
        sns.set(style='white', rc={'figure.figsize': figsize})

    ax = sns.barplot(data=df, x=x, y=y, hue=hue, palette=cmap, alpha=alpha, order=order) 

    if add_points:
        ax = sns.stripplot(data=df, x=x, y=y,  hue=hue,  palette=cmap, size=4,
            edgecolor='black', linewidth=0.25, dodge=True, alpha=0.3, ax=ax)
    
    sns.despine()
    return ax

###############################
#### Quadrant plot utility ####
###############################

def load_model_quadrant_info(
    preproc_dir,
    models_dir,
    task,
    model_name='gpt2-xl',
    window_size=25,
    top_n=5,
    accuracy_type='fasttext_avg_accuracy',
    accuracy_percentile=45,
):
    """
    Load and preprocess model results for analysis.
    
    Parameters:
    - models_dir: str, directory containing model results
    - task: str, name of the task being analyzed
    - model_name: str, name of the model to load (default: 'gpt2-xl')
    - window_size: int, context window size (default: 25)
    - top_n: int, number of top predictions to consider (default: 5)
    - preproc_dir: str, directory containing preprocessed data (default: 'stimuli/preprocessed')
    
    Returns:
    - DataFrame containing processed model results for analysis
    """
    # Load preprocessed data to get prediction word indices
    df_preproc = pd.read_csv(os.path.join(preproc_dir, task, f'{task}_transcript-preprocessed.csv'))
    nwp_idxs = np.where(df_preproc['NWP_Candidate'])[0]
    
    # Load selected model data -- which words were selected for the experiment
    df_selected = pd.read_csv(os.path.join(preproc_dir, task, f'{task}_transcript-selected.csv'))
    selected_idxs = np.where(df_selected['NWP_Candidate'])[0]

    # Load raw model results
    results = load_model_results(models_dir, model_name=model_name, task=task, window_size=window_size, top_n=top_n)
    results.loc[:, 'binary_accuracy'] = results['binary_accuracy'].astype(bool)

    # Now divide into quadrants
    # get xmedian and ymedian --> needs to happen before otherwise plot is off
    x_median = np.nanmedian(results[accuracy_type])
    y_median = np.nanmedian(results['entropy'])
    
    xmin, xmax = results[accuracy_type].max(), results[accuracy_type].min()
    ymin, ymax = results['entropy'].max(), results['entropy'].min()
    
    # divide the data into quadrants based on percentile
    # we use a form of continuous accuracy and entropy
    df_divide = divide_nwp_dataframe(results, accuracy_type=accuracy_type, percentile=accuracy_percentile, drop=False)

    if selected_idxs is not None:
        df_divide = df_divide.loc[selected_idxs]

    # Filter to words of interest
    return df_divide

def calculate_weights(df_human_models, accuracy_type, modalities=['audio', 'text']):
    """
    Calculate raw weights for different modality comparisons.
    
    Parameters:
    - df_human_models: DataFrame with human and model performance data
    - accuracy_type: str, type of accuracy to use
    - modalities: list of str, modalities to compare
    
    Returns:
    - DataFrame containing raw weights for different comparisons
    """
    # Extract audio and text data
    audio, text = [
        df_human_models[df_human_models['modality'] == modality][accuracy_type].to_numpy()
        for modality in modalities
    ]
    
    # Get word indices for the DataFrame index
    word_indices = df_human_models[df_human_models['modality'] == modalities[0]].index
    
    # Calculate model average
    models_only = df_human_models[~df_human_models['modality'].isin(modalities)]

    if 'task' in models_only.columns:
        models_array = pd.pivot(models_only, index=['task', 'word_index'], columns='modality', values=accuracy_type).to_numpy()
    else:
        models_array = pd.pivot(models_only, index=['word_index'], columns='modality', values=accuracy_type).to_numpy()
    
    avg_model = np.nanmean(models_array, axis=1)
    
    # Calculate raw contrasts
    audio_v_model = audio - avg_model
    text_v_model = text - avg_model
    
    # Create DataFrame with all weight types
    df_weights = pd.DataFrame({
        'audio>model': audio_v_model,
        'text>model': text_v_model,
        'human>model': (audio_v_model + text_v_model)/2,
        'audio>text': audio - text
    }, index=word_indices)
    
    return df_weights


def create_joint_density_plot(
    df_human_models,
    model_results,
    word_model_name,
    accuracy_percentile=45,
    weight_type='human>model',
    cmap_name='BuPu',
    bw_adjust=0.65,
):
    """
    Creates a joint density plot comparing model accuracy and entropy, weighted by human performance.
    
    Parameters:
    - df_human_models: DataFrame containing human and model performance data
    - model_results: DataFrame containing model predictions and metrics
    - word_model_name: str, name of the word model being analyzed
    - accuracy_percentile: int, percentile for accuracy threshold (default: 45)
    - weight_type: str, type of weighting to use (default: 'human>model')
    - cmap_name: str, name of colormap to use (default: 'BuPu')
    
    Returns:
    - g: seaborn.JointGrid object containing the plot
    """
    
    # Calculate human vs model contrasts
    accuracy_type = f'{word_model_name}_avg_accuracy'
    
   # Calculate weights DataFrame
    df_weights = calculate_weights(df_human_models, accuracy_type)
    
    # Get statistics for the selected weight type
    weights = df_weights[weight_type].to_numpy()
    weights = linear_norm(weights, -1, 1)

    # Set up plot
    sns.set_theme(style="white")
    cmap = cm.get_cmap(cmap_name)

    # Create joint plot
    g = sns.JointGrid(data=model_results, x=accuracy_type, y="entropy", space=0)

    # # Create custom norm for colormap centered at mean
    # norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    # # Get the density range from the KDE computation first
    # from scipy.stats import gaussian_kde

    # # Compute 2D KDE manually to get density range
    # x = model_results[accuracy_type].to_numpy()
    # y = model_results['entropy'].to_numpy()
    # xy_coords = np.vstack([x, y])  # where x and y are your data points
    # kde = gaussian_kde(xy_coords, weights=weights)
    # density = kde(xy_coords)

    # # Create a custom normalization centered on the midpoint
    # density_min, density_midpoint, density_max = density.min(), density.mean(), density.max()
    # norm = TwoSlopeNorm(vmin=density_min, vcenter=density_midpoint, vmax=density_max)

    # # Create evenly spaced ticks
    # ticks = np.linspace(density_min, density_max, 6)  # 6 ticks for good distribution

    # print (weights.min(), weights.max(), weights.mean())

    # Add scatter and density plots
    g.plot_joint(sns.scatterplot, color="k", alpha=0.75, s=30)

    g.plot_joint(
        sns.kdeplot,
        weights=weights,
        bw_adjust=bw_adjust,
        fill=True,
        thresh=bw_adjust*weights.mean(),
        # hue_norm=norm,
        levels=100,
        cmap=cmap,
        alpha=0.5,
        cbar=True,
        # cbar_kws={
        #     # 'norm': norm,
        #     'ticks': ticks,
        #     'format': '%.4f'  # Format to 4 decimal places
        # }
    )

    g.plot_marginals(sns.kdeplot, color=cmap(0.5), fill=True, bw_adjust=bw_adjust)
    
    # Add median lines
    x_median = np.nanmedian(model_results[accuracy_type])
    y_median = np.nanmedian(model_results['entropy'])
    xmin, xmax = model_results[accuracy_type].min(), model_results[accuracy_type].max()
    ymin, ymax = model_results['entropy'].min(), model_results['entropy'].max()
    
    xmax *= 1.05
    xmin -= 0.25 * xmax

    ymax *= 1.05
    ymin -= 0.25 * ymin
    
    g.fig.axes[0].set_ylim(0, ymax)
    g.fig.axes[0].set_xlim(0, xmax)
    g.fig.axes[0].vlines(x=x_median, ymin=ymin, ymax=ymax, linestyles='dashed', color='.33')
    g.fig.axes[0].hlines(y=y_median, xmin=xmin, xmax=xmax, linestyles='dashed', color='.33')
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    pos_joint_ax = g.ax_joint.get_position()
    pos_marg_x_ax = g.ax_marg_x.get_position()
    g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    
    # # Set labels
    # plt.xlabel('Continuous Accuracy')
    # plt.ylabel('GPT2-XL Entropy')
    
    return g