import os, sys
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.ndimage import gaussian_filter

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors as clr

from preproc_utils import load_model_results, divide_nwp_dataframe

def linear_norm(x, min, max):
    return (x - min) / (max - min)

def get_ordered_accuracy(df_human_models, word_model_name='fasttext'):
    
    # Get order of models by binary accuracy
    ordered_accuracy = df_human_models.loc[:,['modality', f'{word_model_name}_top_word_accuracy']] \
        .groupby(['modality']) \
        .mean() \
        .sort_values(by=f'{word_model_name}_top_word_accuracy').index[::-1]

    return ordered_accuracy


def plot_comparison_identity(ds_a, ds_b, lim):
    # plt.switch_backend('agg')

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)

    ax.plot(lim, lim, 'k--')
    ax.scatter(ds_a, ds_b, s=10, linewidth=0, alpha=0.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    return fig

def create_colormap(dtype='spoken-written', continuous=True, N=256):

    if dtype == 'human-model':
        cmap = clr.LinearSegmentedColormap.from_list('human-model', ['#B2B800', '#E3E3E3', '#C2ADDA'], N=N).reversed()
    else:
        if continuous:
            # spoken_written_cmap = clr.LinearSegmentedColormap.from_list('spoken-written', ['#005208', '#72D16B', '#808080', '#E4B266', '#623800'], N=256)
            cmap = clr.LinearSegmentedColormap.from_list('spoken-written', ['#2C8E00', '#E2E2E2', '#DE8C00'], N=N)
            cmap = cmap.reversed()
        else:
            cmap = sns.color_palette('BuPu', n_colors=9)
            cmap.insert(0, '#82C564')
            cmap.insert(1, '#F7CD84')

    return cmap

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

# For aggregating over tasks + models
def load_task_model_quadrants(preproc_dir, models_dir, task_list, model_names, word_model_name):

    quadrant_columns = [f'{word_model_name}_avg_accuracy', 'entropy', 'word_index', 'entropy_group', 'accuracy_group']

    all_quadrants = []

    for task in task_list:
        # Load all models and quadrants
        quadrants = []
        
        for model_name in model_names:
            # load quadrants and extract the columns we want
            model_quadrants = load_model_quadrant_info(preproc_dir, models_dir, task=task, model_name=model_name)
            model_quadrants = model_quadrants.reset_index().rename(columns={'index': 'word_index'})
            model_quadrants = model_quadrants.loc[:, quadrant_columns]
            
            quadrants.append(model_quadrants)

        # get mean across models
        df_quadrants = pd.concat(quadrants)
        df_quadrants = df_quadrants.groupby('word_index').mean().reset_index()
        df_quadrants['task'] = task

        all_quadrants.append(df_quadrants)

    df_quadrants = pd.concat(all_quadrants).reset_index(drop=True).sort_values(by=['task', 'word_index'])
    return df_quadrants

# Calculate weights comparing 
def calculate_weights(df_distributions):
    df_distributions = df_distributions.sort_values(by=['modality', 'task', 'word_index'])

    # Put models as columns and modality, task, word_index as rows
    df_distributions = pd.pivot(df_distributions, index=['modality', 'task', 'word_index'], columns='model_name', values='human_model_accuracy_diff')
    
    # Separate into audio and text (these are contrasts of audio & text)
    audio, text = [df.to_numpy() for i, df in df_distributions.groupby('modality')]

    # Create DataFrame with all weight types
    df_weights = pd.DataFrame({
        'audio>model': np.nanmean(audio, axis=1),
        'text>model': np.nanmean(text, axis=1),
        'human>model': np.nanmean((audio + text) / 2, axis=1),
        'audio>text': np.nanmean((audio - text), axis=1),
    })

    return df_weights

###############################
#### Custom 3D heatmap plot ####
###############################

# calculates standard deviation of Gaussian spread
def get_sigma(n_points, width_factor=2.5):
    """
    width_factor matches how MATLAB calculates the sigma of the Gaussian
    """

    # calculate sigma based on a width factor (this matches MATLAB's implementation)
    sigma = (n_points - 1) / (2 * width_factor)

    return sigma

def create_gaussian_circle(buffer, sigma):
    """
    Create a 2D Gaussian circular kernel
    
    Parameters:
    - buffer: radius of the circular area
    - sigma: standard deviation of the Gaussian distribution
    
    Returns:
    - 2D numpy array with Gaussian weights
    """
    # Create a grid of coordinates
    x = np.linspace(-buffer, buffer, 2 * buffer + 1)
    y = np.linspace(-buffer, buffer, 2 * buffer + 1)
    xx, yy = np.meshgrid(x, y)
    
    # Create circular mask
    circular_mask = xx**2 + yy**2 <= buffer**2
    
    # Create Gaussian kernel
    gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    return gaussian_kernel

def create_density_map(image, locations, weights, fill_value=0, buffer=10, base_width=200):

    # Create a base image to use 
    density_map = np.zeros(image.shape[:-1], dtype=float)
    density_map[:] = fill_value

    # Calculate sigma for Gaussian filtering
    sigma = get_sigma(base_width)
    
    # Create Gaussian circular kernel
    gaussian_kernel = create_gaussian_circle(buffer, sigma)
    
    # Round up to the nearest point so that we can use them as indices
    loc_idxs = np.round(locations).astype(int)

    # Temporary map to track distributed weights
    distributed_weights_map = np.zeros_like(density_map, dtype=float)
    distributed_weights_map[:] = fill_value

    avg_map = np.ones_like(distributed_weights_map, dtype=float)

    # Add weighted Gaussian circles for each point
    for (loc_x, loc_y), duration in zip(loc_idxs.T, weights):
        
        # Calculate kernel placement
        x_start = max(0, loc_x - buffer)
        x_end = min(density_map.shape[0], loc_x + buffer + 1)
        y_start = max(0, loc_y - buffer)
        y_end = min(density_map.shape[1], loc_y + buffer + 1)
        
        # Calculate kernel slice
        kernel_x_start = buffer - (loc_x - x_start)
        kernel_x_end = buffer + (x_end - loc_x)
        kernel_y_start = buffer - (loc_y - y_start)
        kernel_y_end = buffer + (y_end - loc_y)
        
        # Add weighted Gaussian kernel to density map
        kernel_slice = gaussian_kernel[kernel_x_start:kernel_x_end, kernel_y_start:kernel_y_end]

        # Normalize the kernel slice to sum to 1
        # kernel_slice /= kernel_slice.sum()
        avg_map[x_start:x_end, y_start:y_end] += kernel_slice
        
        # Distribute the weight proportionally using the Gaussian kernel
        distributed_kernel = kernel_slice * duration
        
        # Add to distributed weights map
        distributed_weights_map[x_start:x_end, y_start:y_end] += distributed_kernel

    distributed_weights_map /= avg_map
    return distributed_weights_map

def create_weighted_density_map(
    df, 
    x_column, 
    y_column, 
    weights, 
    buffer=50, 
    base_width=50, 
    fill_value=0.5, 
    img_size=(800, 800, 3), 
    gaussian_sigma=25
):
    """
    Create a weighted density map from input dataframe columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    x_column : str
        Name of the column to use for x-axis
    y_column : str
        Name of the column to use for y-axis
    weights : array-like
        Weights to apply to the points
    x_scale : float, optional
        Scaling factor for x-axis values (default: 1000)
    y_scale : float, optional
        Scaling factor for y-axis values (default: 100)
    buffer : int, optional
        Buffer for density map creation (default: 50)
    base_width : int, optional
        Base width for density map creation (default: 50)
    fill_value : float, optional
        Fill value for density map (default: 0.5)
    img_size : tuple, optional
        Size of the image (default: (800, 800, 3))
    gaussian_sigma : float, optional
        Sigma value for gaussian filter (default: 25)
    
    Returns:
    --------
    numpy.ndarray
        Weighted density map
    """

    # Round and convert to integers
    point_x = round(df[x_column]).astype(int)
    point_y = round(df[y_column]).astype(int)
    points = np.stack((point_x, point_y))
    
    # Normalize weights
    weights_normalized = weights #utils.linear_norm(weights, -1, 1)
    
    # Create initial image
    image = np.zeros(img_size)
    
    # Create the density map
    image_map = create_density_map(
        image, 
        points[::-1], 
        weights_normalized, 
        fill_value=fill_value, 
        buffer=buffer, 
        base_width=base_width
    )
    
    # Apply Gaussian filter
    image_map = gaussian_filter(image_map, sigma=gaussian_sigma)
    
    # Normalize the image map to match the original weights range
    min_weight = np.min(weights_normalized)
    max_weight = np.max(weights_normalized)
    
    if np.max(image_map) > 0:
        image_map = (image_map - np.min(image_map)) / \
                    (np.max(image_map) - np.min(image_map)) * \
                    (max_weight - min_weight) + min_weight
    
    # Remove filled values after gaussian smoothing
    filter_value = np.round(image_map[0,0], 2)
    # image_filter = np.round(image_map, 1) == filter_value
    filter_buffer = 0.1
    image_filter = np.logical_and(
        np.round(image_map, 2) >= (filter_value - filter_buffer),
        np.round(image_map, 2) <= (filter_value + filter_buffer),
    )
    image_map[image_filter] = np.nan
    
    return image_map

def create_joint_density_plot(
    df_human_models,
    df_quadrants,
    weight_type='human>model',
    cmap='BuPu',
):
    # scale axes to use an image 
    padding = 200
    x_col, x_scale = 'entropy', 100
    y_col, y_scale = 'fasttext_avg_accuracy', 1000

    df_copy = df_quadrants.copy()

    # Scale the columns
    df_copy[x_col] = df_copy[x_col] * x_scale
    df_copy[y_col] = df_copy[y_col] * y_scale

    # Grab the type of weights we want
    weights = calculate_weights(df_human_models)[weight_type]

    # Create JointGrid
    g = sns.JointGrid(data=df_copy, x=x_col, y=y_col, space=0)
    g.plot_joint(sns.scatterplot, color="k", alpha=0.75, s=30) # Add scatter and density plots   
    g.plot_marginals(sns.kdeplot, color=cmap(0.5), fill=True) # Plot histograms

    # Add median lines
    x_median = np.nanmedian(df_copy[x_col])
    y_median = np.nanmedian(df_copy[y_col])
    xmin, xmax = df_copy[x_col].min(), df_copy[x_col].max()
    ymin, ymax = df_copy[y_col].min(), df_copy[y_col].max()

    xmax *= 1.125
    xmin -= 0.25 * xmax

    ymax *= 1.1
    ymin -= 0.25 * ymin
        
    g.fig.axes[0].set_ylim(0, ymax)
    g.fig.axes[0].set_xlim(0, xmax)
    g.fig.axes[0].vlines(x=x_median, ymin=ymin-padding, ymax=ymax+padding, linestyles='dashed', color='.33')
    g.fig.axes[0].hlines(y=y_median, xmin=xmin-padding, xmax=xmax+padding, linestyles='dashed', color='.33')

    # Plot density map
    density_map = create_weighted_density_map(df_copy, x_column=x_col, y_column=y_col, weights=weights,
                                            fill_value=0, buffer=10, base_width=10, img_size=(round(xmax), round(ymax), 3))

    im = g.ax_joint.imshow(density_map, cmap=cmap, vmin=-1, vmax=1, alpha=0.9, zorder=10)

    # Modify x and y tick labels
    def format_ticks(x, scale):
        return x / scale

    # X-axis ticks
    x_ticks = g.ax_joint.get_xticks()
    x_ticklabels = [f'{format_ticks(x, x_scale):.0f}' for x in x_ticks]
    g.ax_joint.set_xticklabels(x_ticklabels)

    # Y-axis ticks
    y_ticks = g.ax_joint.get_yticks()
    y_ticklabels = [f'{format_ticks(y, y_scale)}' for y in y_ticks]
    g.ax_joint.set_yticklabels(y_ticklabels)

    # Set axis labels with original column names and scaled units
    g.ax_joint.set_xlabel(f'{x_col}')
    g.ax_joint.set_ylabel(f'{y_col}')

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.2)

    # Remove top and right ticks
    g.ax_marg_x.tick_params(bottom=False)
    g.ax_marg_y.tick_params(left=False)

    # Create a new axes for the colorbar
    # Position it to the right of the y-axis histogram
    cbar_ax = g.fig.add_axes([0.85, g.ax_marg_y.get_position().y0, 0.03, g.ax_marg_y.get_position().height])
    plt.colorbar(im, cax=cbar_ax)

    g.ax_joint.set_ylabel('Average 5-shot Accuracy')
    g.ax_joint.set_xlabel('Entropy')

    return g


# def create_joint_density_plot(
#     df_human_models,
#     model_results,
#     word_model_name,
#     accuracy_percentile=45,
#     weight_type='human>model',
#     cmap_name='BuPu',
#     bw_adjust=0.65,
# ):
#     """
#     Creates a joint density plot comparing model accuracy and entropy, weighted by human performance.
    
#     Parameters:
#     - df_human_models: DataFrame containing human and model performance data
#     - model_results: DataFrame containing model predictions and metrics
#     - word_model_name: str, name of the word model being analyzed
#     - accuracy_percentile: int, percentile for accuracy threshold (default: 45)
#     - weight_type: str, type of weighting to use (default: 'human>model')
#     - cmap_name: str, name of colormap to use (default: 'BuPu')
    
#     Returns:
#     - g: seaborn.JointGrid object containing the plot
#     """
    
#     # Calculate human vs model contrasts
#     accuracy_type = f'{word_model_name}_top_word_accuracy'
#     quadrant_accuracy_type = f'{word_model_name}_avg_accuracy'
    
#    # Calculate weights DataFrame
#     df_weights = calculate_weights(df_human_models, accuracy_type)
    
#     # Get statistics for the selected weight type
#     weights = df_weights[weight_type].to_numpy()
#     weights = linear_norm(weights, -1, 1)

#     # Set up plot
#     sns.set_theme(style="white")
#     cmap = cm.get_cmap(cmap_name)

#     # Create joint plot
#     g = sns.JointGrid(data=model_results, x=quadrant_accuracy_type, y="entropy", space=0)

#     # # Create custom norm for colormap centered at mean
#     # norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

#     # # Get the density range from the KDE computation first
#     # from scipy.stats import gaussian_kde

#     # # Compute 2D KDE manually to get density range
#     # x = model_results[accuracy_type].to_numpy()
#     # y = model_results['entropy'].to_numpy()
#     # xy_coords = np.vstack([x, y])  # where x and y are your data points
#     # kde = gaussian_kde(xy_coords, weights=weights)
#     # density = kde(xy_coords)

#     # # Create a custom normalization centered on the midpoint
#     # density_min, density_midpoint, density_max = density.min(), density.mean(), density.max()
#     # norm = TwoSlopeNorm(vmin=density_min, vcenter=density_midpoint, vmax=density_max)

#     # # Create evenly spaced ticks
#     # ticks = np.linspace(density_min, density_max, 6)  # 6 ticks for good distribution

#     # print (weights.min(), weights.max(), weights.mean())

#     # Add scatter and density plots
#     g.plot_joint(sns.scatterplot, color="k", alpha=0.75, s=30)

#     g.plot_joint(
#         sns.kdeplot,
#         weights=weights,
#         bw_adjust=bw_adjust,
#         fill=True,
#         thresh=bw_adjust*weights.mean(),
#         # hue_norm=norm,
#         levels=100,
#         cmap=cmap,
#         alpha=0.5,
#         cbar=True,
#         # cbar_kws={
#         #     # 'norm': norm,
#         #     'ticks': ticks,
#         #     'format': '%.4f'  # Format to 4 decimal places
#         # }
#     )

#     g.plot_marginals(sns.kdeplot, color=cmap(0.5), fill=True, bw_adjust=bw_adjust)
    
#     # Add median lines
#     x_median = np.nanmedian(model_results[quadrant_accuracy_type])
#     y_median = np.nanmedian(model_results['entropy'])
#     xmin, xmax = model_results[quadrant_accuracy_type].min(), model_results[quadrant_accuracy_type].max()
#     ymin, ymax = model_results['entropy'].min(), model_results['entropy'].max()
    
#     xmax *= 1.05
#     xmin -= 0.25 * xmax

#     ymax *= 1.05
#     ymin -= 0.25 * ymin
    
#     g.fig.axes[0].set_ylim(0, ymax)
#     g.fig.axes[0].set_xlim(0, xmax)
#     g.fig.axes[0].vlines(x=x_median, ymin=ymin, ymax=ymax, linestyles='dashed', color='.33')
#     g.fig.axes[0].hlines(y=y_median, xmin=xmin, xmax=xmax, linestyles='dashed', color='.33')
    
#     # Adjust layout
#     plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
#     pos_joint_ax = g.ax_joint.get_position()
#     pos_marg_x_ax = g.ax_marg_x.get_position()
#     g.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
#     g.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    
#     # # Set labels
#     # plt.xlabel('Continuous Accuracy')
#     # plt.ylabel('GPT2-XL Entropy')
    
#     return g