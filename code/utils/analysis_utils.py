import numpy as np
import sys, glob
from os.path import join, exists
from nilearn import image
import nibabel as nib

from scipy.stats import levene, ttest_rel, f 
from statsmodels.stats.multitest import multipletests
import operator
from itertools import combinations, combinations_with_replacement
from scipy.spatial.distance import cdist

from brainiak.funcalign import srm

def get_submatrices(matrices, submatrix_shape, replacement=False):
    '''
    Divide a square matrix into a series of submatrices
    of a supplied shape
    '''
    from skimage.util import view_as_blocks
    
    # divide a square matrix into submatrices of a provided shape --> goes rows first, then columns
    blocks = view_as_blocks(matrices, submatrix_shape).squeeze()
    
    combination_method = combinations if not replacement else combinations_with_replacement
    submatrices = []
    for x,y in combination_method(np.arange(blocks.shape[0]), 2):
        block = np.mean(([blocks[x,y,...], blocks[y,x,...]]), axis=0)
        submatrices.append(block)
    return np.stack(submatrices).squeeze()

def get_roi_coverage(atlas, masker, threshold=None):
    # first get raw roi_idxs, then also generate them with masking
    roi_idxs = get_roi_idxs(atlas)
    masked_roi_idxs = get_roi_idxs(atlas, masker)
    
    # based on our group mask, calculate ratio of coverage as masked/raw roi size
    coverage = np.stack([masked.size/raw.size for raw, masked in zip(roi_idxs, masked_roi_idxs)])
    
    # remove ROIs that don't meet our current threshold
    if threshold:
        coverage[coverage <= threshold] = None
    
    return coverage

def run_srm(dss, n_iter=10, n_features=50, out_fn=None):
    
    ## DOESN'T PLAY NICE WITH MPI ATM
    n_subs, n_voxels, n_trs = dss.shape
    dss = list(dss)
    
    if n_voxels < n_features:
        print (f'Less voxels than features, setting n_features to be: {n_voxels}')
        n_features = n_voxels
    
    if out_fn and exists(out_fn):
        model = srm.load(out_fn + '.npz')
    else:
        model = srm.SRM(n_iter=n_iter, features=n_features)
        model.fit(dss)

    projected = model.transform(dss)
    reconstructed = [basis.dot(projection) for basis, projection in zip(model.w_, projected)]

    if out_fn:
        model.save(out_fn)
    
    return model, projected, reconstructed

def assemble_piecewise_reconstruction(shape, idxs, reconstructed):
    
    assembled = np.zeros(shape)
    
    for idx, recon in zip(idxs, reconstructed):
        assembled[:, idx, :] = np.stack(recon)
        
    return assembled

def p_from_null(observed, distribution, n_samples, mult_comp_method=None, stat_direction=None):
    
    if not stat_direction:
        numerator = np.sum(np.abs(distribution) >= np.abs(observed), axis=0) 
    else:
        numerator = np.sum(stat_direction(np.abs(distribution), np.abs(observed)), axis=0) 
    
    p = (numerator + 1) / (n_samples + 1)
    
    if len(p.shape) > 1:
        p = np.stack([multipletests(p[...,i], method='fdr_bh')[1] for i in range(p.shape[-1])], axis=-1)
    else:
        p = multipletests(p, method='fdr_bh')[1]
        
    return p

def pvalue_threshold(observed, p_values, alpha=0.05):

    observed_flat, p_flat = observed.flatten(), p_values.flatten()

    if np.argwhere(p_flat >= alpha).any():
         observed_flat[np.argwhere(p_flat >= alpha)] = np.nan

    observed = observed_flat.reshape(observed.shape)

    return observed

def compute_task_similarity(dss, observed, permutation=None, metric='correlation', index='within-across', replacement=False):
    '''
    
    Compute task similarity (derived from distances) between all task results 
    derived from provided keys.
    
    Inputs:
        - dss: 3D np.array of participant beta values (tasks, participants, voxels)
        - keys: dictionary keys. Regressors that are common across all
            tasks in tasks_dss.
        - observed: list of 2D arrays. Rows are participants, columns are
            indices of each ROI.
        - permutation: list of 2D arrays. Rows are participants, columns are
            indices of each ROI. If conducting permutation test, either rows
            or columns can be permuted (or both if you're feeling promiscuous).
        - metric: string. Type of distance to use.     
    
    '''
    
    matrices = compute_task_distance(dss, observed, permutation=permutation, metric=metric, replacement=replacement)
    similarity_index = get_similarity_index(matrices, index=index)
    
    # turn correlation distance into correlation if not using within-across idx
    if index in ['within', 'across'] and metric == 'correlation':
        similarity_index = 1 - similarity_index
    
    return matrices, similarity_index

def compute_task_distance(dss, observed, permutation=None, metric='correlation', replacement=False):
    '''
    
    Compute distance between task results for a given key.
    
    Inputs:
        - dss: 3D np.array of participant beta values (tasks, participants, voxels)
        - observed: list of 2D arrays. Rows are participants, columns are
            indices of each ROI.
        - permutation: list of 2D arrays. Rows are participants, columns are
            indices of each ROI. If conducting permutation test, either rows
            or columns can be permuted (or both if you're feeling promiscuous).
       - metric: string. Type of distance to use. 
    
    Outputs:
        - task_distances: 3D np.array of similarity matrices (task_comparisons, rois, participants)
    '''
    
    if not permutation:
        permutation = observed

    combination_method = combinations if not replacement else combinations_with_replacement

    if len(dss) == 1:
        combs = [(dss.squeeze(), dss.squeeze())]
    else:
        combs = combination_method(dss, 2)

    task_distances = np.stack([compute_roi_distance(x=x_, 
                                                    y=y_,
                                                    observed=observed,
                                                    permutation=permutation,
                                                    metric=metric) for x_, y_ in combs])
    
    return task_distances


def compute_roi_distance(x, y, observed, permutation, metric='correlation'):
    '''
    
    Inputs:
        - x: np.array. Rows are participants, columns are voxels.
        - y: np.array. Rows are participants, columns are voxels.
        - observed: list of 2D arrays. Rows are participants, columns are
            indices of each ROI.
        - permutation: list of 2D arrays. Rows are participants, columns are
            indices of each ROI. If conducting permutation test, either rows
            or columns can be permuted (or both if you're feeling promiscuous).
        - metric: string. Type of distance to use.
    
    '''
    
    #ensure the two arrays are the same size
    assert (x.shape == y.shape)
    
    roi_distances = np.stack([cdist(x[idxs_x],
                                    y[idxs_y], 
                                    metric=metric) for idxs_x, idxs_y in zip(observed, permutation)])
    
    return roi_distances

def get_similarity_index(matrices, index='within-across'):
    '''
    
    Calculate an index from a set of similarity matrices.
    
    Inputs:
        - matrices: 3D np.array of similarity matrices (task_pair, roi, participant)
        - analysis: default is within-across. If None, will calculate the within-across
            beta fingerprint. Other options are 'within' and 'across'.
    
    '''
    
    #average across task pairs
    rois = np.mean(matrices, axis=0)

    # for within-across: magnitude will remain the same, but we want larger numbers 
    # to represent "more similarity" and negative to represent "less similarity"
    # therefore we multiply by -1
    fxs = {'within': lambda x: np.diag(x),
           'across': lambda x: (x.sum(1) - np.diag(x))/(x.shape[1]-1),
           'within-across': lambda x: -1*(fxs['within'](x) - fxs['across'](x)),
           'distinctiveness': lambda x: fxs['within-across'](x) / np.std(fxs['across'](x))
    }
    
    similarity_index = np.asarray(list(map(fxs[index], rois)))
    
    return similarity_index

def results2parcellation(mask, atlas, coverage, rois, results, threshold=None):
    from nilearn.image import new_img_like
    
    # atlas = mask.transform(atlas)
    atlas_shape = atlas.get_fdata().shape
    temp = np.empty(np.prod(atlas_shape))
    temp[:] = np.nan

    for i, roi in enumerate(rois):
        if ~np.isnan(coverage[i]):
            temp[roi] = results[i]
            
    temp = temp.reshape(atlas_shape)
    temp = new_img_like(atlas, temp)
    
    # if len(results.shape) > 1:
    #     temp = np.empty((results.shape[-1], atlas.shape[-1]))
    # else:
    #     temp = np.empty(atlas.shape)

    # temp[:] = np.nan
    
    # for i, roi in enumerate(rois):
    #     if ~np.isnan(coverage[i]):
    #         temp[..., roi] = results[i][..., np.newaxis]
        
    # #reshape each of the datasets to desired shape
    # results = mask.inverse_transform(temp)
    
    return temp

def get_roi_idxs(atlas, mask=None):
    '''
    
    Given a parcellation file, load the file into a dictionary
    of ROI label keys and ROI indices values.
    
    Inputs:
        - mask: NiftiMasker. The masker for our data
        - atlas: Nifti1Image. The parcellation scheme we will use
        
    Outputs:
        - roi_idxs: dict. The array indices of each ROI in the 
            provided atlas
    
    '''
    
    if mask:
        atlas = mask.transform(atlas).squeeze()
    else:
        atlas = atlas.get_fdata().flatten()

    rois = [np.where(atlas == i)[0] for i in range(1, len(np.unique(atlas)))]
    
    return rois

def load_narratives_betas(results_dir, subs, model, regressor, task, space, signal_type, subbrik):
    if 'omnimodel' in model:
        dss_fns = sorted(list(map(lambda x: join(results_dir, x,
                        f'{x}_task-{task}_space-{space}_res-native_desc-{model}_{signal_type}.nii.gz'), subs)))
    else:
        
        dss_fns = sorted(list(map(lambda x: join(results_dir, x,
                        f'{x}_task-{task}_space-{space}_res-native_desc-{model}-{regressor}_{signal_type}.nii.gz'), subs)))
        
    #load a given subbrik from each ds
    dss = [nib.load(fn).slicer[..., subbrik] for fn in dss_fns]

    print (f'Loaded {len(dss)} subjects for {task} - {model}-{regressor}')
        
    return dss

def fetch_resample_schaeffer(mask, scale=400, networks=7, resolution=2, data_dir=None):
    ''''
    Adapted from fmrialign utils - https://github.com/neurodatascience/fmralign-benchmark
    
    Inputs:
        - mask: Nifti image. Brain mask to map parcellation.
        - scale: int. Number of parcels in parcellation  {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}.
        - networks: int. Number of networks in the parcellation {7, 17}.
        - resolution: int. Resolution of the parcellation voxels in mm {1, 2}.
        - data_dir: str. Path of where to download parcellations.
    '''
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.image import resample_to_img
    atlas = fetch_atlas_schaefer_2018(
        n_rois=scale, yeo_networks=networks, resolution_mm=resolution, data_dir=data_dir)["maps"]
    resampled_atlas = resample_to_img(
        atlas, mask, interpolation='nearest')
    return resampled_atlas

def voxelwise_paired_ttest(ds_a, ds_b, axis=0, mult_comp_method=None, alpha=0.05):
    '''
    Conduct a paired ttest of two datasets.
    
    Inputs:
        - ds_a: flattened np.array()
        - ds_b: flattened np.array()
        - axis: the axis of the paired samples. Default is 0.
        - mult_comp_method: conduct multiple comparisons correct
            using a provided method. Default is None (no FDR correction).
        - alpha: level of the statistical test.
            
    Outputs:
        - means: difference of sample means (ds_a - ds_b).
        - tstats: resulting tvalues from the paired test.
        - pvals: resulting pvalues from the paired test. These will
            be corrected values if mult_comp_method is not None.
    
    '''
    
    assert ds_a.shape == ds_b.shape
    
    tstats, pvals = ttest_rel(ds_a, ds_b, axis=axis)
    
    diff_means = np.mean(ds_a - ds_b, axis=axis)

    if mult_comp_method is not None:
        pvals[np.isnan(pvals)] = 1
        pvals = multipletests(pvals, alpha=alpha, method=mult_comp_method)[1]
    
    return diff_means, tstats, pvals

def voxelwise_ftest(ds_a, ds_b, axis=0, mult_comp_method=None, alpha=0.05):
    '''
    Conduct an F-test on two datasets. Finds differences
    in variance such that var(ds_a) > var(ds_b).
    
    Inputs:
        - ds_a: 2D np.array(). Dataset of interest for which
            the F-statistic will be calculated
        - ds_b: Type is list of 2D np.array(). Will average
            variances across datasets within list and conduct
            F-test on the average.
        - mult_comp_method: conduct multiple comparisons correct
            using a provided method. Default is None (no FDR correction).
        - alpha: level of the statistical test.
    Returns:
        - stds: difference of sample standard deviations (ds_a - ds_b).
        - tstats: resulting tvalues from the paired test.
        - pvals: resulting pvalues from the paired test. These will
            be corrected values if mult_comp_method is not None.
    '''
    
    assert np.all([ds_a.shape == ds.shape for ds in ds_b])
    
    dfn = ds_a.shape[axis] - 1
    dfd = ds_b[0].shape[axis] - 1
    
    num = np.var(ds_a, axis=axis)
    denom = np.mean([np.var(ds, axis=axis) for ds in ds_b], axis=0)

    diff_vars = num - denom
    fstats = num/denom
    pvals = 1-f.cdf(fstats, dfn, dfd)

#     fstats, pvals = map(np.array, zip(*[levene(ds_a[:, vox], ds_b[:, vox]) for vox in range(shape[1])]))
    
#     diff_stds = np.std(ds_a, axis=0) - np.std(ds_b, axis=0)
    
    #samples with 0 variance will take on nan value --> replace
    if mult_comp_method is not None:
        pvals[np.isnan(pvals)] = 1
        pvals = multipletests(pvals, alpha=0.05, method=mult_comp_method)[1]
    
    return diff_vars, fstats, pvals

def get_stat_intersection_mask(stats, pvals, shape, axis=0, stat_direction=None, stat_level=0, alpha=0.05):
    '''
    Create a statistical map by finding the intersection 
    of significant values across one or more statistical 
    tests. If only one test, FDR correction is required.
    For multiple tests in place of FDR correction (same concept).
    
    Inputs:
        - stats: 2D np.array() of test statistics.
        - pvals: 2D np.array() of pvalues.
        - axis: Type is int. Axis of different statistical tests.
            Default is 0.
        - stat_direction: Type is operator. Direction to look for
            significance in (e.g., less than, greater than)
        - stat_level: Type is float. Level at which to look for
            significance above/below.
        - alpha: Type is float. Level to look for intersection
            across pvals.Default is 0.05.
        
    Returns:
        - idxs: 1D np.array() of pvalue indices less than
            the provided alpha across all tests.
    '''
    
    if stat_direction is not None:
        idxs = np.argwhere(np.logical_and(np.all(pvals < alpha, axis=axis),
                                   np.all(stat_direction(stats, stat_level), axis=axis)))
    else:
        idxs = np.argwhere(np.all(pvals < alpha, axis=axis))
        
    mask = np.zeros(shape)
    mask[idxs] = 1
    
    return mask