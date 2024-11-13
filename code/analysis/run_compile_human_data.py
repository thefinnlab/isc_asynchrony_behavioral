import analysis_utils as analysis

if __name__ == '__main__':

    
    EXPERIMENT_NAME = 'next-word-prediction'
    EXPERIMENT_VERSION = 'final-multimodal-01'
    TASK = 'wheretheressmoke'

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-task', '--task', type=str, nargs='+')
    parser.add_argument('-word_model', '--word_model', type=str, default='fasttext')
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    ###############################################
    ####### Set paths and directories needed ######
    ###############################################

    results_dir = os.path.join(BASE_DIR, 'experiments',  EXPERIMENT_NAME, 'results', EXPERIMENT_VERSION)
    preproc_dir = os.path.join(BASE_DIR, 'stimuli/preprocessed')
    models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions')

    # Source results from the cleaned results directory
    results_dir = os.path.join(BASE_DIR, 'experiments',  EXPERIMENT_NAME, 'cleaned-results', EXPERIMENT_VERSION)

    behavioral_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
    stim_dir = os.path.join(BASE_DIR, f'stimuli/cut_audio/{EXPERIMENT_VERSION}')

# utils.attempt_makedirs(behavioral_dir)

    ########################################################
    #### Aggregate results from the specified condition ####
    ########################################################

    df_aggregated_results = analysis.aggregate_participant_responses(cleaned_results_dir, stim_dir, task=p.task, modality=modality, n_orders=3)

    ########################################################
    ###### Lemmatize results and recalculate accuracy ######
    ########################################################

    df_lemmatized_results['accuracy'] = (df_lemmatized_results['response'] == df_lemmatized_results['ground_truth']).astype(int)
