import os, sys
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from scipy import stats
from scipy.spatial.distance import cdist

from config import *
import nlp_utils as nlp
from preproc_utils import load_model_results
from text_utils import get_lemma, strip_punctuation


###############################################
########## Lemmatization functions ############
###############################################

def make_transcript_context(word, df_transcript, word_index, range_display=10):
    """
    Create a context for the given word using the transcript information.

    Parameters:
    word (str): The word for which the context needs to be created.
    df_transcript (pandas.DataFrame): DataFrame containing the transcript information.
    word_index (int): The index of the word in the transcript.
    range_display (int, optional): The number of words to include before and after the target word. Defaults to 10.

    Returns:
    tuple:
        - context (str): The context for the given word.
        - index (int): The index of the word within the context.
    """
    # Calculate the start and end indices for the context
    start_index = max(0, word_index - range_display)
    end_index = min(len(df_transcript), word_index + range_display + 1)

    # Get the words before and after the target word
    start_context = df_transcript['word'].iloc[start_index:word_index]
    end_context = df_transcript['word'].iloc[word_index + 1:end_index]

    # Construct the full context
    context = " ".join(start_context) + " " + str(word) + " " + " ".join(end_context)

    # Calculate the index of the target word within the context
    index = word_index - start_index

    return context, index

def lemmatize_word(word, df_transcript, word_index, remove_stopwords=False):
    """
    Lemmatize a word using the context from the transcript.

    Parameters:
    word (str): The word to be lemmatized.
    df_transcript (pandas.DataFrame): DataFrame containing the transcript information.
    word_index (int): The index of the word in the transcript.
    remove_stopwords (bool, optional): If True, remove stopwords from the lemmatized word. Defaults to False.

    Returns:
    str: The lemmatized word.
    """
    context, index = make_transcript_context(word, df_transcript, word_index)
    lemmatized_word, _ = get_pos_tags([context])[index]
    lemmatized_word = get_lemma(lemmatized_word, _, remove_stopwords=remove_stopwords)
    return lemmatized_word

def lemmatize_responses(df_results, df_transcript, response_column='response', debug=False):
    """
    Lemmatize the responses and ground truth words in the given DataFrame.

    Parameters:
    df_results (pandas.DataFrame): DataFrame containing the participant responses.
    df_transcript (pandas.DataFrame): DataFrame containing the transcript information.
    response_column (str, optional): Name of the column containing the responses. Defaults to 'response'.
    debug (bool, optional): If True, print the original and lemmatized words. Defaults to False.

    Returns:
    pandas.DataFrame: The input DataFrame with lemmatized responses and ground truth.
    """

    print (f'Lemmatizing column: {response_column}')

    for _, row in df_results.iterrows():

        # Lemmatize the response
        response_lemma = lemmatize_word(
            word=row[response_column], 
            df_transcript=df_transcript, 
            word_index=row['word_index']
        )

        # Replace response with the lemma
        df_results.at[_, response_column] = response_lemma

        if debug:
            print(f'Word: {row[response_column]} \t Lemma: {response_lemma}')
    
    return df_results

# def calculate_results_accuracy(df_results):

#     # compare response to ground truth --> cast as integer
#     df_results['accuracy'] = df_results['response'] == df_results['ground_truth']
#     df_results['accuracy'] = df_results['accuracy'].astype(int)

#     df_accuracy = df_results.groupby(['prolific_id', 'modality', 'subject'])['accuracy'].mean() \
#     .reset_index() \
#     .sort_values(by='accuracy', ascending=True)

#     return df_results, df_accuracy

###############################################
###### Human data compilation functions #######
###############################################

def get_audio_duration(filepath):
    """
    Gets the duration of an audio file in milliseconds.

    Parameters:
    filepath (str): Path to the audio file.

    Returns:
    float: Duration of the audio file in milliseconds.
    """
    y, sr = librosa.load(filepath, sr=None)
    return librosa.get_duration(y=y, sr=sr) * 1000

def get_subject_audio_durations(stim_dir, n_orders=3):
    """
    Get the audio durations for all stimulus files in a directory.

    Parameters:
    stim_dir (str): Directory path containing the stimulus files.
    n_orders (int, optional): Number of stimulus presentation orders. Defaults to 3.

    Returns:
    pandas.DataFrame: DataFrame containing the stimulus order, audio filename, and duration.
    """
    # Create a DataFrame to store the results
    columns = ['stim_order', 'audio_filename', 'duration']
    df = pd.DataFrame(columns=columns)

    # Iterate over the stimulus orders
    for order in range(1, n_orders + 1):
        order_dir = os.path.join(stim_dir, f'sub-{str(order).zfill(5)}')
        audio_files = sorted(glob.glob(os.path.join(order_dir, '*')))

        # Iterate over the audio files and get the durations
        for audio_file in audio_files:
            duration = get_audio_duration(audio_file)
            df = df.append({
                'stim_order': order - 1,
                'audio_filename': audio_file,
                'duration': duration
            }, ignore_index=True)

    return df

def load_participant_results(sub_dir, sub):
    """
    Load and preprocess participant results from a CSV file.

    Parameters:
    sub_dir (str): Directory path containing the CSV file.
    sub (str): Participant ID.

    Returns:
    tuple:
        - prolific_id (str): The participant's Prolific ID.
        - demographics (pandas.DataFrame): Demographic information (age, race, ethnicity, gender).
        - experience (pandas.DataFrame): Participant's experience with moths and stories.
        - responses (pandas.DataFrame): Participant's responses during the test phase.
    """

    # Load and filter down to response trials
    df_results = pd.read_csv(os.path.join(sub_dir, f'{sub}_next-word-prediction.csv')).fillna(False)
    df_results['word_index'] = df_results['word_index'].astype(int)

    # Grab the prolific id
    prolific_id = list(set(df_results['prolific_id']))[0]

    # Filter down demographics
    demographics = df_results[df_results['experiment_phase'].str.contains('demographics').fillna(False)]
    demographics = demographics[['experiment_phase', 'response']].reset_index(drop=True)
    assert len(demographics) == 4, "Expected 4 demographic answers"

    # Filter down to questions about moth/story experience
    experience = df_results[df_results['experiment_phase'].str.contains('experience').fillna(False)]
    experience = experience[['experiment_phase', 'response']].reset_index(drop=True)
    assert len(experience) == 2, "Expected 2 experience answers"

    # Filter down to get the responses
    responses = df_results[df_results['experiment_phase'] == 'test']
    responses.loc[:,'response'] = responses['response'].str.lower()
    responses = responses[['critical_word', 'word_index', 'entropy_group', 'accuracy_group', 'response', 'rt']].reset_index(drop=True)

    return prolific_id, demographics, experience, responses


def aggregate_participant_responses(results_dir, stim_dir, task, modality, n_orders=3, debug=False):
    """
    Aggregate participant responses for a given task and modality.

    Parameters:
    results_dir (str): Directory path containing the participant results.
    stim_dir (str): Directory path containing the stimulus files.
    task (str): Name of the task.
    modality (str): Modality of the task (e.g., 'audio', 'visual').
    n_orders (int, optional): Number of stimulus presentation orders. Defaults to 3.

    Returns:
    pandas.DataFrame: Aggregated participant responses.
    """
    print(f'Aggregating {task} - {modality}')

    # Initialize the results DataFrame
    columns = ['prolific_id', 'modality', 'subject', 'word_index', 'response', 'ground_truth', 'entropy_group', 'accuracy_group', 'rt']
    df_results = pd.DataFrame(columns=columns)

    # Get the subject audio durations
    df_order_durations = get_subject_audio_durations(os.path.join(stim_dir, task), n_orders=n_orders)

    # Get subject directories
    sub_dirs = sorted(glob.glob(os.path.join(results_dir, task, modality, f'sub*')))
    
    print(f'Total of {len(sub_dirs)} subjects')

    for sub_dir in tqdm(sub_dirs):
        sub = os.path.basename(sub_dir)

        # Get the current stimulus order
        current_order = (int(sub.split('-')[-1]) - 1) % n_orders
        df_duration = df_order_durations[df_order_durations['stim_order'] == current_order].reset_index(drop=True)

        if debug:
            print(f'Subject: {sub}')
            print(f'Current order: {current_order}')

        if os.path.exists(sub_dir):
            # Load participant results
            current_id, demographics, experience, responses = load_participant_results(sub_dir, sub)
            responses['response'] = responses['response'].fillna('')

            # Append responses to the results DataFrame
            for index, row in responses.iterrows():
                df_results = df_results.append({
                    'prolific_id': current_id,
                    'modality': modality,
                    'subject': sub,
                    'word_index': row['word_index'],
                    'response': row['response'],
                    'ground_truth': row['critical_word'].lower(),
                    'entropy_group': row['entropy_group'],
                    'accuracy_group': row['accuracy_group'],
                    'rt': float(row['rt']) - df_duration.loc[index, 'duration']
                }, ignore_index=True)
        else:
            print(f'File not exists: {modality}, {sub}')

    # Calculate results one-shot accuracy
    df_results['accuracy'] = (df_results['response'] == df_results['ground_truth']).astype(int)

    return df_results

###############################################
########### Analysis of human data ############
###############################################

def get_human_probs(responses):
    
    unique, counts = np.unique(responses, return_counts=True)
    probs = counts / sum(counts)
    
    return probs, unique

# def strip_punctuation(text):
    
#     full_text = re.sub('[^A-Za-z0-9]+', '', text)
    
#     return full_text

def analyze_human_results(df_transcript, df_results, word_model_info, window_size=25, top_n=None, drop_rt=None):
    """

    Compile results from all subjects into a distribution of "humans"

    """

    # Decide if we want to filter RTs
    if drop_rt:
        print (f'Dropping trials with RTs longer than {drop_rt} seconds')
        drop_rt = drop_rt * 1000

    # Load word-level and sentence-level models
    word_model_name, word_model = word_model_info
    modality = np.unique(df_results['modality'])[0]

    # Instantiate the dataframe
    df_analysis = pd.DataFrame(columns=[
        'modality',
        'word_index',
        'ground_truth',
        'top_pred',
        'accuracy',
        'predictability',
        'top_prob',
        'n_predictions',
        'entropy',
        'normalized_entropy',
        'bert_top_word_accuracy',
        f'{word_model_name}_top_word_accuracy',
        f'{word_model_name}_avg_accuracy',
        f'{word_model_name}_max_accuracy',
        f'{word_model_name}_weighted_pred-gt_accuracy',
        f'{word_model_name}_prediction_distances',
        f'{word_model_name}_weighted_prediction_distances',
        f'{word_model_name}_centroid_prediction_distances',
        'entropy_group',
        'accuracy_group',
        'n_rt_drops',
        'average_rt',
        'std_rt',
    ])

    # Load sentence transformer + get indices of transcript segments for the current window size
    tokenizer, model = nlp.load_mlm_model(model_name='sentence-transformers/all-mpnet-base-v2', cache_dir=CACHE_DIR)
    segments = nlp.get_segment_indices(n_words=len(df_transcript), window_size=window_size, bidirectional=True)
    
    # Go through each response word 
    for index, df_index in tqdm(df_results.groupby('word_index')):

        # Grab global information across participants
        ground_truth, entropy_group, accuracy_group = df_index \
            .reset_index(drop=True) \
            .loc[0, ['ground_truth', 'entropy_group', 'accuracy_group']]
        
        ##############################################
        ### If we want to drop trials based on RTs ###
        ##############################################

        # find RTs less than the desired RT and filter responses
        if drop_rt:
            rt_filter = df_index['rt'] <= drop_rt
            df_index = df_index.loc[rt_filter, :]
        
            print (f'Kept {sum(rt_filter)} responses')
    
        human_responses = df_index['response']
        rts = df_index['rt']

        ####################################################
        ### Get probabilities for words across responses ###
        ####################################################

        # get probabilities and unique words
        human_probs, unique_words = get_human_probs(human_responses)
        
        #######################################
        #### Calculate probability metrics ####
        #### 1. Predictability             ####
        #### 2. Entropy                    ####
        #### 3. Top probabiltity           ####
        #######################################

        # predictability is the count of number of accurate predictions over total predictions
        predictability = sum(np.asarray(human_responses) == ground_truth) / len(human_responses)
        
        # entropy + entropy normalized by the number of items in the distribution
        entropy = stats.entropy(human_probs)
        normalized_entropy = entropy / np.log(len(human_probs))

        #############################################
        #### Calculate accuracy metrics          ####
        #### 1. Binary accuracy                  ####
        #### 2. Word continuous accuracy         ####
        #### 3. Contextual continuous accuracy   ####
        #############################################

        # Sort probabilities from highest to lowest
        sorted_prob_idxs = np.argsort(human_probs)[::-1]
        sorted_probs = human_probs[sorted_prob_idxs]
    
        # If we are using top-N words (e.g., top 5), trim down to those words
        if top_n is not None and len(unique_words) < top_n:
            # Select top-N from all unique words and sort from most to least
            all_word_idxs = sorted_prob_idxs[:len(unique_words)]
            top_n_words = unique_words[all_word_idxs]
        else:
            # Otherwise top-N words are all unique words --> just sort the words
            top_word_idxs = sorted_prob_idxs[:top_n]
            top_n_words = unique_words[top_word_idxs]

        # Top word is the first word -- Top prob is the associated probability
        top_word = top_n_words[0]
        top_prob = sorted_probs[0]

        #####################################################
        ####### 1. Binary accuracy: one-shot accuracy #######
        #####################################################

        accuracy = int(top_word == ground_truth)

        ##########################################################################
        ####### 2. Word continuous accuracy:                             #########
        #######    Cosine similarity between prediction and ground-truth #########
        #######    using the current word model                          #########
        ##########################################################################

        top_word_accuracy, _ = nlp.get_word_vector_metrics(word_model, [top_word], ground_truth[0])
        max_pred_similarity, _ = nlp.get_word_vector_metrics(word_model, top_n_words, ground_truth[0], method='max')

        ##########################################################################
        ####### 3. Contextual continuous accuracy:                       #########
        #######    Cosine similarity in BERT space using surrounding     #########
        #######    transcript context for prediction and ground-truth    #########
        ##########################################################################

        # Find index of the word within its surrounding context
        current_segment = segments[index]
        word_index = np.where(current_segment == index)[0]

        # Create a dataframe and substitute the top word
        df_substitute = df_transcript.copy()
    
        # Make input and calculate contextual embedding for ground truth
        df_substitute.loc[index, 'word'] = ground_truth
        inputs = nlp.transcript_to_input(df_substitute, idxs=current_segment)
        bert_ground_truth_embedding = nlp.extract_word_embeddings([inputs], tokenizer, model, word_indices=word_index).squeeze()
        bert_ground_truth_embedding = bert_ground_truth_embedding[-1, :][np.newaxis]

        # Repeat for the predicted word
        df_substitute.loc[index, 'word'] = top_word
        inputs = nlp.transcript_to_input(df_substitute, idxs=current_segment)
        bert_top_word_embedding = nlp.extract_word_embeddings([inputs], tokenizer, model, word_indices=word_index).squeeze()
        bert_top_word_embedding = bert_top_word_embedding[-1, :][np.newaxis]

        # Calculate cosine similarity between ground truth and prediction   
        bert_similarity = 1 - cdist(bert_ground_truth_embedding, bert_top_word_embedding, metric='cosine').squeeze()

        #############################################
        #### Prediction density metrics          ####
        #### 1. Binary accuracy                  ####
        #### 2. Word continuous accuracy         ####
        #### 3. Contextual continuous accuracy   ####
        #############################################

        # pred_similarity = similarity of all words to ground truth word
        # pred_distances = similarity of all words from each other
        pred_similarity, pred_distances = nlp.get_word_vector_metrics(word_model, top_n_words, ground_truth[0])
        
        # Weight these scores by the probability (e.g., more probable items contribute more)
        weighted_pred_distances = np.nanmean(pred_distances * sorted_probs)
        weighted_pred_similarity = np.nanmean(pred_similarity * sorted_probs)

        # Calculate distance spread of predicted vectors from the centroid of the vectors
        predicted_vectors = [word_model[word] for word in unique_words if word in word_model]
        predicted_vectors = np.stack(predicted_vectors)

        centroid_distance = cdist(predicted_vectors.mean(0)[np.newaxis], predicted_vectors, metric='cosine')
        centroid_distance = np.nanmean(centroid_distance)

        # pred_distances = nan when there was only one prediction
        if np.isnan(pred_distances):
            pred_distances = 0

        df_analysis.loc[len(df_analysis)] = {
            'modality': modality,
            'word_index': index,
            'ground_truth': ground_truth,
            'top_pred': top_word,
            'accuracy': accuracy,
            'predictability': predictability,
            'top_prob': top_prob,
            'n_predictions': len(unique_words),
            'entropy': entropy,
            'normalized_entropy': np.nan_to_num(normalized_entropy), 
            'bert_top_word_accuracy': bert_similarity,
            f'{word_model_name}_top_word_accuracy': top_word_accuracy,
            f'{word_model_name}_avg_accuracy': np.nanmean(pred_similarity),
            f'{word_model_name}_max_accuracy': max_pred_similarity,
            f'{word_model_name}_weighted_pred-gt_accuracy': weighted_pred_similarity,
            f'{word_model_name}_prediction_distances': pred_distances,
            f'{word_model_name}_weighted_prediction_distances': weighted_pred_distances,
            f'{word_model_name}_centroid_prediction_distances': centroid_distance,
            'entropy_group': entropy_group,
            'accuracy_group': accuracy_group,
            'n_rt_drops': sum(rt_filter) if drop_rt is not None else 0,
            'average_rt': rts.mean(),
            'std_rt': rts.std(),
        }
    
    return df_analysis

###############################################
########### Analysis of LLM data ##############
###############################################
    
# def get_model_word_quadrants(df_model_results, df_transcript, task, selected_idxs=None, accuracy_type='fasttext_avg_accuracy', accuracy_percentile=50, top_n=5, window_size=25):
    
#     # # FOR DIVIDING THE MODEL RESULTS INTO QUADRANTS
#     # ACCURACY_TYPE = accuracy_type
#     # ACCURACY_PERCENTILE = 50
#     # WINDOW_SIZE = 100
#     # TOP_N = 5
    
#     # preproc_dir = os.path.join(BASE_DIR, 'stimuli', 'preprocessed')
    
#     # load our preprocessed file --> get the indices of the prediction words
#     # df_preproc = pd.read_csv(os.path.join(preproc_dir, task, f'{task}_transcript-preprocessed.csv'))

#     selected_rows = np.where(df_transcript['NWP_Candidate'])[0]
    
#     # select based on model quadrants --> trim down to only the words of interest
#     # df_model_results = load_model_results(models_dir, model_name=model_name, task=task, window_size=WINDOW_SIZE, top_n=TOP_N)
#     # model_results.loc[:, 'binary_accuracy'] = model_results['binary_accuracy'].astype(bool)
#     # model_results = model_results.iloc[nwp_idxs]
    
#     # now grab the current model divided over the 50th percentile
#     # while we originally divided words on the 45th percentile of gpt2, we want to see patterns across models
#     df_divide = divide_nwp_dataframe(model_results, accuracy_type=accuracy_type, percentile=accuracy_percentile, drop=False)
    
#     return df_divide.loc[selected_idxs, ['entropy_group', 'accuracy_group']]

def analyze_model_accuracy(df_transcript_selected, models_dir, model_name, word_model_info, task, top_n=1, window_size=25, lemmatize=False):
    """
    Perform analysis of model predictions (similar to analyze_human_results). Formats
    the model predictions dataframe similar to the human dataframe so the two can be collapsed.

    """

    # Get word model information
    word_model_name, word_model = word_model_info

    # Rows for words used in the next-word prediction experiment
    # Columns of the model predictions that we care about for comparison
    selected_rows = np.where(df_transcript_selected['NWP_Candidate'])[0]
    selected_columns = ['ground_truth_word', 'top_n_predictions', 'top_prob', 'ground_truth_prob', f'{word_model_name}_avg_accuracy', 'entropy']

    # Load results for the specified model -- this is output from the prediction-extraction script
    df_model_results = load_model_results(models_dir, model_name=model_name, task=task, top_n=top_n, window_size=window_size)

    # Grab model binarized entropy/accuracy quadrants
    # df_model_quadrants = get_model_word_quadrants(model_name, task, selected_rows, accuracy_type=f'{word_model_name}_max_accuracy').reset_index(drop=True)

    # Select only the words used for next-word prediction
    df_model_results = df_model_results.loc[selected_rows, selected_columns].reset_index()
    df_model_results['top_n_predictions'] = df_model_results['top_n_predictions'].str[0] # get the top predicted word

    # Rename the columns to make the dataframe in line with human results
    df_model_results = df_model_results.rename(columns={
            'index': 'word_index',
            'ground_truth_word': 'ground_truth',
            'ground_truth_prob': 'predictability', 
            'top_n_predictions': 'top_pred',
    })

    # Add other information to the dataframe
    df_model_results['modality'] = model_name
    df_model_results['accuracy'] = (df_model_results['top_pred'] == df_model_results['ground_truth']).astype(int) # binary accuracy

    # df_model_results.loc[:, ['entropy_group', 'accuracy_group']] = df_model_quadrants.loc[: ['entropy_group', 'accuracy_group']] 

    if lemmatize:
        # Lemmatize the response and the ground truth
        for response_col in ['top_pred', 'ground_truth']:
            df_model_results = lemmatize_responses(df_model_results, df_transcript_selected, response_column=response_col)

    print (f"Total missing values: {df_model_results[f'{word_model_name}_avg_accuracy'].isna().sum()}")

    return df_model_results


def compare_human_model_distributions(tokenizer, word_model, human_responses, all_responses, model_logits, ground_truth):
    
    df = pd.DataFrame(columns=[
        'top_word_human', 
        'top_word_model',
        'top_word_model_adjusted',
        'prob_human',
        'prob_model',
        'prob_model_adjusted', 
        'prob_model_human_pred',
        'predictability_model',
        'predictability_human',
        'continuous_predictability_human',
        'log_odds_predictability_model',
        'log_odds_predictability_human',
        'log_odds_continuous_predictability_human',
        'entropy',
        'kl_divergence',
        'relative_entropy',
        'wasserstein_dist',
        'jensenshannon_dist',
        'ks_stat'
    ])
    
    pre_filter = len(human_responses)
    human_responses = list(filter(None, human_responses))
    post_filter = len(human_responses)
    
    if pre_filter != post_filter:
        print (f'Removed {pre_filter - post_filter} empty responses')
    
    model_probs = F.softmax(model_logits, dim=-1).squeeze()
    prob_model = model_probs.max().item()
    top_word_model = tokenizer.decode(model_probs.argmax())

    entropy = stats.entropy(model_probs)
    
    ## get ground truth word prob
    gt_token = tokenizer.encode(ground_truth)
    gt_predictability_model = model_probs[gt_token].mean(0).item()
   
    # continuous predictability - average semantic distance of words from ground truth word
    human_predictability = sum(np.asarray(human_responses) == ground_truth) / len(human_responses)
    continuous_predictability = (1 - distance.cdist(word_model[ground_truth][np.newaxis], word_model[human_responses], metric='cosine')).mean()

    if human_predictability == 0:
        log_odds_human_predictability = statistics.log_odds(1e-2)
    else:
        log_odds_human_predictability = statistics.log_odds(human_predictability)

    log_odds_model_predictability = statistics.log_odds(gt_predictability_model)
    log_odds_continuous_predictability = statistics.log_odds(continuous_predictability)
        
    # get the probability distribution of the human responses --> also return the unique words
    human_probs, unique_words = get_human_probs(human_responses)
    prob_human = human_probs.max()
    
    # get the words indices in the overall array then add in the human probs
    word_idxs = [all_responses.index(word) for word in unique_words]    
    temp = np.zeros(len(all_responses))
    temp[word_idxs] = human_probs
    human_probs = temp
    
    # get probability of the words humans chose within the model distribution
    # then normalize to the number of samples
    model_adjusted_probs = np.asarray([nlp.get_word_prob(tokenizer, word, model_logits) for word in all_responses])
    model_adjusted_probs = model_adjusted_probs / model_adjusted_probs.sum()

    # select the probability of the top word that humans chose
    prob_model_adjusted = model_adjusted_probs[model_adjusted_probs.argmax()]
    prob_model_human_pred = model_adjusted_probs[human_probs.argmax()]

    # grab the human and model top words
    top_word_human = all_responses[human_probs.argmax()]
    top_word_model_adjusted = all_responses[model_adjusted_probs.argmax()]

    # now calculate kl divergence between the human and adjusted model distribution
    # measures how different P (human) is from Q (model) distribution
    #  KL divergence of P from Q is the expected excess surprise from 
    #  using Q as a model when the actual distribution is P
    kl_divergence = kl_div(human_probs, model_adjusted_probs)
    kl_divergence[np.isinf(kl_divergence)] = 0
    kl_divergence = kl_divergence.sum().item()
    
    relative_entropy = rel_entr(human_probs, model_adjusted_probs).sum().item()
    
    # earth movers distance between adjusted probs
    wasserstein_dist = stats.wasserstein_distance(human_probs, model_adjusted_probs)
    
    jensenshannon_dist = distance.jensenshannon(human_probs, model_adjusted_probs)
    
    ks_stats = stats.kstest(human_probs, model_adjusted_probs)
    
    df.loc[len(df)] = {
        'top_word_human': top_word_human,
        'top_word_model': top_word_model,
        'top_word_model_adjusted': top_word_model,
        'prob_human': prob_human,
        'prob_model': prob_model,
        'prob_model_adjusted': prob_model_adjusted, 
        'prob_model_human_pred': prob_model_human_pred,
        'predictability_model': gt_predictability_model,
        'predictability_human': human_predictability,
        'continuous_predictability_human': continuous_predictability,
        'log_odds_predictability_human': log_odds_human_predictability.astype(float),
        'log_odds_predictability_model': log_odds_model_predictability.astype(float),
        'log_odds_continuous_predictability_human': log_odds_continuous_predictability.astype(float),
        'entropy': entropy,
        'kl_divergence': kl_divergence,
        'relative_entropy': relative_entropy,
        'wasserstein_dist': wasserstein_dist,
        'jensenshannon_dist': jensenshannon_dist,
        'ks_stat': ks_stats[0]
    }
    
    return df