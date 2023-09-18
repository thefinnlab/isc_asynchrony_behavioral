import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from collections import defaultdict

import math, random
import numpy as np
import json
import re

import pandas as pd
import wave, contextlib
from praatio import textgrid as tgio

### FUNCTIONS FOR EDITING GENTLE TRANSCRIPTS

def update_dataframe_from_praat(df, textgrid):
    
    df = df.copy()
    
    for idx in range(len(df)):
        
        word = textgrid.getTier('word').entries[idx]
        
        df.loc[idx, 'Onset'] = word.start
        df.loc[idx, 'Offset'] = word.end
        df.loc[idx, 'Duration'] = word.end - word.start
        
    return df

def dataframe_to_textgrid(df, audio_fn):
    """
    Take a filename and its associated transcription and fill in all the gaps
    """
    
    with contextlib.closing(wave.open(audio_fn, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    rearranged_words = []
    file_ons = 0

    rearranged_words = []

    for ix, word in df.iterrows():

        # if word['Case'] == 'success' or word['Case'] == 'assumed':
        word_ons = np.round(word['Onset'], 3)
        word_off = np.round(word['Offset'], 3)
        target = word['Word_Written']
        rearranged_words.append((word_ons, word_off, target))
        # else:
        #     # search forwards and backwards to find the previous and next word
        #     # use the end and start times to get word times 
        #     target = content['words'][ix]['Word_Written']
        #     prev_end, next_start = align_missing_word(content, ix)
        #     rearranged_words.append((prev_end, next_start, target))

    # adjust for overlap in times
    for ix, word_times in enumerate(rearranged_words):
        if ix != 0:
            prev_start, prev_end, prev_word = rearranged_words[ix-1]
            curr_start, curr_end, curr_word = word_times

            # if the current start time is before the previous end --> adjust
            if curr_start < prev_end:
                rearranged_words[ix] = (prev_end, curr_end, curr_word)
    
    tg = tgio.Textgrid()
    tg.addTier(tgio.IntervalTier('word', rearranged_words))
    return tg

def gentle_to_textgrid(alignment_fn, path):
    """
    Take a filename and its associated transcription and fill in all the gaps
    """
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    rearranged_words = []
    file_ons = 0
    
    # load the alignment file
    with open(alignment_fn, encoding='utf-8') as f:
        content = json.load(f)
    all_ons = content['words'][0]['start']
    
    for ix, word in enumerate(content['words']):
        # if the word was successfully aligned
        if word['case'] == 'success' or word['case'] == 'assumed':
            word_ons = np.round(word['start'], 3)
            word_off = np.round(word['end'], 3)
            target = word['alignedWord']
            rearranged_words.append((word_ons, word_off, target))
        else:
            # search forwards and backwards to find the previous and next word
            # use the end and start times to get word times 
            target = content['words'][ix]['word']
            prev_end, next_start = align_missing_word(content, ix)
            
            rearranged_words.append((prev_end, next_start, target))
    
    # adjust for overlap in times
    for ix, word_times in enumerate(rearranged_words):
        if ix != 0:
            prev_start, prev_end, prev_word = rearranged_words[ix-1]
            curr_start, curr_end, curr_word = word_times

            # if the current start time is before the previous end --> adjust
            if curr_start < prev_end:
                rearranged_words[ix] = (prev_end, curr_end, curr_word)
    
    tg = tgio.Textgrid()
    tg.addTier(tgio.IntervalTier('word', rearranged_words))
    return content, tg

def gentle_fill_missing_words(alignment_fn):
    '''
    A simple way to fill missing aligned words
    '''
    
    # load the alignment file
    with open(alignment_fn, encoding='utf-8') as f:
        content = json.load(f)
        
    for ix, word in enumerate(content['words']):
        if word['case'] != 'success':
            prev_end, next_start = align_missing_word(content, ix)
            content['words'][ix].update({'start': prev_end, 'end': next_start, 'case': 'assumed'})
            
    return content

def align_missing_word(content, ix):
    '''
    Searches from a word in both directions and then distributes time evenly
    '''
    # keep track of how many are missing
    forward_ix = ix
    forward_missing = 0
    
    # search forward
    while True:
        # move one forward
        forward_ix += 1
        if content['words'][forward_ix]['case'] == 'success':
            next_start = np.round(content['words'][forward_ix]['start'], 3)
            break
        else:
            forward_missing += 1
    
    # keep track of how many are missing
    back_ix = ix
    back_missing = 0
    
    while True:
        # move one backwards
        back_ix -= 1
        
        if content['words'][back_ix]['case'] == 'success':
            prev_end = np.round(content['words'][back_ix]['end'], 3)
            break
        else:
            back_missing += 1
    
    # space evenly between the number of missing items
    total_missing = back_missing + forward_missing + 1 # add one to include current item
    x_vals = np.linspace(prev_end, next_start, total_missing + 2)[1:-1] # add 2 to pad the points on either side
    
    # if there is anything missing
    # normalize indices to 0
    missing_ixs = np.arange(ix-back_missing,ix+forward_missing+1)
    
    # index of the value in the interpolated array
    arr_ix = np.argwhere(ix == missing_ixs)
    
    # then extract value from that array and round
    next_start = x_vals[arr_ix].squeeze()
    next_start = np.round(next_start, 3)
    
    # have to adjust prev end to be the interpolated value
    if len(missing_ixs) > 1 and arr_ix:
        prev_end = x_vals[np.argwhere(ix == missing_ixs)-1].squeeze()
        prev_end = np.round(prev_end, 3)
    
    return prev_end, next_start


### FUNCTIONS FOR SELECTING CANDIDATES FOR PREDICTION ###

lemmatizer = WordNetLemmatizer()

# generate the explained tags --> we will use these to make more sense of the outputs
tags_explained = nltk.data.load('help/tagsets/upenn_tagset.pickle')
STOP_WORDS = stopwords.words('english')

STOP_UTTERANCES = ['yes', 'well', 'oh', 'mhm', 'um', 'boom']

STOP_WORDS.extend(STOP_UTTERANCES)

# POS tag mapping, format: {Treebank tag (1st letter only): Wordnet}
tagset_mapping = defaultdict(
	lambda: 'n',   # defaults to noun
	{
		'N': 'n',  # noun types
		'P': 'n',  # pronoun types, predeterminers
		'V': 'v',  # verb types
		'J': 'a',  # adjective types
		'D': 'a',  # determiner
		'R': 'r'   # adverb types
	})


def create_word_prediction_df(align_fn, fill_missing_times=False):
	'''

	Preprocessing to get a dataframe for next word prediction. Applies the following steps:
		1. Loads the file
		2. Applies part of speech tagging to each word (used for lemmatization)
		3. Lemmatizes each word and evaluates whether a stop word
		4. If the word was aligned, adds the times of onset and offset
		5. Lastly interpolates onset times to recover those of any missing words
	'''
	
	# load the alignment file
	if fill_missing_times:
		data = gentle_fill_missing_words(align_fn)
	else:	
		with open(align_fn, encoding='utf-8') as f:
			data = json.load(f)
	
	# grab the original transcript
	transcript = data['transcript']
	words_list = data['words']

	# go and extract each word --> pos tagging here incorporates context
	all_words = [word['word'] for word in words_list]
	_, pos_tags = map(list,zip(*pos_tag(all_words)))

	# go through each word
	df_stack = []

	for i, current_word in enumerate(words_list):
		
		# first tokenize each word --> this makes it easier to 
		tokens = word_tokenize(current_word['word'])
		tag = tagset_mapping[pos_tags[i][0]]

		# as some words may be broken into multiple tokens, we need to lemmatize all tokens
		# then make sure we don't have any stopwords as part of the tokens
		lemmas = [lemmatizer.lemmatize(re.sub("[^a-zA-Z\s-]+", '', token.lower()), pos=tag) for token in tokens]
		stop_word = all([lemma in STOP_WORDS for lemma in lemmas if lemma]) # evaluate if not empty string
		
		# we'll start the word dictionary here, but only add the times if we have the aligned times
		word_dict = {
			'Word_Written': current_word['word'],
			'Case': current_word['case'],
			'POS': pos_tags[i], # extract pos_tag for the word
			'POS_Definition': tags_explained[pos_tags[i]][0], # get the explained tag
			'Punctuation': transcript[current_word['endOffset']:words_list[i+1]['startOffset']]
									  if i+1 < len(words_list) else transcript[current_word['endOffset']:], # punctuation following the word (use subsequent word)
			'Stop_Word': stop_word, # true or false if a stopword
		}

		# make sure that we've aligned the word --> could also check its a word in the vocabulary
		if fill_missing_times or 'alignedWord' in current_word:
			aligned_dict = {
				'Word_Vocab': current_word['word'] if fill_missing_times else current_word['alignedWord'],
				'Onset': current_word['start'],
				'Offset': current_word['end'],
				'Duration': current_word['end'] - current_word['start'], # calculate duration of the current word
			}
			word_dict.update(aligned_dict)
		
		df_stack.append(
			pd.DataFrame(
				word_dict,
				index=[i],
			)
		)
	
	df_stack = pd.concat(df_stack)
	
	# if interpolate_missing_times:
	# 	df_stack['Onset'].interpolate()
	
	return df_stack

def clean_hyphenated_words(df):
	
	print ("\nHYPHEN CLEANING\n You will see a hyphenated word. Enter \'y' if the word is meant to be hyphenated or \'n' if not.\n")
	
	hyphenated = df['Punctuation'].str.contains('-')
	hyphenated_idxs = np.where(hyphenated)[0]
	
	for idx in hyphenated_idxs:
		# grab the current row and following row from the dataframe
		df_rows = df.iloc[idx:idx+2]
		hyphenated_word = '-'.join(df_rows['Word_Written'])
		
		# establish some context for the word
		precontext = ' '.join(df.iloc[idx-10:idx]['Word_Written'])
		postcontext = ' '.join(df.iloc[idx+2:idx+10]['Word_Written'])
		print (f'\nContext: {precontext} ___ {postcontext}')
		print (f'Word: {hyphenated_word}')

		response = input()
	
		if response == 'y':
			
			hyphenated_entry = {
				'Word_Written': hyphenated_word,
				'Case': df_rows['POS'].to_list()[-1],
				'POS': df_rows['POS'].to_list()[-1],
				'POS_Definition': df_rows['POS_Definition'].to_list()[-1],
				'Punctuation': df_rows['Punctuation'].to_list()[-1] ,
				'Stop_Word': hyphenated_word.lower() in STOP_WORDS,
				'Word_Vocab': hyphenated_word,
				'Onset': df_rows['Onset'].to_list()[0],
				'Offset': df_rows['Offset'].to_list()[-1],
				'Duration': df_rows['Offset'].to_list()[-1] - df_rows['Onset'].to_list()[0]
			}
			
			df.at[idx, :] = hyphenated_entry
			df = df.drop(idx+1).reset_index(drop=True)

			# we've dropped an index
			hyphenated_idxs -= 1 
			
			print (f'Word updated to: {hyphenated_word}')
		else:
			# otherwise add padding on each side to ensure it's not hyphenated
			df.at[idx, 'Punctuation'] =  ' - '
			
			hyphenated_word = ' - '.join(df_rows['Word_Written'])
			print (f'Words separated to: {hyphenated_word}')
	  
	df = df.reset_index(drop=True)
	
	return df

def clean_named_entities(df):
	'''
	Label the named entities in the transcript to avoid selecting them as candidates.
	'''
	
	print ("\nNAMED ENTITY CLEANING\nYou will see a potential named entity (e.g., person, place). Enter \'y' if the word is or refers to a named entity and \'n' otherwise.\n")
	
	# fix instructions?
	named_entities = pd.Series(df['POS'] == 'NNP') & pd.Series(df['Stop_Word'] == False)
	named_entity_idxs = np.where(named_entities)[0]
	df['Named_Entity'] = False

	for idx in named_entity_idxs:
		# grab the current row and following row from the dataframe
		df_rows = df.iloc[idx]
		ne_word = df_rows['Word_Written']

		precontext = ' '.join(df.iloc[idx-10:idx]['Word_Written'])
		postcontext = ' '.join(df.iloc[idx+1:idx+10]['Word_Written'])

		precontext = re.sub(u"(\u2018|\u2019)", "'", precontext)
		postcontext = re.sub(u"(\u2018|\u2019)", "'", postcontext)
		print (f'\nContext: {precontext} ___ {postcontext}')
		print (f'Word: {ne_word}')
		
		response = input()
	
		if response == 'y':
			df.at[idx, 'Named_Entity'] = True

	df = df.reset_index(drop=True)
	return df


### FUNCTIONS FOR RANDOMIZATION OF ORDERS #####
### modified from https://stackoverflow.com/questions/93353/create-many-constrained-random-permutation-of-a-list

def get_pool(items, n_elements_per_subject, use_each_times):
	pool = {}
	for n in items:
		pool[n] = use_each_times
	
	return pool

def rebalance(ret, pool, n_elements_per_subject):
	max_item = None
	max_times = None
	
	for item, times in pool.items():
		if max_times is None:
			max_item = item
			max_times = times
		elif times > max_times:
			max_item = item
			max_times = times
	
	next_item, times = max_item, max_times

	candidates = []
	for i in range(len(ret)):
		item = ret[i]

		if next_item not in item:
			candidates.append( (item, i) )
	
	swap, swap_index = random.choice(candidates)

	swapi = []
	for i in range(len(swap)):
		if swap[i] not in pool:
			swapi.append( (swap[i], i) )
	
	which, i = random.choice(swapi)
	
	pool[next_item] -= 1
	pool[swap[i]] = 1
	swap[i] = next_item

	ret[swap_index] = swap

def create_balanced_orders(items, n_elements_per_subject, use_each_times, consecutive_limit=2,  error=1):
	'''
	Returns a set of unique lists under the constraints of 
	- n_elements_per_subject (must be less than items)
	- use_each_times: number of times each item should be seen across subjects

	Together these define the number of subjects

	'''

	n_subjects = math.ceil((use_each_times * len(items)) / n_elements_per_subject)

	print (f'Creating orders for {n_subjects} subjects')

	pool = get_pool(items, n_elements_per_subject, use_each_times)
	
	ret = []
	while len(pool.keys()) > 0:
		while len(pool.keys()) < n_elements_per_subject:
			rebalance(ret, pool, n_elements_per_subject)
		
		selections = sorted(random.sample(pool.keys(), n_elements_per_subject))
		
		for i in selections:
			pool[i] -= 1
			if pool[i] == 0:
				del pool[i]

		ret.append( selections )
		
		unique, counts = np.unique(ret, return_counts=True)
		
		if all(np.logical_and(counts <= use_each_times + error, counts >= use_each_times)):
			   break
	return ret

def consecutive(data, stepsize=1):
	return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def get_consecutive_list_idxs(orders, consecutive_length):
	
	# Find lists with consecutive items violating our constraint
	idxs = np.where([np.any(np.asarray(list(map(len, consecutive(order)))) >= consecutive_length) for order in orders])[0]
	
	return idxs

def sort_consecutive_constraint(orders, consecutive_length=3):
	
	# Get sets of all orders
	all_order_idxs = np.arange(len(orders))
	
	# Find lists with consecutive items violating our constraint
	consecutive_order_idxs = get_consecutive_list_idxs(orders, consecutive_length)
	
	while len(consecutive_order_idxs):

		for order_idx in consecutive_order_idxs:
			# Select the current list violating the constraint
			current_list = np.asarray(orders[order_idx])

			random_list_options = np.setdiff1d(all_order_idxs, order_idx)

			# Find all sets of consecutive items in the current list --> find their lengths
			consecutive_items = consecutive(current_list)
			consecutive_lengths = np.asarray(list(map(len, consecutive_items)))

			# Find sets of slices that violate the constraint
			violations = np.where(consecutive_lengths >= consecutive_length)[0]

			for violation in violations:
				# Select items that need to be swapped --> these will be swapped into a randomly selected list
				swap_items = consecutive_items[violation][1::2]

				for item in swap_items:
					swap_idx = np.where(current_list == item)[0]

					# Select a random other list
					random_list_idx = random.choice(random_list_options)
					random_list = np.asarray(orders[random_list_idx])

					# Find choices not within our current list
					swap_choices = np.setdiff1d(random_list, current_list)

					# Select a random choice
					choice = random.choice(swap_choices)
					
					# Make sure we didn't violate our constraint again with either list
					while (
						np.isin(choice,current_list) or 
						np.isin(item,random_list)
					):
						
						# Select a random other list
						random_list_idx = random.choice(random_list_options)
						random_list = np.asarray(orders[random_list_idx])

						# Find choices not within our current list
						swap_choices = np.setdiff1d(random_list, current_list)

						# Select a random choice
						choice = random.choice(swap_choices)
					
					# Find the index to swap to
					choice_idx = np.where(random_list == choice)[0]

					# Swap the two items
					current_list[swap_idx] = choice
					random_list[choice_idx] = item

					# Set them in the overall orders
					orders[order_idx] = sorted(current_list)
					orders[random_list_idx] = sorted(random_list)
					
		# Find lists with consecutive items violating our constraint
		consecutive_order_idxs = get_consecutive_list_idxs(orders, consecutive_length)
	return orders