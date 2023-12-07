import os, sys
import gensim.downloader
from gensim.models import KeyedVectors
from subprocess import run
import numpy as np
import pandas as pd
import regex as re
from pathlib import Path

WORD_MODELS = {
	'glove': 'glove.42B.300d.zip',
	'word2vec': 'word2vec-google-news-300'
	# 'glove.twitter.27B' : ['https://nlp.stanford.edu/data/glove.twitter.27B.zip',
	# 					   'glove.twitter.27B.zip'],
	# 'glove.42B.300d' : ['https://nlp.stanford.edu/data/glove.42B.300d.zip',
	# 					'glove.42B.300d.zip'],
	# # download directly from google drive doesn't work --> download file manually from link
	# 'GoogleNews-vectors-negative300': ["https://drive.google.com/u/0/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download",
	# 								   'GoogleNews-vectors-negative300.bin']
}

# def download_gensim_model(model_name, cache_dir):


	
# 	assert (model_name in WORD_MODELS)
	
# 	# grab the url for downloading and the model filename
# 	url, fn = WORD_MODELS[model_name]
	
# 	# find the path to our models
# 	model_dir = os.path.join(cache_dir, model_name)
# 	model_fn = os.path.join(model_dir, fn)
# 	path = Path(fn)
	
# 	if not os.path.exists(model_dir):
# 		os.makedirs(model_dir)
	
# 	# if we don't currently have the model, we have to unzip
# 	# arg 1 is where to place the downloaded file, arg2 is url
# 	if not os.path.exists(model_fn) and 'drive' not in url: 
# 		run(f'wget -P {model_dir} -c {url}', shell=True)
	
# 		if path.suffixes[-1] == '.gz':
# 			run(f'gzip -d {model_fn}', shell=True)
# 		elif path.suffixes[-1] == '.zip':
# 			run(f'unzip {model_fn}', shell=True)
	
# 	if 'glove' in model_name:
# 		model_fn = os.path.join(os.path.dirname(model_fn), os.path.basename(model_fn).replace('.zip', '.txt'))

# 	return model_fn

def load_gensim_model(model_name, cache_dir=None):
	'''
	Given the path to a glove model file,  load the model
	into the gensim word2vec format for ease
	'''

	if cache_dir:
		os.environ['GENSIM_DATA_DIR'] = cache_dir

	if 'glove' in model_name:
		# find the path to our models
		model_name = os.path.splitext(WORD_MODELS[model_name])[0]
		model_dir = os.path.join(cache_dir, model_name)

		model_fn = os.path.join(model_dir, f'gensim-{model_name}.bin')
		vocab_fn = os.path.join(model_dir, f'gensim-vocab-{model_name}.bin')

		# if the files don't already exist, load the files and save out for next time
		print (f'Loading {model_name} from saved .bin file.')

		model = KeyedVectors.load_word2vec_format(model_fn, vocab_fn, binary=True)

	elif 'word2vec' in model_name:
		print (f'Loading {model_name} from saved .bin file.')
		model = gensim.downloader.load(WORD_MODELS[model_name])
		
	return model

def get_basis_vector(model, pos_words, neg_words):
	basis = model[pos_words].mean(0) - model[neg_words].mean(0)
	basis = basis / 1
	return basis

def get_word_score(model, basis, word):
	word_score = np.dot(model[word], basis)
	return word_score

'''Serializable/Pickleable class to replicate the functionality of collections.defaultdict'''
class autovivify_list(dict):
		def __missing__(self, key):
				value = self[key] = []
				return value

		def __add__(self, x):
				'''Override addition for numeric types when self is empty'''
				if not self and isinstance(x, Number):
						return x
				raise ValueError

		def __sub__(self, x):
				'''Also provide subtraction method'''
				if not self and isinstance(x, Number):
						return -1 * x
				raise ValueError


def find_word_clusters(labels_array, cluster_labels):
	cluster_to_words = autovivify_list()
	for c, i in enumerate(cluster_labels):
		cluster_to_words[i].append(labels_array[c])
	return cluster_to_words

def get_word_clusters(model, cluster, words, norm=True):
	from sklearn.metrics import silhouette_samples
	vectors = np.stack([model.get_vector(word, norm=norm) for word in words])
	
	# Fit model to samples
	cluster.fit(vectors)
	clusters = find_word_clusters(words, cluster.labels_)
	
	scores = silhouette_samples(vectors, cluster.labels_)
	
	return clusters, cluster.labels_, scores

#### STUFF FOR TRANSFORMERS ######
import torch
from torch.nn import functional as F
from scipy.special import rel_entr, kl_div
from scipy import stats
from scipy.spatial.distance import cdist

CLM_MODELS_DICT = {
	'bloom': 'bigscience/bloom-560m',
	'gpt2': 'gpt2',
	'gpt2-xl': 'gpt2-xl',
	'gpt-neo-x': 'EleutherAI/gpt-neo-1.3B',
	'roberta': 'roberta-base',
	'electra': "google/electra-base-generator",
	'llama2': 'meta-llama/Llama-2-7b-hf',
	'mistral': 'mistralai/Mistral-7B-v0.1',
	'xlm-prophetnet': "microsoft/xprophetnet-large-wiki100-cased"
}

MLM_MODELS = ['roberta', 'electra', 'xlm-prophetnet']

def load_clm_model(model_name, cache_dir=None):
	'''
	Use a model from the sentence-transformers library to get
	sentence embeddings. Models used are trained on a next-sentence
	prediction task and evaluate the likelihood of S2 following S1.
	'''
	# set the path of where to download models
	# this NEEDS to be run before loading from transformers
	if cache_dir:
		os.environ['TRANSFORMERS_CACHE'] = cache_dir

	from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelWithLMHead
		
	if model_name not in CLM_MODELS_DICT:
		print (f'Model not in dictionary - please download and add it to the dictionary')
		sys.exit(0)
	
	# load a tokenizer and a model
	tokenizer = AutoTokenizer.from_pretrained(CLM_MODELS_DICT[model_name])
	
	if model_name in ['electra', 'xlm-prophetnet']:
		config = AutoConfig.from_pretrained(CLM_MODELS_DICT[model_name])
		config.is_decoder = True
		model = AutoModelForCausalLM.from_pretrained(CLM_MODELS_DICT[model_name], config=config)
	else:
		model = AutoModelForCausalLM.from_pretrained(CLM_MODELS_DICT[model_name])
	
	return tokenizer, model

def get_clm_predictions(inputs, model, tokenizer):

	if any(model_name in model.name_or_path for model_name in MLM_MODELS):
		# append a mask token to the inputs
		inputs = [f'{ins} {tokenizer.mask_token}' for ins in inputs]
		tokens = tokenizer(inputs, return_tensors="pt")
		
		with torch.no_grad():
			logits = model(**tokens).logits[:, -2, :]
	else:
		tokens = tokenizer(inputs, return_tensors="pt")
		
		with torch.no_grad():
			logits = model(**tokens).logits[:, -1, :]
	
	# get the probability of the logits
	probs = F.softmax(logits, dim=-1)
	
	return probs

def get_segment_indices(n_words, window_size, bidirectional=False):
	'''
	Given n_words (a total number of words in a transcript) and the 
	size of a context window, return the indices for extracting segments
	of text.
	'''

	if bidirectional:
		indices = []
		for i in range(0, n_words):
			# add right side context of half the window size while increasing left side context
			if i <= window_size // 2:
				idxs = np.arange(0, (i + window_size // 2) + 1)
			# add left side context while reducing right side context size
			elif i >= (n_words - window_size // 2):
				idxs = np.arange((i - window_size // 2), n_words + 1)
			else:
				idxs = np.arange(i - window_size // 2, (i + window_size // 2) + 1)

			indices.append(idxs)
	else:
		indices = [
			np.arange(i-window_size, i) if i > window_size else np.arange(0, i)
			for i in range(1, n_words)
		]
		
	return indices

def create_results_dataframe():
	
	df = pd.DataFrame(columns = ['ground_truth_word',
							 'top_n_predictions', 
							 'binary_accuracy', 
							 'glove_continuous_accuracy', 
							 'glove_prediction_density',
							 'word2vec_continuous_accuracy', 
							 'word2vec_prediction_density',
							 'entropy', 
							 'relative_entropy'])
	
	return df

def get_model_statistics(ground_truth_word, probs, tokenizer, prev_probs=None, word_models=None, top_n=1):
	'''
	Given a probability distribution, calculate the following statistics:
		- binary accuracy (was GT word in the top_n predictions)
		- continuous accuracy (similarity of GT to top_n predictions)
		- entropy (certainty of the model's prediction)
		- kl divergence 
	'''
	
	df = create_results_dataframe()
	
	# sort the probability distribution --> apply flip so that top items are returned in order
	top_predictions = np.argsort(probs.squeeze()).flip(0)[:top_n]
	
	# convert the tokens into words
	top_words = [tokenizer.decode(item).strip().lower() for item in top_predictions]
	ground_truth_word = ground_truth_word.lower()
	
	############################
	### MEASURES OF ACCURACY ###
	############################
	
	# is the ground truth in the list of top words?
	binary_accuracy = ground_truth_word in top_words
	
	# go through each model and compute continuous accuracy
	# make sure a word model is defined
	word_model_scores = {}

	if word_models:
		for model_name, word_model in word_models.items():

			# how similar was the ground truth to the list of top words
			# make sure we have a word model to use and that the word of interest is a key
			words_in_model =  any([word in word_model for word in top_words])

			if (ground_truth_word in word_model) and (words_in_model):
				# get word vectors from model
				ground_truth_vector = word_model[ground_truth_word][np.newaxis]
				predicted_vectors = [word_model[word] for word in top_words if word in word_model]
				predicted_vectors = np.stack(predicted_vectors)

				# calculate cosine similarity
				gt_pred_similarity = 1 - cdist(ground_truth_vector, predicted_vectors, metric='cosine')
				gt_pred_similarity = np.nanmean(gt_pred_similarity)

				# calculate spread of predictions as average pairwise distances
				pred_distances = cdist(predicted_vectors, predicted_vectors, metric='cosine')
				pred_distances = np.nanmean(pred_distances).squeeze()
			else:
				gt_pred_similarity = np.nan
				pred_distances = np.nan

			word_model_scores[model_name] = {
				'continuous_accuracy': gt_pred_similarity,
				'cluster_density': pred_distances
			}
	
	###############################
	### MEASURES OF UNCERTAINTY ###
	###############################
	
	# get entropy of the distribution
	entropy = stats.entropy(probs, axis=-1)[0]
	
	# if there was a previous distribution that we can use, get the KL divergence
	# between current distribution and previous distribution
	if prev_probs is not None:
		kl_divergence = kl_div(probs, prev_probs)
		kl_divergence[torch.isinf(kl_divergence)] = 0
		kl_divergence = kl_divergence.sum().item()
	else:
		kl_divergence = np.nan
		
	df.loc[len(df)] = {
		'ground_truth_word': ground_truth_word,
		'top_n_predictions': top_words,
		'binary_accuracy': binary_accuracy,
		'glove_continuous_accuracy': word_model_scores['glove']['continuous_accuracy'],
		'glove_prediction_density': word_model_scores['glove']['cluster_density'],
		'word2vec_continuous_accuracy': word_model_scores['word2vec']['continuous_accuracy'],
		'word2vec_prediction_density': word_model_scores['word2vec']['cluster_density'],
		'entropy': entropy,
		'relative_entropy': kl_divergence,
	}
	
	return df

def load_mlm_model(model_name, cache_dir=None):
	'''
	Use a model from the sentence-transformers library to get
	sentence embeddings. Models used are trained on a next-sentence
	prediction task and evaluate the likelihood of S2 following S1.
	'''
	# set the path of where to download models
	# this NEEDS to be run before loading from transformers
	if cache_dir:
		os.environ['TRANSFORMERS_CACHE'] = cache_dir

	from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelWithLMHead

	# Load model from HuggingFace Hub
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	model = AutoModel.from_pretrained(model_name)
	
	return tokenizer, model

def subwords_to_words(sentence, tokenizer):
	
	word_token_pairs = []
	
	# split the sentence on spaces + punctuation (excluding apostrophes and hyphens within words)
	for m in re.finditer(r"[\w]+['-]?[\w]*", sentence):
		word = m.group(0)
		tokens = tokenizer.encode(word, add_special_tokens=False)
		char_idxs = (m.start(), m.end()-1)
		
		word_token_pairs.append((word, tokens, char_idxs))
	
	return word_token_pairs

def extract_word_embeddings(sentences, tokenizer, model, word_indices=None):
	'''
	Given a list of sentences, pass them through the tokenizer/model. Then pair
	sub-word tokens into the words of the actual sentence and extract the true
	word embeddings. 
	
	If wanted, can return only certain indices (specified by word_indices)
	
	Currently not robust to different length strings MBMB
	'''
	
	if isinstance(sentences, str):
		sentences = [sentences]
	
	if not sentences:
		return []
	
	# get the full sentence tokenized
	encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
	
	# get the embeddings
	with torch.no_grad():
		model_output = model(**encoded_inputs)
	
	all_embeddings = []
	
	# bring together the current sentence, its tokens, and its embeddings
	for i, sent in enumerate(sentences):
		# now pair subwords into words for the current sentence
		subword_word_pairs = subwords_to_words(sent, tokenizer)
		
		embeddings = []
		
		# for the current set of word subword pairs, get the embeddings
		for (word, tokens, char_span) in subword_word_pairs:
			
			# given the character to token mapping in the sentence, 
			# find the first and last token indices
			start_token = encoded_inputs.char_to_token(batch_or_char_index=i, char_index=char_span[0])
			end_token = encoded_inputs.char_to_token(batch_or_char_index=i, char_index=char_span[-1])
			
			# extract the embedding for the given word
			word_embed = model_output['last_hidden_state'][i, start_token:end_token+1, :].sum(0)
			embeddings.append(word_embed)
			
		# stack the embeddings together
		embeddings = torch.stack(embeddings)
		
		# make sure the mapping happened correctly
		assert (len(sent.split()) == embeddings.shape[0])
		
		all_embeddings.append(embeddings)
	
	all_embeddings = torch.stack(all_embeddings)
	
	if word_indices:
		return all_embeddings[:, word_indices, :]
	else:
		return all_embeddings
