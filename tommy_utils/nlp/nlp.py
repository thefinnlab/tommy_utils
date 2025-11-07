import os
import sys
from pathlib import Path
from numbers import Number

import numpy as np
import pandas as pd
import regex as re
import torch
from torch.nn import functional as F

import gensim.downloader
from gensim.models import KeyedVectors
from gensim.models import fasttext
from gensim import downloader as api
import fasttext.util as ftutil

from scipy.special import rel_entr, kl_div
from scipy import stats
from scipy.spatial.distance import cdist, pdist

##################################
##### MODEL CONFIGURATION ########
##################################

WORD_MODELS = {
	'glove': 'glove.42B.300d.zip',
	'word2vec': 'word2vec-google-news-300',
	'fasttext': 'cc.en.300.bin'
}

CLM_MODELS_DICT = {
	'bloom': 'bigscience/bloom-560m',
	'gpt2': 'gpt2',
	'gpt2-xl': 'gpt2-xl',
	'gpt-neo-x': 'EleutherAI/gpt-neo-1.3B',
	'llama2': 'meta-llama/Llama-2-7b-hf',
	'mistral': 'mistralai/Mistral-7B-v0.1',
	'qwen3-8B': 'Qwen/Qwen3-8B',
	'qwen3-32B': 'Qwen/Qwen3-32B',
	'llama3.1-8B': 'meta-llama/Llama-3.1-8B',
	'llama3.1-70B': 'meta-llama/Llama-3.1-70B',
	'gemma3-1b-pt': 'google/gemma-3-1b-pt'
}

MLM_MODELS_DICT = {
	'bert': 'bert-base-uncased',
	'roberta': 'roberta-base',
	'electra': 'google/electra-base-generator',
	'xlm-prophetnet': 'microsoft/xprophetnet-large-wiki100-cased'
}

MULTIMODAL_MODELS_DICT = {
	'clip': "openai/clip-vit-base-patch32"
}

##################################
##### WORD EMBEDDINGS ############
##################################

def load_word_model(model_name, cache_dir=None):
	"""Load word embedding models (GloVe, Word2Vec, FastText).

	Parameters
	----------
	model_name : str
		Name of the model ('glove', 'word2vec', 'fasttext')
	cache_dir : str, optional
		Cache directory for model files

	Returns
	-------
	model : gensim.models.KeyedVectors
		Loaded word embedding model
	"""
	if cache_dir:
		os.environ['GENSIM_DATA_DIR'] = cache_dir

	if 'glove' in model_name:
		# find the path to our models
		model_name = os.path.splitext(WORD_MODELS[model_name])[0]
		model_dir = os.path.join(cache_dir, model_name)

		model_fn = os.path.join(model_dir, f'gensim-{model_name}.bin')
		vocab_fn = os.path.join(model_dir, f'gensim-vocab-{model_name}.bin')

		print(f'Loading {model_name} from saved .bin file.')
		model = KeyedVectors.load_word2vec_format(model_fn, vocab_fn, binary=True)

	elif 'word2vec' in model_name:
		print(f'Loading {model_name} from saved .bin file.')
		model = api.load(WORD_MODELS[model_name])

	elif 'fasttext' in model_name:
		print(f'Loading {model_name} from saved .bin file.')
		curr_dir = os.getcwd()

		# Set FastText directory
		if cache_dir:
			fasttext_dir = os.path.join(cache_dir, 'fasttext')
		else:
			fasttext_dir = os.path.join(os.environ['HOME'], 'fasttext')

		if not os.path.exists(fasttext_dir):
			os.makedirs(fasttext_dir)

		os.chdir(fasttext_dir)
		ftutil.download_model('en', if_exists='ignore')
		os.chdir(curr_dir)

		model = fasttext.load_facebook_vectors(os.path.join(fasttext_dir, WORD_MODELS[model_name]))

	return model

def get_basis_vector(model, pos_words, neg_words):
	"""Create a semantic basis vector from positive and negative word sets."""
	basis = model[pos_words].mean(0) - model[neg_words].mean(0)
	return basis

def get_word_score(model, basis, word):
	"""Project word onto semantic basis vector."""
	return np.dot(model[word], basis)

##################################
##### WORD CLUSTERING ############
##################################

class autovivify_list(dict):
	"""Serializable defaultdict-like class for lists."""

	def __missing__(self, key):
		value = self[key] = []
		return value

	def __add__(self, x):
		"""Override addition for numeric types when self is empty."""
		if not self and isinstance(x, Number):
			return x
		raise ValueError

	def __sub__(self, x):
		"""Override subtraction for numeric types when self is empty."""
		if not self and isinstance(x, Number):
			return -1 * x
		raise ValueError

def find_word_clusters(labels_array, cluster_labels):
	"""Map cluster labels to their member words."""
	cluster_to_words = autovivify_list()
	for c, i in enumerate(cluster_labels):
		cluster_to_words[i].append(labels_array[c])
	return cluster_to_words

def get_word_clusters(model, cluster, words, norm=True):
	"""Cluster words based on their embeddings."""
	from sklearn.metrics import silhouette_samples

	vectors = np.stack([model.get_vector(word, norm=norm) for word in words])

	cluster.fit(vectors)
	clusters = find_word_clusters(words, cluster.labels_)
	scores = silhouette_samples(vectors, cluster.labels_)

	return clusters, cluster.labels_, scores

##################################
##### TRANSFORMER MODELS #########
##################################

def load_clm_model(model_name, cache_dir=None):
	"""Load causal language models (GPT, LLaMA, etc.).

	Parameters
	----------
	model_name : str
		Model name from CLM_MODELS_DICT or MLM_MODELS_DICT
	cache_dir : str, optional
		Cache directory for model files

	Returns
	-------
	tokenizer : transformers.PreTrainedTokenizer
		Model tokenizer
	model : transformers.PreTrainedModel
		Causal language model
	"""
	if cache_dir:
		os.environ['TRANSFORMERS_CACHE'] = cache_dir

	from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

	if (model_name not in CLM_MODELS_DICT) and (model_name not in MLM_MODELS_DICT):
		print(f'Model not in dictionary - please download and add it to the dictionary')
		sys.exit(0)

	# Load tokenizer
	if model_name in CLM_MODELS_DICT:
		tokenizer = AutoTokenizer.from_pretrained(CLM_MODELS_DICT[model_name])
	elif model_name in MLM_MODELS_DICT:
		tokenizer = AutoTokenizer.from_pretrained(MLM_MODELS_DICT[model_name])
	else:
		sys.exit(0)

	if not tokenizer.pad_token:
		tokenizer.add_special_tokens({'pad_token': '[PAD]'})

	# Load model with appropriate configuration
	if model_name in ['electra', 'xlm-prophetnet']:
		config = AutoConfig.from_pretrained(MLM_MODELS_DICT[model_name])
		config.is_decoder = True
		model = AutoModelForCausalLM.from_pretrained(MLM_MODELS_DICT[model_name], config=config, use_safetensors=True)
	elif model_name == 'roberta':
		model = AutoModelForCausalLM.from_pretrained(MLM_MODELS_DICT[model_name], use_safetensors=True)
	else:
		model = AutoModelForCausalLM.from_pretrained(CLM_MODELS_DICT[model_name], use_safetensors=True)

	model.eval()

	return tokenizer, model

def load_mlm_model(model_name, cache_dir=None):
	"""Load masked language models (BERT, RoBERTa, etc.).

	Parameters
	----------
	model_name : str
		Model name from MLM_MODELS_DICT
	cache_dir : str, optional
		Cache directory for model files

	Returns
	-------
	tokenizer : transformers.PreTrainedTokenizer
		Model tokenizer
	model : transformers.PreTrainedModel
		Masked language model
	"""
	if cache_dir:
		os.environ['TRANSFORMERS_CACHE'] = cache_dir

	from transformers import AutoTokenizer, AutoModel

	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	model = AutoModel.from_pretrained(model_name, use_safetensors=True)

	model.eval()

	return tokenizer, model

def load_multimodal_model(model_name, modality, cache_dir=None):
	"""Load multimodal models (CLIP, etc.).

	Parameters
	----------
	model_name : str
		Model name from MULTIMODAL_MODELS_DICT
	modality : str
		Modality to load ('vision' or 'language')
	cache_dir : str, optional
		Cache directory for model files

	Returns
	-------
	tokenizer : transformers.PreTrainedTokenizer or AutoProcessor
		Tokenizer/processor for the specified modality
	model : transformers.PreTrainedModel
		Multimodal model
	"""
	if cache_dir:
		os.environ['TRANSFORMERS_CACHE'] = cache_dir

	from transformers import AutoTokenizer, AutoProcessor, AutoModel

	if model_name not in MULTIMODAL_MODELS_DICT:
		print(f'Model not in dictionary - please download and add it to the dictionary')
		sys.exit(0)

	model = AutoModel.from_pretrained(MULTIMODAL_MODELS_DICT[model_name], use_safetensors=True)

	if modality == 'vision':
		tokenizer = AutoProcessor.from_pretrained(MULTIMODAL_MODELS_DICT[model_name])
	elif modality == 'language':
		tokenizer = AutoTokenizer.from_pretrained(MULTIMODAL_MODELS_DICT[model_name])

	model.eval()

	return tokenizer, model

##################################
##### MODEL INFERENCE ############
##################################

def get_clm_predictions(inputs, model, tokenizer, out_fn=None):
	"""Get next-word predictions from causal language model.

	Parameters
	----------
	inputs : list of str
		Input text sequences
	model : transformers.PreTrainedModel
		Language model
	tokenizer : transformers.PreTrainedTokenizer
		Tokenizer
	out_fn : str, optional
		Path to save logits

	Returns
	-------
	probs : torch.Tensor
		Next-word probabilities (n_inputs, vocab_size)
	"""
	if any(model_name in model.name_or_path for model_name in MLM_MODELS_DICT.keys()):
		# For MLM models, append mask token
		inputs = [f'{ins} {tokenizer.mask_token}' for ins in inputs]
		tokens = tokenizer(inputs, return_tensors="pt").to(model.device)
		with torch.no_grad():
			logits = model(**tokens).logits[:, -2, :]
	else:
		tokens = tokenizer(inputs, return_tensors="pt").to(model.device)
		with torch.no_grad():
			logits = model(**tokens).logits[:, -1, :]

	probs = F.softmax(logits, dim=-1).detach().cpu()

	if out_fn:
		torch.save(logits.cpu(), out_fn)

	return probs

##################################
##### TEXT PROCESSING ############
##################################

def get_segment_indices(n_words, window_size, bidirectional=False):
	"""Generate context window indices for each word.

	Parameters
	----------
	n_words : int
		Total number of words
	window_size : int
		Size of context window
	bidirectional : bool
		Whether to use bidirectional context

	Returns
	-------
	indices : list of np.ndarray
		Context indices for each word
	"""
	if bidirectional:
		indices = []
		for i in range(0, n_words):
			if i <= window_size // 2:
				# Growing right context at start
				idxs = np.arange(0, (i + window_size // 2) + 1)
			elif i >= (n_words - window_size // 2):
				# Growing left context at end
				idxs = np.arange((i - window_size // 2), n_words)
			else:
				# Full bidirectional context
				idxs = np.arange(i - window_size // 2, (i + window_size // 2) + 1)

			indices.append(idxs)
	else:
		# Unidirectional (left) context
		indices = [
			np.arange(i-window_size, i) if i > window_size else np.arange(0, i)
			for i in range(1, n_words + 1)
		]

	return indices

def transcript_to_input(df_transcript, idxs, add_punctuation=False):
	"""Convert transcript segment to model input string.

	Parameters
	----------
	df_transcript : pd.DataFrame
		Transcript dataframe
	idxs : array-like
		Indices to extract
	add_punctuation : bool
		Whether to include punctuation

	Returns
	-------
	inputs : str
		Concatenated text segment
	"""
	inputs = []

	for i, row in df_transcript.iloc[idxs].iterrows():
		if add_punctuation:
			item = row['word'] + row['punctuation']
		else:
			item = row['word']

		inputs.append(str(item).strip())

	# Join into sentence
	inputs = ' '.join(inputs)
	return inputs

##################################
##### EMBEDDING EXTRACTION #######
##################################

def get_word_prob(tokenizer, word, logits, softmax=True):
	"""Get probability of a word from model logits.

	Parameters
	----------
	tokenizer : transformers.PreTrainedTokenizer
		Tokenizer
	word : str
		Target word
	logits : torch.Tensor
		Model logits
	softmax : bool
		Whether to apply softmax

	Returns
	-------
	prob : float
		Mean probability across subword tokens
	"""
	idxs = tokenizer(word)['input_ids']

	if softmax:
		probs = F.softmax(logits, dim=-1)
	else:
		probs = logits

	word_prob = probs[:, idxs]

	return word_prob.squeeze().mean().item()

def subwords_to_words(sentence, tokenizer):
	"""Map subword tokens back to words.

	Parameters
	----------
	sentence : str
		Input sentence
	tokenizer : transformers.PreTrainedTokenizer
		Tokenizer

	Returns
	-------
	word_token_pairs : list of tuple
		List of (word, tokens, char_indices) tuples
	"""
	word_token_pairs = []

	# Split on spaces and punctuation (excluding apostrophes and hyphens within words)
	regex_split_pattern = r'(\w|\.\w|\:\w|\'\w|\'\w|\-\w|\S)+'

	for m in re.finditer(regex_split_pattern, sentence):
		word = m.group(0)
		tokens = tokenizer.encode(word, add_special_tokens=False)
		char_idxs = (m.start(), m.end()-1)

		word_token_pairs.append((word, tokens, char_idxs))

	return word_token_pairs

def extract_word_embeddings(sentences, tokenizer, model, word_indices=None):
	"""Extract word-level embeddings from transformer models.

	Aggregates subword tokens into word-level representations across all layers.

	Parameters
	----------
	sentences : str or list of str
		Input sentences
	tokenizer : transformers.PreTrainedTokenizer
		Tokenizer
	model : transformers.PreTrainedModel
		Model
	word_indices : int or list of int, optional
		Specific word indices to return

	Returns
	-------
	embeddings : torch.Tensor
		Word embeddings (n_sentences, n_words, n_layers, hidden_size)
	"""
	if isinstance(sentences, str):
		sentences = [sentences]

	if not sentences:
		return []

	# Tokenize sentences
	encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

	# Get model outputs
	with torch.no_grad():
		model_output = model(**encoded_inputs, output_hidden_states=True)

	all_embeddings = []

	for i, sent in enumerate(sentences):
		# Map subwords to words
		subword_word_pairs = subwords_to_words(sent, tokenizer)

		embeddings = []

		for (word, tokens, char_span) in subword_word_pairs:
			# Find token indices for this word
			start_token = encoded_inputs.char_to_token(batch_or_char_index=i, char_index=char_span[0])
			end_token = encoded_inputs.char_to_token(batch_or_char_index=i, char_index=char_span[-1])

			# Sum subword embeddings across all layers
			word_embed = torch.stack([layer[i, start_token:end_token+1, :].sum(0) for layer in model_output['hidden_states']])
			embeddings.append(word_embed)

		embeddings = torch.stack(embeddings)

		# Verify word count matches
		if len(sent.split()) != embeddings.shape[0]:
			print(subword_word_pairs)
			print(len(subword_word_pairs))
			print(embeddings.shape)
			print(len(sent.split()))

		assert len(sent.split()) == embeddings.shape[0]

		all_embeddings.append(embeddings)

	all_embeddings = torch.stack(all_embeddings)

	if word_indices is not None:
		return all_embeddings[:, word_indices, :]
	else:
		return all_embeddings

##################################
##### PREDICTION ANALYSIS ########
##################################

def create_results_dataframe():
	"""Create dataframe for word prediction analysis results."""
	df = pd.DataFrame(columns=[
		'ground_truth_word',
		'ground_truth_prob',
		'top_n_predictions',
		'top_prob',
		'binary_accuracy',
		'glove_avg_accuracy',
		'glove_max_accuracy',
		'glove_prediction_density',
		'word2vec_avg_accuracy',
		'word2vec_max_accuracy',
		'word2vec_prediction_density',
		'fasttext_avg_accuracy',
		'fasttext_max_accuracy',
		'fasttext_prediction_density',
		'entropy',
		'relative_entropy'])

	return df

def get_word_vector_metrics(word_model, predicted_words, ground_truth_word, method='mean'):
	"""Evaluate semantic similarity and prediction density.

	Parameters
	----------
	word_model : gensim.models.KeyedVectors
		Word embedding model
	predicted_words : list of str
		Predicted words
	ground_truth_word : str
		Ground truth word
	method : str
		Aggregation method ('mean', 'max', or None)

	Returns
	-------
	similarity : float
		Semantic similarity to ground truth
	density : float
		Prediction cluster density
	"""
	words_in_model = any([word in word_model for word in predicted_words])

	if (ground_truth_word in word_model) and (words_in_model):
		# Get word vectors
		ground_truth_vector = word_model[ground_truth_word][np.newaxis]
		predicted_vectors = [word_model[word] for word in predicted_words if word in word_model]
		predicted_vectors = np.stack(predicted_vectors)

		# Calculate cosine similarity
		gt_pred_similarity = 1 - cdist(ground_truth_vector, predicted_vectors, metric='cosine')

		if method == 'max':
			gt_pred_similarity = np.nanmax(gt_pred_similarity)
		elif method == 'mean':
			gt_pred_similarity = np.nanmean(gt_pred_similarity)

		# Calculate prediction spread as average pairwise distances
		if predicted_vectors.shape[0] != 1:
			pred_distances = pdist(predicted_vectors, metric='cosine')
			pred_distances = np.nanmean(pred_distances).squeeze()
		else:
			pred_distances = np.nan
	else:
		gt_pred_similarity = np.nan
		pred_distances = np.nan

	return gt_pred_similarity, pred_distances

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
	top_prob = probs.squeeze().max().item()
	
	# convert the tokens into words
	top_words = [tokenizer.decode(item).strip().lower() for item in top_predictions]
	ground_truth_word = ground_truth_word.lower()

	# softmax already performed by here, dont need to do again
	ground_truth_prob = get_word_prob(tokenizer, word=ground_truth_word, logits=probs, softmax=False)

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

			avg_pred_similarity, pred_distances = get_word_vector_metrics(word_model, top_words, ground_truth_word)

			max_pred_similarity, _ = get_word_vector_metrics(word_model, top_words, ground_truth_word, method='max')
			
			word_model_scores[model_name] = {
				'avg_accuracy': avg_pred_similarity,
				'max_accuracy': max_pred_similarity,
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
		'ground_truth_prob': ground_truth_prob,
		'top_n_predictions': top_words,
		'top_prob': top_prob,
		'binary_accuracy': binary_accuracy,
		'glove_avg_accuracy': word_model_scores['glove']['avg_accuracy'],
		'glove_max_accuracy': word_model_scores['glove']['max_accuracy'],
		'glove_prediction_density': word_model_scores['glove']['cluster_density'],
		'word2vec_avg_accuracy': word_model_scores['word2vec']['avg_accuracy'],
		'word2vec_max_accuracy': word_model_scores['word2vec']['max_accuracy'],
		'word2vec_prediction_density': word_model_scores['word2vec']['cluster_density'],
		'fasttext_avg_accuracy': word_model_scores['fasttext']['avg_accuracy'],
		'fasttext_max_accuracy': word_model_scores['fasttext']['max_accuracy'],
		'fasttext_prediction_density': word_model_scores['fasttext']['cluster_density'],
		'entropy': entropy,
		'relative_entropy': kl_divergence,
	}
	
	return df
