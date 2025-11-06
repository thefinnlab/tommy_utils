import os
import json
import itertools
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torchvision
import torchaudio
import torchlens as tl
from torchvision import transforms

from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline

import himalaya
from himalaya.backend import get_backend

from himalaya.ridge import (
	ColumnTransformerNoStack,
	BandedRidgeCV
)

from himalaya.kernel_ridge import (
	KernelRidgeCV,
	MultipleKernelRidgeCV,
	ColumnKernelizer,
	Kernelizer,
)

from .delayer import Delayer
from .custom_solvers import (
	GroupLevelBandedRidge,
	GroupLevelMultipleKernelRidgeCV,
	solve_group_level_group_ridge_random_search,
	solve_group_level_multiple_kernel_ridge_random_search
)

from . import nlp

# Register custom solvers with Himalaya
BandedRidgeCV.ALL_SOLVERS['group_level_random_search'] = solve_group_level_group_ridge_random_search
MultipleKernelRidgeCV.ALL_SOLVERS['group_level_random_search'] = solve_group_level_multiple_kernel_ridge_random_search

ENCODING_FEATURES = {
	'visual': [
		'alexnet',
		'clip',
	],
	'audio': [
		'spectral',
	],
	'language': [
		'phoneme',
		'word2vec',
		'gpt2',
		'gpt2-xl',
	]
}

# get path of the encoding_utils file --> find the relative path of the phonemes file
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PHONEMES_FN = os.path.join(FILE_DIR, 'data/nlp/cmudict-0.7b.phones.txt')
CMU_PHONEMES = pd.read_csv(PHONEMES_FN, header=None, names=['phoneme', 'type'], sep="\t")

##################################
##### FEATURE CONFIGURATION ######
##################################

def get_modality_features(modality):
	"""Get available feature extractors for a given modality.

	Parameters
	----------
	modality : str
		One of 'audiovisual', 'audio', 'text', or 'visual'

	Returns
	-------
	list
		List of available feature extractor names
	"""
	modality_map = {
		'audiovisual': ['visual', 'audio', 'language'],
		'audio': ['audio', 'language'],
		'text': ['language'],
		'visual': ['visual']
	}

	items = modality_map.get(modality, [])
	modality_features = []

	for item in items:
		if ENCODING_FEATURES.get(item):
			modality_features.extend(ENCODING_FEATURES[item])

	return modality_features

##################################
##### TRANSCRIPT PROCESSING ######
##################################

def load_gentle_transcript(transcript_fn, start_offset=None):
	"""Load and process a Gentle alignment transcript.

	Parameters
	----------
	transcript_fn : str
		Path to Gentle JSON transcript file
	start_offset : float, optional
		Time offset to apply to all timestamps

	Returns
	-------
	pd.DataFrame
		Transcript with columns: word, start, end, punctuation
	"""
	with open(transcript_fn) as f:
		data = json.load(f)

	transcript = data['transcript']
	df_transcript = pd.json_normalize(data['words'])

	for i, row in df_transcript.iterrows():
		# get the punctuation of the current row
		if i+1 < len(df_transcript):
			start_punc, end_punc = row['endOffset'], df_transcript.loc[i+1, 'startOffset']
			word_punctuation = transcript[start_punc:end_punc]
		else:
			word_punctuation = transcript[row['endOffset']:]

		df_transcript.loc[i, 'punctuation'] = word_punctuation

	# Interpolate missing times
	df_transcript['start'] = df_transcript['start'].interpolate()
	df_transcript['word'] = df_transcript.word.str.lower()

	# Apply time offset if provided
	if start_offset:
		df_transcript['start'] -= start_offset
		df_transcript['end'] -= start_offset

	return df_transcript

##################################
##### LANGUAGE FEATURES ##########
##################################

def create_phoneme_features(df_transcript):
	"""Convert Gentle transcript to one-hot phoneme features.

	Parameters
	----------
	df_transcript : pd.DataFrame
		Gentle transcript with phoneme alignments

	Returns
	-------
	times : np.ndarray
		Onset times for each phoneme
	phoneme_features : np.ndarray
		One-hot encoded phonemes (39 dimensions)
	"""
	phoneme_time_features = []

	for i, row in df_transcript.iterrows():
		if row['case'] != 'success' or row['alignedWord'] == '<unk>':
			continue

		phoneme_start = row['start']
		word_phonemes = []
		
		for item in row['phones']:
			phoneme = item['phone'].split('_')[0].upper()
			one_hot_phoneme = np.asarray(CMU_PHONEMES['phoneme'] == phoneme).astype(int)

			if sum(one_hot_phoneme) != 1:
				print(f'Word {i} - skipping phoneme: {phoneme}')
				phoneme_start += item['duration']
				continue

			phoneme_info = (phoneme_start, one_hot_phoneme)
			word_phonemes.append(phoneme_info)
			phoneme_start += item['duration']

		phoneme_time_features.append(word_phonemes)

	phoneme_time_features = sum(phoneme_time_features, [])
	times, phoneme_features = [np.stack(item) for item in zip(*phoneme_time_features)]

	print(f'Phoneme feature space is size: {phoneme_features.shape}')

	return times, phoneme_features

def create_word_features(df_transcript, word_model):
	"""Extract word embeddings from Gensim model.

	Parameters
	----------
	df_transcript : pd.DataFrame
		Gentle transcript
	word_model : gensim.models
		Word embedding model (Word2Vec, FastText, etc.)

	Returns
	-------
	times : np.ndarray
		Word onset times
	word_features : np.ndarray
		Word embedding vectors
	"""
	word_time_features = []

	for i, row in df_transcript.iterrows():
		word = row['word']

		# Check if word exists in model vocabulary
		if 'fasttext' in str(type(word_model)) or word in word_model.key_to_index:
			word_vector = word_model[word]
		else:
			continue

		word_time_info = (row['start'], word_vector)
		word_time_features.append(word_time_info)

	times, word_features = [np.stack(item) for item in zip(*word_time_features)]

	print(f'Word feature space is size: {word_features.shape}')

	return times, word_features

def create_transformer_features(df_transcript, tokenizer, model, window_size=25, bidirectional=False, add_punctuation=False):
	"""Extract contextualized word embeddings from transformer models.

	Parameters
	----------
	df_transcript : pd.DataFrame
		Gentle transcript
	tokenizer : transformers.PreTrainedTokenizer
		Tokenizer for the model
	model : transformers.PreTrainedModel
		Transformer model
	window_size : int
		Context window size for contextualization
	bidirectional : bool
		Whether to use bidirectional context
	add_punctuation : bool
		Whether to include punctuation in context

	Returns
	-------
	times : np.ndarray
		Word onset times
	word_features : np.ndarray
		Contextualized embeddings (n_layers, n_words, hidden_size)
	"""
	word_time_features = []

	# Create segments for windowed contextualization
	segments = nlp.get_segment_indices(n_words=len(df_transcript), window_size=window_size, bidirectional=bidirectional)

	for (i, row), segment in zip(df_transcript.iterrows(), segments):
		print(f'Processing segment {i+1}/{len(df_transcript)}')

		inputs = nlp.transcript_to_input(df_transcript, segment, add_punctuation=add_punctuation)

		# Extract embeddings for the last word (target word)
		word_embeddings = nlp.extract_word_embeddings([inputs], tokenizer, model, word_indices=-1)
		word_embeddings = word_embeddings.squeeze()

		word_time_info = (row['start'], word_embeddings.squeeze())
		word_time_features.append(word_time_info)

	times, word_features = [np.stack(item) for item in zip(*word_time_features)]

	# Move layers to first dimension: (n_layers, n_words, hidden_size)
	word_features = np.moveaxis(word_features, 0, 1)

	print(f'Transformer feature space is size: {word_features.shape}')

	return times, word_features

##################################
##### VISION FEATURES ############
##################################

VISION_MODELS_DICT = {
	'alexnet': [
		'maxpool2d_1_3',
		'maxpool2d_2_6',
		'relu_3_8',
		'relu_4_10',
		'maxpool2d_3_13',
		'relu_6_18',
		'relu_7_21'],
	'resnet50': [
		'layer1.2.relu:3', 
		'layer2.3.relu:3', 
		'layer3.5.relu:3', 
		'layer4.2.relu:3'
	]
}

def load_torchvision_model(model_name):
	return getattr(torchvision.models, model_name)(pretrained=True)

def chunk(it, size):
	it = iter(it)
	return iter(lambda: tuple(itertools.islice(it, size)), ())

def get_layer_tensors(model_output, flatten=True):
	
	layer_tensors = []
	
	for layer in model_output.layers_with_saved_activations:
		arr = model_output[layer].tensor_contents.detach()
		
		if flatten:
			arr = torch.flatten(arr, 1, -1)
		
		layer_tensors.append(arr)
	return layer_tensors

def chunk_video_clips(decoder, batch_size, trim=None):
	"""Batch video frames from a decoder.

	Parameters
	----------
	decoder : video decoder
		Video decoder object with frame indexing
	batch_size : int
		Number of frames per batch
	trim : tuple, optional
		(start, end) frame indices to trim to

	Yields
	------
	torch.Tensor
		Batched frames
	"""
	if trim:
		idxs = (i for i in range(trim[0], trim[1]))
	else:
		idxs = (i for i in range(len(decoder)))

	while True:
		sl = list(itertools.islice(idxs, batch_size))
		if not sl:
			break

		frames = [decoder[i] for i in sl]
		yield torch.stack(frames)

def create_vision_features(images, model_name, batch_size=8):
	"""Extract vision features from video frames.

	Parameters
	----------
	images : video decoder
		Video frame decoder
	model_name : str
		Name of vision model ('alexnet', 'resnet50', 'clip')
	batch_size : int
		Batch size for processing

	Returns
	-------
	times : np.ndarray
		Frame times in seconds
	vision_features : np.ndarray or list of np.ndarray
		Vision features (single array for CLIP, list of arrays for layer-wise models)
	"""
	if model_name == 'clip':
		tokenizer, model = nlp.load_multimodal_model(model_name=model_name, modality='vision')
	else:
		model = load_torchvision_model(model_name)
		model_layers = VISION_MODELS_DICT[model_name]

	video_fps = int(images.metadata.average_fps)
	n_frames = len(images)
	times = np.arange(0, n_frames / video_fps, 1/video_fps)

	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	vision_features = []

	for i, batch in enumerate(chunk_video_clips(images, batch_size=batch_size)):
		print(f'Processing batch {i+1}/{len(images)//batch_size}')

		if model_name == 'clip':
			inputs = tokenizer(images=batch, return_tensors='pt')
			vision_embeddings = model.get_image_features(**inputs).detach()
			# Normalize CLIP features
			vision_embeddings = vision_embeddings / vision_embeddings.norm(p=2, dim=-1, keepdim=True)
		else:
			batch = torch.stack([transform(x) for x in batch])
			# Use torchlens to extract intermediate layer activations
			model_output = tl.log_forward_pass(model, batch, layers_to_save=model_layers)
			vision_embeddings = get_layer_tensors(model_output)

		vision_features.append(vision_embeddings)

	# Stack features across batches
	if model_name == 'clip':
		vision_features = np.vstack(vision_features)
		print(f'Vision feature space has shape {vision_features.shape}')
	else:
		vision_features = [np.vstack(item) for item in zip(*vision_features)]
		print(f'Vision feature space has {len(vision_features)} layers')

	return times, vision_features

##################################
##### AUDIO FEATURES #############
##################################

def load_torchaudio_model(model_name):
	bundle = getattr(torchaudio.pipelines, model_name.upper())
	model = bundle.get_model()
	return bundle, model

def create_spectral_features(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
	"""Create mel-spectrogram features from audio.

	Parameters
	----------
	audio : torch.Tensor
		Audio waveform
	sr : int
		Sample rate
	n_fft : int
		FFT window size
	hop_length : int
		Hop length between frames
	n_mels : int
		Number of mel frequency bins

	Returns
	-------
	times : np.ndarray
		Time points for each spectral frame
	melspec : np.ndarray
		Mel-spectrogram features (n_frames, n_mels)
	"""
	mel_spectrogram = torchaudio.transforms.MelSpectrogram(
		sample_rate=sr,
		center=True,
		n_fft=n_fft,
		hop_length=hop_length,
		pad_mode="reflect",
		power=2.0,
		norm="slaney",
		n_mels=n_mels,
		mel_scale="htk",
	)

	melspec = mel_spectrogram(audio).squeeze()

	# Create evenly-spaced time points for spectrogram frames
	times = np.linspace(0, audio.shape[-1]/sr, melspec.shape[-1])

	# Transpose to (n_frames, n_mels)
	return times, melspec.T

##################################
##### CROSS-VALIDATION ###########
##################################

def generate_leave_one_run_out(n_samples, run_onsets, random_state=None, n_runs_out=1):
	"""Generate a leave-one-run-out split for cross-validation.
	
	Generates as many splits as there are runs.
	
	Parameters
	----------
	n_samples : int
		Total number of samples in the training set.
	run_onsets : array of int of shape (n_runs, )
		Indices of the run onsets.
	random_state : None | int | instance of RandomState
		Random state for the shuffling operation.
	n_runs_out : int
		Number of runs to leave out in the validation set. Default to one.
	
	Yields
	------
	train : array of int of shape (n_samples_train, )
		Training set indices.
	val : array of int of shape (n_samples_val, )
		Validation set indices.
	"""
	random_state = check_random_state(random_state)
	
	n_runs = len(run_onsets)

	if n_runs_out >= len(run_onsets):
		raise ValueError("More runs requested for validation than there are "
						 "total runs. Make sure that n_runs_out is less than "
						 "than the number of runs (e.g., len(run_onsets)).")

	# Generate all combinations of runs for validation
	all_val_runs = np.array(list(itertools.combinations(range(n_runs), n_runs_out)))
	all_val_runs = random_state.permutation(all_val_runs)

	print(f'Total number of validation runs: {len(all_val_runs)}')
	
	all_samples = np.arange(n_samples)
	runs = np.split(all_samples, run_onsets[1:])
	
	if any(len(run) == 0 for run in runs):
		raise ValueError("Some runs have no samples. Check that run_onsets "
						 "does not include any repeated index, nor the last "
						 "index.")
	
	for val_runs in all_val_runs:
		train = [runs[jj] for jj in range(n_runs) if jj not in val_runs]
		val = [runs[jj] for jj in range(n_runs) if jj in val_runs]

		assert len(val) == n_runs_out  # Verify correct number of validation runs
		train, val = [np.hstack(x) for x in [train, val]]
		assert not np.isin(train, val).any()  # Ensure no overlap

		yield train, val

##################################
##### MODEL BUILDING #############
##################################

def create_banded_features(features, feature_names):
	"""Prepare features for banded ridge regression.

	Parameters
	----------
	features : list of np.ndarray
		List of feature arrays for different feature spaces
	feature_names : list of str
		Names for each feature space

	Returns
	-------
	features : np.ndarray
		Concatenated features across all feature spaces
	feature_space_info : list of tuple
		List of (name, slice) pairs for each feature space
	"""
	features_dim = [feature.shape[1] for feature in features]

	# Create slices for each feature space
	feature_space_idxs = np.concatenate([[0], np.cumsum(features_dim)])
	feature_space_slices = [slice(*item) for item in zip(feature_space_idxs[:-1], feature_space_idxs[1:])]

	assert len(feature_space_slices) == len(feature_names)

	# Concatenate feature spaces horizontally
	features = np.concatenate(features, axis=1)

	# Pair names with slices
	feature_space_info = [(name, slice) for name, slice in zip(feature_names, feature_space_slices)]

	return features, feature_space_info

def get_concatenated_data(data, indices, precision='float32'):
	
	if len(indices) > 1:
		data_split = np.concatenate(itemgetter(*indices)(data), axis=0).astype(precision)
	else:
		data_split = np.stack(itemgetter(*indices)(data), axis=0).astype(precision)

	# Convert nan to num
	data_split = np.nan_to_num(data_split)

	# Convert inf to num
	data_split[np.isinf(data_split)] = 0

	return data_split

def get_train_test_splits(x, y, train_indices, test_indices, precision='float32', group_level=False):
	
	# Get train data
	if group_level:
		assert (len(x) == 1)
		X_train = get_concatenated_data(x, [0], precision)
		X_test = get_concatenated_data(x, [0], precision)
	else:
		X_train = get_concatenated_data(x, train_indices, precision)
		X_test = get_concatenated_data(x, test_indices, precision)

	# Get test data
	Y_train = get_concatenated_data(y, train_indices, precision)
	Y_test = get_concatenated_data(y, test_indices, precision)
	
	return X_train, Y_train, X_test, Y_test

def create_banded_model(model, delays, feature_space_infos, kernel=None, n_jobs=None, force_cpu=False):

	'''
		delays: list of ints. Number of delays to use when 
			making a delayer
		feature_space_infos: list of tuples. Names of each 
			feature space and 
	'''

	scaler = StandardScaler(with_mean=True, with_std=False) # demean, but keep std as contains information
	delayer = Delayer(delays=delays) # delays are in indices --> needs to be scales to TRs

	if kernel:   
		preprocess_pipeline = make_pipeline(scaler, delayer, Kernelizer(kernel=kernel))
	else:
		preprocess_pipeline = make_pipeline(scaler, delayer)
	
	# preprocessing for each feature space
	feature_tuples = [
		(name, preprocess_pipeline, slice_)
		for name, slice_ in feature_space_infos
	]

	if kernel:
		# put them together
		column_transformer = ColumnKernelizer(feature_tuples, n_jobs=n_jobs, force_cpu=force_cpu)
	else:
		# put them together
		column_transformer = ColumnTransformerNoStack(feature_tuples, n_jobs=n_jobs)

	pipeline = make_pipeline(column_transformer, model)

	return pipeline

def build_encoding_pipeline(X, Y, inner_cv, feature_space_infos=None, delays=[1,2,3,4],
	n_iter=20, n_targets_batch=200, n_alphas_batch=5, n_targets_batch_refit=200,
	Y_in_cpu=False, force_cpu=False, solver="random_search", alphas=np.logspace(1, 20, 20),
	n_jobs=None, force_banded_ridge=False):
	"""Build an encoding model pipeline with ridge regression.

	Parameters
	----------
	X : list of np.ndarray
		Feature arrays
	Y : list of np.ndarray
		Target arrays
	inner_cv : int or cross-validation generator
		Inner cross-validation strategy
	feature_space_infos : list of tuple, optional
		Feature space names and slices for banded ridge
	delays : list of int
		HRF delays to model
	n_iter : int
		Number of random search iterations
	n_targets_batch : int
		Batch size for targets during CV
	n_alphas_batch : int
		Batch size for alphas
	n_targets_batch_refit : int
		Batch size for targets during refit
	Y_in_cpu : bool
		Keep Y in CPU memory
	force_cpu : bool
		Force CPU computation
	solver : str
		Solver name ('random_search', 'group_level_random_search', etc.)
	alphas : np.ndarray
		Alpha values to search
	n_jobs : int, optional
		Number of parallel jobs
	force_banded_ridge : bool
		Force banded ridge even when n_samples < n_features

	Returns
	-------
	pipeline : sklearn.pipeline.Pipeline
		Complete encoding pipeline
	"""
	# Solver parameters
	N_TARGETS_BATCH = n_targets_batch
	N_ALPHAS_BATCH = n_alphas_batch
	N_TARGETS_BATCH_REFIT = n_targets_batch_refit
	N_ITER = n_iter
	ALPHAS = alphas
	RANDOM_STATE = 42

	# Validate input shapes
	if solver == 'group_level_random_search':
		assert all([X[0].shape[0] == y.shape[0] for y in Y])
	else:
		assert len(X) == len(Y)

	n_samples = np.concatenate(X).shape[0]
	n_features = np.concatenate(X).shape[1]

	# Standard preprocessing: demean and add HRF delays
	scaler = StandardScaler(with_mean=True, with_std=False)
	delayer = Delayer(delays=delays)

	# Multiple feature spaces: use banded or multiple kernel ridge
	if feature_space_infos:

		if (n_samples > n_features or force_banded_ridge):
			print(f'Using banded ridge')

			solver_function = BandedRidgeCV.ALL_SOLVERS[solver]

			if solver == 'group_level_random_search':
				if not all([y.shape == Y[0].shape for y in Y]):
					raise ValueError("To use group level random search, all "
						"groups need to have same number of samples.")

				n_samples_group = Y[0].shape[0]

				solver_params = dict(n_samples_group=n_samples_group, n_iter=N_ITER, alphas=ALPHAS, 
					n_targets_batch=N_TARGETS_BATCH, n_alphas_batch=N_ALPHAS_BATCH, 
					n_targets_batch_refit=N_TARGETS_BATCH_REFIT)

				banded_model = GroupLevelBandedRidge(groups="input", solver=solver, 
					solver_params=solver_params, cv=inner_cv, Y_in_cpu=Y_in_cpu, force_cpu=force_cpu)

			elif solver == 'random_search':
				solver_params = dict(n_iter=N_ITER, alphas=ALPHAS, n_targets_batch=N_TARGETS_BATCH,
					n_alphas_batch=N_ALPHAS_BATCH, n_targets_batch_refit=N_TARGETS_BATCH_REFIT)

				banded_model = BandedRidgeCV(groups="input", solver=solver, 
					solver_params=solver_params, cv=inner_cv, Y_in_cpu=Y_in_cpu, force_cpu=force_cpu)


			pipeline = create_banded_model(banded_model, delays=delays, feature_space_infos=feature_space_infos, 
				n_jobs=n_jobs)

		else:
			print(f'Using multiple kernel ridge')

			solver_function = MultipleKernelRidgeCV.ALL_SOLVERS[solver]

			if solver == 'group_level_random_search':

				if not all([y.shape == Y[0].shape for y in Y]):
					raise ValueError("To use group level random search, all "
						"groups need to have same number of samples.")

				n_samples_group = Y[0].shape[0]

				solver_params = dict(n_samples_group=n_samples_group, n_iter=N_ITER, alphas=ALPHAS, 
					n_targets_batch=N_TARGETS_BATCH, n_alphas_batch=N_ALPHAS_BATCH, 
					n_targets_batch_refit=N_TARGETS_BATCH_REFIT, Ks_in_cpu=force_cpu)

				mkr_model = GroupLevelMultipleKernelRidgeCV(kernels="precomputed", solver=solver, 
					solver_params=solver_params, cv=inner_cv, Y_in_cpu=Y_in_cpu)
				
			elif solver == 'random_search':
				solver_params = dict(n_iter=N_ITER, alphas=ALPHAS, n_targets_batch=N_TARGETS_BATCH,
					n_alphas_batch=N_ALPHAS_BATCH, n_targets_batch_refit=N_TARGETS_BATCH_REFIT, Ks_in_cpu=force_cpu)

				mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
								  solver_params=solver_params, cv=inner_cv, Y_in_cpu=Y_in_cpu)

			elif solver == 'hyper_gradient':
				solver_params = dict(max_iter=N_ITER, n_targets_batch=N_TARGETS_BATCH, tol=1e-3,
					initial_deltas="ridgecv", max_iter_inner_hyper=1, hyper_gradient_method="direct")

				mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
												  solver_params=solver_params, cv=inner_cv, Y_in_cpu=Y_in_cpu)

			pipeline = create_banded_model(mkr_model, delays=delays, feature_space_infos=feature_space_infos, 
				kernel="linear", n_jobs=n_jobs, force_cpu=force_cpu)
	# Single feature space: use standard kernel ridge
	else:
		solver_params = dict(n_targets_batch=N_TARGETS_BATCH, n_alphas_batch=N_ALPHAS_BATCH,
						   n_targets_batch_refit=N_TARGETS_BATCH_REFIT)

		ridge = KernelRidgeCV(kernel="linear", alphas=ALPHAS, cv=inner_cv, Y_in_cpu=Y_in_cpu, force_cpu=force_cpu)

		pipeline = make_pipeline(scaler, delayer, ridge)

	return pipeline

##################################
##### MODEL EVALUATION ###########
##################################

BANDED_RIDGE_MODELS = [
	'GroupRidgeCV',
	'BandedRidgeCV',
	'GroupLevelBandedRidgeCV',
]

KERNEL_RIDGE_MODELS = [
	'KernelRidgeCV',
	'MultipleKernelRidgeCV',
	'GroupLevelMultipleKernelRidgeCV'
]

def get_all_banded_metrics(pipeline, X_test, Y_test, use_split=False):
	"""Compute comprehensive metrics for encoding model.

	Parameters
	----------
	pipeline : sklearn.pipeline.Pipeline
		Fitted encoding pipeline
	X_test : np.ndarray
		Test features
	Y_test : np.ndarray
		Test targets
	use_split : bool
		Whether to compute split metrics

	Returns
	-------
	results : dict
		Dictionary containing predictions, correlations, R2, and residuals
	"""
	backend = get_backend()

	# Get reference array for type casting
	if pipeline[-1].__class__.__name__ in BANDED_RIDGE_MODELS:
		ref_arr = pipeline[-1].__dict__['coef_']
	elif pipeline[-1].__class__.__name__ in KERNEL_RIDGE_MODELS:
		ref_arr = pipeline[-1].__dict__['dual_coef_']
	else:
		raise ValueError(f'Model must be a form of banded ridge or kernel ridge model')

	X_test = backend.asarray_like(X_test, ref_arr)
	Y_test = backend.asarray_like(Y_test, ref_arr)

	results = {}

	metrics = {
		'correlation': getattr(himalaya.scoring, 'correlation_score'),
		'correlation-split': getattr(himalaya.scoring, 'correlation_score_split'),
		'r2': getattr(himalaya.scoring, 'r2_score'),
		'r2-split': getattr(himalaya.scoring, 'r2_score_split')
	}

	# predict and make as same type of array as Y_test
	Y_pred = pipeline.predict(X_test)
	Y_pred = backend.asarray_like(Y_pred, Y_test)
	results['prediction'] = Y_pred

	if use_split:
		Y_pred_split = pipeline.predict(X_test, split=True)
		Y_pred_split = backend.asarray_like(Y_pred_split, Y_test)
		results['prediction-split'] = Y_pred_split

	for metric, fx in metrics.items():
		if 'split' in metric:
			if use_split:
				score = fx(Y_test, Y_pred)
			else:
				continue
		else:
			score = fx(Y_test, Y_pred)

		results[metric] = score

	# Calculate residuals
	results['residuals'] = (Y_test - results['prediction'])

	if use_split:
		results['residuals-split'] = (Y_test - results['prediction-split'])

	# Move to CPU and convert to numpy
	results = {k: np.asarray(backend.to_cpu(v)) for k, v in results.items()}

	return results

##################################
##### MODEL PERSISTENCE ##########
##################################

def save_model_parameters(pipeline):
	"""Save model parameters to dictionary for serialization.

	Parameters
	----------
	pipeline : sklearn.pipeline.Pipeline
		Fitted encoding pipeline

	Returns
	-------
	d : dict
		Dictionary containing model info and hyperparameters
	"""
	backend = get_backend()

	d = {}

	d['info'] = {
		'module': pipeline[-1].__class__.__module__,
		'name': pipeline[-1].__class__.__name__,
	}

	if d['info']['name'] in BANDED_RIDGE_MODELS:
		d['hyperparameters'] = {
			'deltas_': backend.to_cpu(pipeline[-1].__dict__['deltas_']),
			'coef_': backend.to_cpu(pipeline[-1].__dict__['coef_'])
		}
	elif d['info']['name'] in KERNEL_RIDGE_MODELS:
		d['hyperparameters'] = {
			'deltas_': backend.to_cpu(pipeline[-1].__dict__['deltas_']),
			'dual_coef': backend.to_cpu(pipeline[-1].__dict__['dual_coef_'])
		}
	else:
		raise ValueError(f'Model must be a form of banded ridge or kernel ridge model')

	return d

def load_model_from_parameters(d, args={}):

	# make sure we use the backend to cast to type
	backend = get_backend()

	module = __import__(model_info['info']['module'], fromlist=[model_info['info']['name']])
	base_ = getattr(module, model_info['info']['name'])(**args)

	for k, v in model_info['hyperparameters'].items():
		base_.__dict__[k] = backend.to_cpu(v)
		
	return base_

##################################
##### SIGNAL RESAMPLING ##########
##################################

# Lanczos interpolation adapted from Huth Lab
# https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/ridge_utils/interpdata.py

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
	"""Interpolate data using Lanczos resampling.

	Parameters
	----------
	data : np.ndarray
		Data to interpolate (rows = timepoints, columns = features)
	oldtime : np.ndarray
		Original time points
	newtime : np.ndarray
		Target time points (evenly spaced)
	window : int
		Number of lobes in sinc function
	cutoff_mult : float
		Cutoff frequency multiplier
	rectify : bool
		Whether to rectify positive and negative components separately

	Returns
	-------
	newdata : np.ndarray
		Interpolated data at new time points
	"""
	# Calculate cutoff frequency from target sampling rate
	cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult

	# Build sinc interpolation matrix
	sincmat = np.zeros((len(newtime), len(oldtime)))
	for ndi in range(len(newtime)):
		sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)

	if rectify:
		# Interpolate positive and negative components separately
		newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)),
							np.dot(sincmat, np.clip(data, 0, np.inf))])
	else:
		newdata = np.dot(sincmat, data)

	return newdata

def lanczosfun(cutoff, t, window=3):
	"""Compute windowed sinc (Lanczos) function.

	Parameters
	----------
	cutoff : float
		Cutoff frequency
	t : float or np.ndarray
		Time points
	window : int
		Number of lobes (window size)

	Returns
	-------
	val : float or np.ndarray
		Lanczos function values
	"""
	t = t * cutoff
	val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
	val[t==0] = 1.0
	val[np.abs(t)>window] = 0.0
	return val