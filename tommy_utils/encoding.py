import os, sys
import json
import pandas as pd
import numpy as np
import itertools
from operator import itemgetter

from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import check_cv
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

import himalaya
from himalaya.backend import get_backend, set_backend

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

import torch
import torchvision 
import torchaudio

from torchvision import transforms
import torch.nn.functional as F
import torchlens as tl

from .delayer import Delayer
from .custom_solvers import solve_group_level_group_ridge_random_search, GroupLevelBandedRidge
from . import nlp

# modify banded ridge to contain our custom solver
BandedRidgeCV.ALL_SOLVERS['group_level_random_search'] = solve_group_level_group_ridge_random_search

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
	]
}

# get path of the encoding_utils file --> find the relative path of the phonemes file
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PHONEMES_FN = os.path.join(FILE_DIR, 'data/cmudict-0.7b.phones.txt')
CMU_PHONEMES = pd.read_csv(PHONEMES_FN, header=None, names=['phoneme', 'type'], sep="\t")

def get_modality_features(modality):
	
	if modality == 'audiovisual':
		items = ['visual', 'audio', 'language']
	elif modality == 'audio':
		items = ['audio', 'language']
	elif modality == 'text':
		items = ['language']
	
	modality_features = []
	
	for item in items:

		if not ENCODING_FEATURES[item]:
			continue

		modality_features.extend(ENCODING_FEATURES[item])
	
	return modality_features

def load_gentle_transcript(transcript_fn, start_offset=None):
	
	# Load the stimulus transcript
	with open(transcript_fn) as f:
		data = json.load(f)

	transcript = data['transcript']

	# Get the transcript as a dataframe
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

	# Convert all words to lowercase
	df_transcript['word'] = df_transcript.word.str.lower()
	
	# If offset is necessary to apply to the stimulus
	if start_offset:
		df_transcript['start'] = df_transcript['start'] - start_offset
		df_transcript['end'] = df_transcript['end'] - start_offset
	
	return df_transcript

def create_phoneme_features(df_transcript):
	'''
	converts a gentle transcript phoneme transcription 
	to a one hot feature space (39 dimensions)
	'''

	phoneme_time_features = []
	
	# go through each transcribed word
	for i, row in df_transcript.iterrows():
		# set the start time of the phonemes
		phoneme_start = row['start']
		word_phonemes = []
		
		if row['case'] != 'success' or row['alignedWord'] == '<unk>':
			continue
		
		for item in row['phones']:
			# get the phoneme discarding gentle's info --> this aligns to CMU dictionary
			# then turn to one hot vector
			phoneme = item['phone'].split('_')[0].upper()
			one_hot_phoneme = np.asarray(CMU_PHONEMES['phoneme'] == phoneme).astype(int)
		
			# ensure that vector is truly one hot
			if sum(one_hot_phoneme) != 1:
				print (i)
			assert (sum(one_hot_phoneme) == 1)
		
			# then save the phoneme start + one hot vector and increment time
			phoneme_info = (phoneme_start, one_hot_phoneme)
			word_phonemes.append(phoneme_info)
			phoneme_start += item['duration']

		phoneme_time_features.append(word_phonemes)
	
	phoneme_time_features = sum(phoneme_time_features, [])
	times, phoneme_features = [np.stack(item) for item in zip(*phoneme_time_features)]

	print (f'Phoneme feature space is size: {phoneme_features.shape}')

	return times, phoneme_features

def create_word_features(df_transcript, word_model):
	'''
	given a gentle transcript, get word features 
	using a gensim model
	'''
	word_time_features = []
	
	for i, row in df_transcript.iterrows():
		
		word = row['word']
		
		if 'fasttext' in str(type(word_model)) or word in word_model.key_to_index:
			word_vector = word_model[word]
		else:
			continue
	
		word_time_info = (row['start'], word_vector)
		word_time_features.append(word_time_info)
	
	times, word_features = [np.stack(item) for item in zip(*word_time_features)]

	print (f'Word feature space is size: {word_features.shape}')

	return times, word_features

def create_transformer_features(df_transcript, tokenizer, model, window_size=25, bidirectional=False, add_punctuation=False):
	'''
	given a gentle transcript and a transformer architecture, get
	word embeddings for each word (contextualized by a context window)
	'''
	word_time_features = []

	# create a list of indices that we will iterate through to sample the transcript
	segments = nlp.get_segment_indices(n_words=len(df_transcript), window_size=window_size, bidirectional=bidirectional)

	for (i, row), segment in zip(df_transcript.iterrows(), segments):

		print (f'Processing segment {i+1}/{len(df_transcript)}')
		
		# get the prepared input for the word embedding extraction
		inputs = nlp.transcript_to_input(df_transcript, segment, add_punctuation=add_punctuation)

		# select the last word only to extract embeddings for
		# returns an array of size sentence x n_layers x size
		word_embeddings = nlp.extract_word_embeddings([inputs], tokenizer, model, word_indices=-1)
		word_embeddings = word_embeddings.squeeze()
		
		word_time_info = (row['start'], word_embeddings.squeeze())
		word_time_features.append(word_time_info)
	
	times, word_features = [np.stack(item) for item in zip(*word_time_features)]

	# move layers to be the first element
	word_features = np.moveaxis(word_features, 0, 1)

	print (f'Transformer feature space is size: {word_features.shape}')
	
	return times, word_features

###################################
##### Stuff for vision models #####
###################################

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

def chunk_video_clips(video_clips, batch_size):
	'''
	Helper function to gather and transform a videoclips object
	'''
	idxs = (i for i in range(video_clips.num_clips()))
	while True:
		sl = list(itertools.islice(idxs, batch_size))
		if not sl:
			break

		# video clips returns video, audio, info, video_idx
		# we need to roll the video to have channels first
		# then transform and stack
		frames = [torch.moveaxis(video_clips.get_clip(i)[0], -1, 0).squeeze() for i in sl]
		yield torch.stack(frames)

def create_vision_features(image_info, model_name, batch_size=8):
	'''
	given a set of frames and the meta 
	'''

	if model_name == 'clip':
		tokenizer, model = nlp.load_multimodal_model(model_name=model_name, modality='vision')
	else:
		model = load_torchvision_model(model_name)
		model_layers = VISION_MODELS_DICT[model_name]

	times, images = image_info
	
	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	
	vision_features = []

	for i, batch in enumerate(chunk_video_clips(images, batch_size=batch_size)):
	
		print (f'Processing batch {i+1}/{len(images)//batch_size}')
		
		# pass the frames through the transform 
		# use torchlens to log the forward pass of the model
		# batch = torch.stack([transform(x) for x in batch])
		if model_name == 'clip':
			inputs = tokenizer(images=batch, return_tensors='pt')
			vision_embeddings = model.get_image_features(**inputs).detach()
		else:
			batch = torch.stack([transform(x) for x in batch])
			model_output = tl.log_forward_pass(model, batch, layers_to_save=model_layers)
			vision_embeddings = get_layer_tensors(model_output)

		vision_features.append(vision_embeddings)
	
	# stack tensors for each layer across the batches
	if model_name == 'clip':
		vision_features = np.vstack(vision_features)
		print (f'Vision feature space has shape {vision_features.shape}')
	else:
		vision_features = [np.vstack(item) for item in zip(*vision_features)]
	
		print (f'Vision feature space has {len(vision_features)} layers')
	
	return times, vision_features

###################################
##### Stuff for audio models ######
###################################

def load_torchaudio_model(model_name):
	bundle = getattr(torchaudio.pipelines, model_name.upper())
	model = bundle.get_model()
	return bundle, model

def create_spectral_features(audio, sr, n_fft = 1024, hop_length=512, n_mels=128):
	
	times = np.linspace(0, audio.shape[-1]/sr, sr)
	
	mel_spectrogram = torchaudio.transforms.MelSpectrogram(
		sample_rate=sr,
		center = True,
		n_fft=n_fft,
		hop_length=hop_length,
		pad_mode="reflect",
		power=2.0,
		norm="slaney",
		n_mels=n_mels,
		mel_scale="htk",
	)

	melspec = mel_spectrogram(audio)
	
	padding = (sr - melspec[0].shape[-1]) / 2
	left_pad = np.ceil(padding).astype(int)
	right_pad = np.floor(padding).astype(int)

	# here, pad = (padding_left, padding_right, padding_top, padding_bottom)
	source_pad = F.pad(melspec[0], pad=(left_pad, right_pad,  0, 0)).T

	return times, source_pad

##################################
##### MODEL SETUP FUNCTIONS ######
##################################

from sklearn.utils.validation import check_random_state

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
	
	# With permutations, we are sure that all runs are used as validation runs.
	# However here for n_runs_out > 1, a run can be chosen twice as validation
	# in the same split.
	# all_val_runs = np.array(
	#     [random_state.permutation(n_runs) for _ in range(n_runs_out)])

	if n_runs_out >= len(run_onsets):
		raise ValueError("More runs requested for validation than there are "
						 "total runs. Make sure that n_runs_out is less than "
						 "than the number of runs (e.g., len(run_onsets)).")

	all_val_runs = np.array(list(itertools.combinations(range(n_runs), n_runs_out)))
	all_val_runs = random_state.permutation(all_val_runs)

	print (f'Total number of validation runs: {len(all_val_runs)}')
	
	all_samples = np.arange(n_samples)
	runs = np.split(all_samples, run_onsets[1:])
	
	if any(len(run) == 0 for run in runs):
		raise ValueError("Some runs have no samples. Check that run_onsets "
						 "does not include any repeated index, nor the last "
						 "index.")
	
	for val_runs in all_val_runs: #.T:
	
		train = [runs[jj] for jj in range(n_runs) if jj not in val_runs]
		val = [runs[jj] for jj in range(n_runs) if jj in val_runs]

		# ensure that we pulled the right number of validation runs
		assert (len(val) == n_runs_out)

		# stack them horizontally for use in indexing
		train, val = [np.hstack(x) for x in [train, val]]

		# ensure no overlap between sets
		assert (not np.isin(train, val).any())
		
		yield train, val

def load_banded_features(fns, feature_names):
	'''
	Given a set of filenames create prerequisites
	for banded ridge regression

	Returns
		- concatenated features across separate feature spaces
		- list of names of each feature space and the index in 
			the overall list of feature spaces
	'''
	features = [np.load(fn) for fn in fns]
	features_dim = [feature.shape[1] for feature in features]
	
	# create slices by cumulative sum of spaces
	feature_space_idxs = np.concatenate([[0], np.cumsum(features_dim)])
	feature_space_slices = [slice(*item) for item in zip(feature_space_idxs[:-1], feature_space_idxs[1:])]

	# then combine with the names of each space
	assert (len(feature_space_slices) == len(feature_names))

	# concatenate the feature spaces together
	features = np.concatenate(features, axis=1)

	# now pair the feature space info
	feature_space_info = [(name, slice) for name, slice in zip(feature_names, feature_space_slices)]

	return features, feature_space_info

def get_concatenated_data(data, indices):
	
	if len(indices) > 1:
		data_split = np.concatenate(itemgetter(*indices)(data), axis=0)
	else:
		data_split = np.stack(itemgetter(*indices)(data), axis=0)
	return data_split

def get_train_test_splits(x, y, train_indices, test_indices, precision='float32', group_level=False):
	
	# Get train data
	if group_level:
		assert (len(x) == 1)
		X_train = get_concatenated_data(x, [0]).astype(precision)
		X_test = get_concatenated_data(x, [0]).astype(precision)
	else:
		X_train = get_concatenated_data(x, train_indices).astype(precision)
		X_test = get_concatenated_data(x, test_indices).astype(precision)

	# Get test data
	Y_train = get_concatenated_data(y, train_indices).astype(precision)
	Y_test = get_concatenated_data(y, test_indices).astype(precision)
	
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
	'''
	Builds an encoding model given two lists of equal length:
		- X: predictors -->
			- if X is a list of arrays, then all are assumed to come 
				from the same feature space
			- if X is a list of lists, then 
		- Y: values to be predicted 
		- inner_folds: number of folds within the inner loop to be used

	The size of X and Y should be equal
	
	The number of elements within each item of X are the number
	of feature spaces to be used
	
	'''

	## Static parameters for solver
	N_TARGETS_BATCH = n_targets_batch
	N_ALPHAS_BATCH = n_alphas_batch
	N_TARGETS_BATCH_REFIT = n_targets_batch_refit

	# for multiple kernel ridge
	N_ITER = n_iter # --> should be higher remember to change
	ALPHAS = alphas
	RANDOM_STATE = 42

	# ensure that X and Y are the same length
	if solver == 'group_level_random_search':
		assert all([X[0].shape[0] == y.shape[0] for y in Y])
	else:
		assert (len(X) == len(Y))

	n_samples = np.concatenate(X).shape[0]
	n_features = np.concatenate(X).shape[1]

	## Standard parameters
	# scaler --> zscores the predictors
	# delayer --> estimates the HRF
	scaler = StandardScaler(with_mean=True, with_std=False)
	delayer = Delayer(delays=delays) # delays are in indices --> needs to be scales to TRs

	# if feature info is provided we have multiple feature spaces and use
	# banded ridge
	if feature_space_infos:

		if (n_samples > n_features or force_banded_ridge):

			print (f'Using banded ridge')

			## TLB --> TRY ADDING IN RETURN_WEIGHTS AND SEE WHAT HAPPENS
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

			print (f'Using multiple kernel ridge')

			# We use 20 random-search iterations to have a reasonably fast example.

			## TLB --> TRY ADDING IN RETURN_WEIGHTS AND SEE WHAT HAPPENS
			solver_function = MultipleKernelRidgeCV.ALL_SOLVERS[solver]

			if solver == 'random_search':
				solver_params = dict(n_iter=N_ITER, alphas=ALPHAS, n_targets_batch=N_TARGETS_BATCH,
					n_alphas_batch=N_ALPHAS_BATCH, n_targets_batch_refit=N_TARGETS_BATCH_REFIT,Ks_in_cpu=force_cpu)

			elif solver == 'hyper_gradient':
				solver_params = dict(max_iter=N_ITER, n_targets_batch=N_TARGETS_BATCH, tol=1e-3,
					initial_deltas="ridgecv", max_iter_inner_hyper=1, hyper_gradient_method="direct")

			mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
											  solver_params=solver_params, cv=inner_cv, Y_in_cpu=Y_in_cpu)

			pipeline = create_banded_model(mkr_model, delays=delays, feature_space_infos=feature_space_infos, 
				kernel="linear", n_jobs=n_jobs, force_cpu=force_cpu)


	else:       
		solver_params=dict(n_targets_batch=N_TARGETS_BATCH, n_alphas_batch=N_ALPHAS_BATCH, 
						   n_targets_batch_refit=N_TARGETS_BATCH_REFIT)
		
		ridge = KernelRidgeCV(kernel="linear", alphas=ALPHAS, cv=inner_cv, Y_in_cpu=Y_in_cpu, force_cpu=force_cpu)
		
		pipeline = make_pipeline(scaler, delayer, ridge)

	# return outer_cv, pipeline
	return pipeline

#################################
##### MODEL SAVING FUNCTIONS ####
#################################

def get_all_banded_metrics(pipeline, X_test, Y_test):

	backend = get_backend()

	print (backend)

	device = pipeline[-1].__dict__.coef_.device()

	X_test = backend.asarray(X_test, device=device)
	Y_test = backend.asarray(Y_test, device=device)

	metrics = {
		'correlation': getattr(himalaya.scoring, 'correlation_score'),
		'correlation-split': getattr(himalaya.scoring, 'correlation_score_split'),
		'r2': getattr(himalaya.scoring, 'r2_score'),
		'r2-split': getattr(himalaya.scoring, 'r2_score_split')
	}

	Y_pred = pipeline.predict(X_test)
	Y_pred_split = pipeline.predict(X_test, split=True)

	results = {
		'prediction': Y_pred,
		'prediction-split': Y_pred_split,
	}

	for metric, fx in metrics.items():
		if 'split' in metric:
			score = fx(Y_test, Y_pred)
		else:
			score = fx(Y_test, Y_pred)
		
		results[metric] = score

	# now calculate residuals
	results['residuals'] = (Y_test - results['prediction'])
	results['residuals-split'] = (Y_test - results['prediction-split'])

	# move to cpu and cast as numpy array
	results = {k: np.asarray(backend.to_cpu(v)) for k, v in results.items()}

	return results

def save_model_parameters(pipeline):
	'''
	Given a pipeline used to build 
	'''

	BANDED_RIDGE_MODELS = [
		'GroupLevelBandedRidgeCV', 
		'GroupRidgeCV', 
		'BandedRidgeCV', 
		'KernelRidgeCV', 
		'MultipleKernelRidgeCV'
	]

	backend = get_backend()

	d = {}

	d['info'] = {
		'module': pipeline[-1].__class__.__module__,
		'name': pipeline[-1].__class__.__name__,
	}

	if d['info']['name'] in BANDED_RIDGE_MODELS:    
		d['hyperparameters'] = {
			'deltas_':backend.to_cpu(pipeline[-1].__dict__['deltas_']),
			'coef_': backend.to_cpu(pipeline[-1].__dict__['coef_'])
		}
	else:
		raise ValueError(f'Model must be a form of banded ridge model')

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
##### DOWNSAMPLING FUNCTIONS #####
##################################

## Taken from Huth/Lebel repository for Lancosz filtering
# Repository found here: https://github.com/HuthLab/deep-fMRI-dataset/blob/master/encoding/ridge_utils/interpdata.py#L154

def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
	"""Interpolates the columns of [data], assuming that the i'th row of data corresponds to
	oldtime(i). A new matrix with the same number of columns and a number of rows given
	by the length of [newtime] is returned.
	
	The time points in [newtime] are assumed to be evenly spaced, and their frequency will
	be used to calculate the low-pass cutoff of the interpolation filter.
	
	[window] lobes of the sinc function will be used. [window] should be an integer.
	"""
	## Find the cutoff frequency ##
	cutoff = 1/np.mean(np.diff(newtime)) * cutoff_mult
	# print "Doing lanczos interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window)
	
	## Build up sinc matrix ##
	sincmat = np.zeros((len(newtime), len(oldtime)))
	for ndi in range(len(newtime)):
		sincmat[ndi,:] = lanczosfun(cutoff, newtime[ndi]-oldtime, window)
	
	if rectify:
		newdata = np.hstack([np.dot(sincmat, np.clip(data, -np.inf, 0)), 
							np.dot(sincmat, np.clip(data, 0, np.inf))])
	else:
		## Construct new signal by multiplying the sinc matrix by the data ##
		newdata = np.dot(sincmat, data)

	return newdata

def lanczosfun(cutoff, t, window=3):
	"""Compute the lanczos function with some cutoff frequency [B] at some time [t].
	[t] can be a scalar or any shaped numpy array.
	If given a [window], only the lowest-order [window] lobes of the sinc function
	will be non-zero.
	"""
	t = t * cutoff
	val = window * np.sin(np.pi*t) * np.sin(np.pi*t/window) / (np.pi**2 * t**2)
	val[t==0] = 1.0
	val[np.abs(t)>window] = 0.0
	return val# / (val.sum() + 1e-10)