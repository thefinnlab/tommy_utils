import pandas as pd

def create_stim_times_regressor(df, onset_var, timing_offset=0):
	'''
	Takes a DataFrame containing sentence onsets, durations,
	and similarities. Creates an amplitude-modulated regressor
	(with duration)
	'''
	print (f'There are {len(df)} entries')

	df = df.dropna()
	print (f'There are {len(df)} entries after removing NaNs')

	#create offset for onset time if provided
	df[onset_var] = df[onset_var] + timing_offset

	regressor = [f'{onset:.2f}' for onset in df[onset_var].values]
	return df, regressor


def create_AM_regressor(df, AM_var, onset_var, timing_offset=0):
	'''
	Create an amplitude-modulated regressor. Pairs
	event onset times with amplitude values.
	'''
	print (f'There are {len(df)} entries')

	df = df.dropna()
	print (f'There are {len(df)} entries after removing NaNs')
	
	#create offset for onset time if provided
	df[onset_var] = df[onset_var] + timing_offset
	regressor = [f'{onset:.2f}*{amplitude:.2f}' for onset, amplitude in df[[onset_var, AM_var]].values]
	
	return df, regressor

def create_DM_regressor(df, DM_var, onset_var, timing_offset=0, min_duration=0):
	'''
	Takes a DataFrame containing sentence onsets, durations,
	and similarities. Creates an amplitude-modulated regressor
	(with duration)
	'''
	print (f'There are {len(df)} entries')

	df = df.dropna()
	print (f'There are {len(df)} entries after removing NaNs')

	#create offset for onset time if provided
	df[onset_var] = df[onset_var] + timing_offset
	
	# add the time filter as a column to the dataframe
	time_filter = df[DM_var] >= min_duration
	df['time_filter'] = time_filter
	
	print (f'{sum(time_filter)} entries are longer than {min_duration} seconds')

	filtered = df[df['time_filter']]
	regressor = [f'{onset:.2f}:{duration:.2f}' for onset, duration in filtered[[onset_var, DM_var]].values]
	return df, regressor

def create_DM_AM_regressor(df, DM_var, AM_var, onset_var, timing_offset=0, min_duration=0):
	'''
	Takes a DataFrame containing sentence onsets, durations,
	and similarities. Creates an amplitude-modulated regressor
	(with duration)
	'''
	print (f'There are {len(df)} entries')

	df = df.dropna()
	print (f'There are {len(df)} entries after removing NaNs')

	#create offset for onset time if provided
	df[onset_var] = df[onset_var] + timing_offset
	
	# add the time filter as a column to the dataframe
	time_filter = df[DM_var] >= min_duration
	df['time_filter'] = time_filter
	
	print (f'{sum(time_filter)} entries are longer than {min_duration} seconds')

	filtered = df[df['time_filter']]
	regressor = [f'{onset:.2f}*{amplitude:.2f}:{duration:.2f}' for onset, duration, amplitude in filtered[[onset_var, DM_var, AM_var]].values]
	return df, regressor

def write_regressor(regressor_list, out_fn):
	with open(out_fn, 'w') as f:
		f.write(' '.join(regressor_list))