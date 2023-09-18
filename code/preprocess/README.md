# Steps for preprocessing transcripts

Below are general steps involved in preprocessing gentle aligned transcripts for next word/sentence/phrasal prediction experiments. While the specifics will vary by experiment, the order of scripts run will be the same.

All these steps assume that a gentle forced-aligned transcript exists for a given stimulus. 

## Next-Word Prediction

To set up a next-word prediction experiment, run the scripts in the following order:

1. Run `select_next_word_candidates.py` with the following arguments:
	- `task`: name of the task that has a gentle transcript
	- `overwrite`: True/False --> whether to overwrite existing edited files
	- This will output two sets of files:
		- A `.csv` and `.json` file containing the times and candidate prediction words (these are edited by `segment_audio.py`)
		- Backup files to the `src` directory (these will not be edited by the following scripts)
2. The next steps may require multiple iterations:
	1. Run `segment_audio.py` --> this will produce the following files:
		- A .csv file containing lines indicating following:
			- `filename`: indicates the name of a segment
			- `word_index`: relates these files to the preprocessed dataframe
			- `critical_word`: the word that is meant to be predicted
			- `checked`: a boolean --> indicates if the file has been checked for word leakage
			- `adjusted`: a boolean --> indicates if times were adjusted in Praat
		- An audio file per critical word (e.g., a word that is meant to be predicted by the participant)
	2. Perform editing of transcript times:
		- Open audio file and transcript in Praat 
		- Open the .csv file of audio segments
		- Go through each file listening for leakage at the end of the file. Specifically listen for any part of the word that is to be predicted 
			- If there is no leakage, set the `checked` column for that file to `1`
			- If there is leakage, adjust the times in Praat. Set both the `checked` and `adjusted` column to `1`
		- After all files are checked, save out the Praat transcript:
			- Use the following format `TASK-transcript_praat.TextGrid`
	3. Run `update_word_times.py` --> this will adjust the preprocessed file times with those in the Praat textgrid