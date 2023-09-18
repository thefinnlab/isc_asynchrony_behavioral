# isc_asynchrony_behavioral

Scripts for running language prediction experiments.

## Alignment Cleaning Instructions

Below are general steps involved in preprocessing gentle aligned transcripts for next word/sentence/phrasal prediction experiments. While the specifics will vary by experiment, the order of scripts run will be the same.

All these steps assume that a gentle forced-aligned transcript exists for a given stimulus. 

### Setup

Follow these steps to perform setup of the alignment cleaning scripts:
1. Open a terminal window at the directory `alignment_cleaning` 
2. Set up a conda environment by running `conda env create --name alignment_cleaning --file=environment.yml`
3. Activate the conda environment by typing `conda activate alignment_cleaning`
4. Open the `config.py` located in `/code/utils/` directory
5. Change the base directory to the following `/PATHTOFOLDER/alignment_cleaning/` --> to find what the path is type `pwd`

### Creating the prediction candidates file 

The following steps describe how to create candidate words for prediction:

1. Run `select_next_word_candidates.py` with the name of a task (any of the transcript names)
	- To test this out, run the command `python select_next_word_candidates.py black`
	- This uses python to run the script `select_next_word_candidates` on the task `black`
2. Clean hyphenated words:
	- Sometimes the transcription makes mistakes in which words should be hyphenated. An example of a hyphenated word is "pre-recorded" while an example of a mistake is "am-all".
	- Select words meant to be hyphenated by typing `y` for yes
	- Flag words not meant to be hyphenated by typing `n` for no
3. Clean named entities:
	- A named entity is a real-world object (e.g., person, location, organization, product). As these words are likely not known by participants and are specific to the current transcript we remove them as possibilities for prediction. Some examples of named entities are would be "Tommy Botch", "Dartmouth College", "Hanover", "New Hampshire"
	- Select words that are named entities by typing `y` for yes
	- Flag words not meant to be hyphenated by typing `n` for no

After completing these steps, the following directory should be created `/stimuli/preprocessed/TASK/` with files required to run the audio segmentations.

### Segmenting audio and ensuring no word leakage

Next, we check all words meant to be predicted for leakage (e.g., the ability to hear the word or part of the word in advance of it being said). The following steps may require multiple iterations.

1. Run `segment_audio.py` with the name of a task (any of the transcript names)
	- To test this out, run the command `python segment_audio.py TASK` where task is the of a task (e.g., black)
	- This script will output a file named `TASK_transcript-segments.csv`
2. Open files for editing:
	- The directory `/stimuli/cut_audio/TASK/`
	- The segments file `TASK_transcript-segments.csv`
	- Within Praat open:
		- The task audio --> found as filename `/stimuli/audio/TASK_audio.wav`
		- The TextGrid --> found as filename `/stimuli/preprocessed/TASK/black_transcript-praat.TextGrid`
3. Go through each of the cut audio files, following along in Praat:
	- If the cut audio file has leakage:
		1. Adjust the audio timings in Praat
		2. Within `TASK_transcript-segments.csv`, place a `1` in the columns `checked` and `adjusted`
	- If the cut audio file does not have leakage:
		1. Within `TASK_transcript-segments.csv`, place a `1` in the columns `checked`

After checking all the cut audio files, rerun `segment_audio.py`. This will adjust the audio times. You will then need to repeat the following steps for the files that had the audio times changed.