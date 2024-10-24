import os
import pickle
from collections import OrderedDict, defaultdict
import numpy as np
import scipy.stats as stats
from scipy import signal
from scipy.fftpack import dct, fft
from tqdm import tqdm

from src.utils.text_processing import (
    find_stress_syllable_start,
    syllabify,
    CelexReader,
    read_textgrid_file,
    read_lab_file,
    remove_breaks_from_lab_lines,
    python_lowercase_remove_punctuation,
    nb_syllables,
)
from src.utils.helsinki_features import WordBreakExtractor
from src.utils.utils import min_length_of_lists, sec_to_idx, equal_length_or_none
from src.utils.prosody_tools.misc import read_wav, normalize_std
from src.utils.prosody_tools import (
    f0_processing,
    smooth_and_interp,
    energy_processing,
    duration_processing,
)

INVALID_SYMBOLS = ["<unk>"]

class ProsodyFeatureExtractor:
    def __init__(self, csv_path, data_cache=None, language=None, 
                 extract_f0=False, f0_mode="dct", f0_n_coeffs=4, f0_stress_localizer=None, 
                 f0_window=500, f0_resampling_length=100, celex_path=None, extract_energy=False, 
                 energy_mode="mean", extract_word_duration=False, word_duration_mode="syllable_norm", 
                 extract_duration=False, extract_pause_before=False, extract_pause_after=False, 
                 extract_prominence=False, prominence_mode="mean", f0_min=50, f0_max=400, 
                 f0_voicing=50, energy_min_freq=200, energy_max_freq=5000, f0_weight=1.0, 
                 energy_weight=0.5, duration_weight=1, unallowed_symbols=INVALID_SYMBOLS):
        
        # Initialize parameters
        self.csv_path = csv_path
        self.data_cache = data_cache
        self.language = language

        # Feature extraction flags
        self.extract_f0 = extract_f0
        self.f0_mode = f0_mode
        self.f0_n_coeffs = f0_n_coeffs
        self.f0_stress_localizer = f0_stress_localizer
        self.f0_window = f0_window
        self.f0_resampling_length = f0_resampling_length
        
        # Initialize Celex manager if needed
        if self.extract_f0 and self.language == 'stress':
            self.celex_path = celex_path
            self.celex_manager = CelexReader(celex_path)
        
        # More feature extraction flags
        self.extract_energy = extract_energy
        self.energy_mode = energy_mode
        self.extract_word_duration = extract_word_duration
        self.word_duration_mode = word_duration_mode
        self.extract_duration = extract_duration
        self.extract_pause_before = extract_pause_before
        self.extract_pause_after = extract_pause_after
        self.extract_prominence = extract_prominence
        self.prominence_mode = prominence_mode

        # F0 and energy parameters
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_voicing = f0_voicing
        self.energy_min_freq = energy_min_freq
        self.energy_max_freq = energy_max_freq

        # Feature weights
        self.f0_weight = f0_weight
        self.energy_weight = energy_weight
        self.duration_weight = duration_weight

        self.unallowed_symbols = unallowed_symbols

        # Initialize pause extractors if needed
        if self.extract_pause_before:
            self.pause_before_extractor = WordBreakExtractor(modes="before")
        if self.extract_pause_after:
            self.pause_after_extractor = WordBreakExtractor(modes="after")

        self.samples = []
        self.file_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        self.extracted_features = self._get_extracted_features()

    def _get_extracted_features(self):
        """Determine which features to extract based on class attributes."""
        features = [
            "f0", "energy", "word_duration", "duration",
            "pause_before", "pause_after", "prominence"
        ]
        return [f for f in features if getattr(self, f"extract_{f}", False)]

    def extract_and_cache_features(self):
        """Extract features from files and cache the results."""
        print(f"Extracted features: {self.extracted_features}")

        self.process_files()  # Assuming this method populates self.samples

        d_correct = defaultdict(list)
        for sample in self.samples:
            d_correct['texts'].append(sample['text'])
            for feature in self.extracted_features:
                if feature == 'f0':
                    d_correct['f0'].append(sample['features']['f0_parameterized'])
                else:
                    d_correct[feature].append(sample['features'][feature])

        self._save_to_cache(dict(d_correct))

    def _save_to_cache(self, data):
        """Save extracted features to a cache file."""
        feature_name = self.extracted_features[0] if self.extracted_features else 'unknown'
        if feature_name == 'f0':
            file_name = f"{feature_name}_{self.f0_mode}_{self.f0_n_coeffs}.pkl"
        else:
            file_name = f"{feature_name}.pkl"

        self.cache_path = os.path.join(self.data_cache, self.file_name, file_name)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved samples to {self.cache_path}")

    def process_files(self, verbose=False):
        """
        Process files based on the input CSV file and extract features.
        
        Returns:
            list: A list of dictionaries containing extracted features and metadata.
        """
        failed_alignments = 0
        total_nb_syllables_not_found = 0

        # Read the CSV file
        df = pd.read_csv(self.csv_path)

        # Ensure required columns are present
        required_columns = ['words_path', 'wav_path']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain the following columns: {required_columns}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
            words_path = row['words_path']
            wav_path = row['wav_path']
            phonemes_path = row.get('phonemes_path', None)

            if not os.path.exists(words_path) or not os.path.exists(wav_path):
                print(f"Error: File not found - {words_path} or {wav_path}")
                continue

            if verbose:
                print(f"Processing file {words_path}")

            try:
                features, nb_syll_not_found = self._extract_features(
                    words_path=words_path,
                    wav_path=wav_path,
                    phonemes_path=phonemes_path,
                )
            except Exception as e:
                print(f'Error in feature extraction: {str(e)}')
                failed_alignments += 1
                continue

            if features is None:
                failed_alignments += 1
                continue

            total_nb_syllables_not_found += nb_syll_not_found

            # Create a sample dictionary with all columns as metadata
            sample = OrderedDict({
                "features": features,
                "words_path": words_path,
                "wav_path": wav_path,
                "phonemes_path": phonemes_path,
            })

            # Add all other columns as metadata
            for col in df.columns:
                if col not in ['words_path', 'wav_path', 'phonemes_path']:
                    sample[col] = row[col]

            self.samples.append(sample)

        if verbose:
            print(f"Failed alignments: {failed_alignments}")
            print(f"Total number of syllables not found: {total_nb_syllables_not_found}")
            print(f"Total processed utterances: {len(self.samples)}")

        return self.samples
        
    def _extract_features(self, words_path, wav_path, phonemes_path=None):
        """Extract all specified features from a single file."""
        features = {}
        nb_syllables_not_found = 0

        # Read audio file
        fs, waveform = read_wav(wav_path)

        # Load information for the current words
        if words_path.endswith('.TextGrid'):
            word_lines, end_time = read_textgrid_file(words_path, tier='word')
        else:
            word_lines = read_lab_file(words_path)
            end_time = float(word_lines[-1][1])

        # Remove breaks from lab lines
        word_lines = remove_breaks_from_lab_lines(word_lines)

        # Now load phonemes if applicable
        if phonemes_path:
            if phonemes_path.endswith('.TextGrid'):
                phoneme_lines, _ = read_textgrid_file(words_path, tier='word')
            else:
                phoneme_lines = read_lab_file(phonemes_path)

            # remove breaks from phoneme lines
            phoneme_lines = remove_breaks_from_lab_lines(phoneme_lines)

        # Extract pauses if needed
        if self.extract_pause_before:
            pause_before = self.pause_before_extractor.extract_from_lab_lines(word_lines)
            if pause_before is None:
                return None, None
            features["pause_before"] = [round(pause, 3) for pause in pause_before if pause is not None]

        if self.extract_pause_after:
            pause_after = self.pause_after_extractor.extract_from_lab_lines(word_lines)
            if pause_after is None:
                return None, None
            features["pause_after"] = [round(pause, 3) for pause in pause_after if pause is not None]
        
        # Check for invalid words
        words = [word for _, _, word in word_lines]
        if any(word in self.unallowed_symbols for word in words):
            return None, None

        features["words"] = words

        # Extract features
        f0 = self._extract_f0(waveform, fs) if self.extract_f0 else None
        energy = self._extract_energy(waveform, fs) if self.extract_energy else None
        duration = self._extract_duration(waveform, fs, word_lines, min_length_of_lists([f0, energy])) if self.extract_duration else None
        prominence = self._extract_prominence(f0, energy, duration) if self.extract_prominence else None

        if self.extract_word_duration:
            features["word_duration"] = self._extract_word_duration(word_lines)

        # Extract per-word features
        if self.extract_f0:
            f0_per_word, cnt_not_found = self._extract_f0_per_word(word_lines, f0, phoneme_lines, end_time)
            nb_syllables_not_found += cnt_not_found
            features["f0_parameterized"] = [self._parameterize_f0(f) for f in f0_per_word]

        if self.extract_energy:
            energy_per_word = self._extract_feature_per_word(word_lines, energy, end_time)
            features["energy"] = (
                [np.mean(e) for e in energy_per_word] if self.energy_mode == "mean" else
                [np.max(e) for e in energy_per_word] if self.energy_mode == "max" else
                energy_per_word
            )

        if self.extract_duration:
            features["duration"] = self._extract_feature_per_word(word_lines, duration, end_time)

        if self.extract_prominence:
            prominence_per_word = self._extract_feature_per_word(word_lines, prominence, end_time)
            features["prominence"] = (
                [np.mean(p) for p in prominence_per_word] if self.prominence_mode == "mean" else
                [np.max(p) for p in prominence_per_word] if self.prominence_mode == "max" else
                prominence_per_word
            )

        return features, nb_syllables_not_found

    def _parameterize_f0(self, f0):
        """Parameterize F0 curve based on specified mode."""
        if self.f0_mode == "dct":
            return dct(f0, type=2, norm="ortho")[: self.f0_n_coeffs]
        elif self.f0_mode == "fft":
            return fft(f0)[: self.f0_n_coeffs]
        elif self.f0_mode == "poly":
            return np.polyfit(np.arange(len(f0)), f0, self.f0_n_coeffs - 1)[: self.f0_n_coeffs]
        elif self.f0_mode == "mean":
            precision = len(f0) // self.f0_n_coeffs
            return np.array([np.mean(f0[i:i+precision]) for i in range(0, len(f0), precision)])
        else:
            raise ValueError(f"Unknown f0_mode: {self.f0_mode}")

    def _extract_f0(self, waveform, fs):
        """Extract and process F0 from waveform."""
        f0_raw = f0_processing.extract_f0(waveform=waveform, fs=fs, f0_min=self.f0_min, 
                                          f0_max=self.f0_max, voicing=self.f0_voicing)
        f0_interpolated = f0_processing.process(f0_raw)
        return normalize_std(f0_interpolated)

    def _extract_energy(self, waveform, fs):
        """Extract and process energy from waveform."""
        energy = energy_processing.extract_energy(waveform=waveform, fs=fs, 
                                                  min_freq=self.energy_min_freq, 
                                                  max_freq=self.energy_max_freq, method="rms")
        energy_smooth = smooth_and_interp.peak_smooth(energy, 30, 3)
        return normalize_std(energy_smooth)

    def _extract_duration(self, waveform, fs, word_lines, resample_length):
        """Extract and process duration signal."""
        duration_signal = duration_processing.duration(word_lines, rate=fs)
        duration_norm = normalize_std(duration_signal)
        return signal.resample(duration_norm, resample_length)

    def _extract_prominence(self, f0, energy, duration):
        """Extract prominence as a weighted combination of F0, energy, and duration."""
        min_length = min(len(f0), len(energy), len(duration))
        f0, energy, duration = f0[:min_length], energy[:min_length], duration[:min_length]
        prominence = (self.f0_weight * f0 + self.energy_weight * energy + 
                      self.duration_weight * duration)
        prominence = smooth_and_interp.remove_bias(prominence, 800)
        return normalize_std(prominence)

    def _extract_word_duration(self, word_lines):
        """Extract word duration based on specified mode."""
        word_duration = [float(end) - float(start) for start, end, _ in word_lines]
        if self.word_duration_mode == "char_norm":
            return [duration / len(word) for duration, (_, _, word) in zip(word_duration, word_lines)]
        elif self.word_duration_mode == "absolute":
            return [round(duration, 3) for duration in word_duration]
        elif self.word_duration_mode == "syllable_norm":
            return [duration / nb_syllables(word) if nb_syllables(word) > 0 else 0
                    for duration, (_, _, word) in zip(word_duration, word_lines)]
        else:
            raise ValueError(f"Unknown word_duration_mode: {self.word_duration_mode}")

    def _extract_feature_per_word(self, word_lines, feature, end_time):
        """Extract feature values for each word."""
        return [feature[sec_to_idx(float(start), end_time, len(feature)):
                        sec_to_idx(float(end), end_time, len(feature))]
                for start, end, _ in word_lines]

    def _extract_f0_per_word(self, word_lines, f0, phoneme_lines, end_time=0, verbose=False):
        """Extract F0 for each word, with optional stress localization."""
        cnt_not_found = 0
        f0_per_word = []
        for start, end, word in word_lines:
            start_idx = sec_to_idx(float(start), end_time, len(f0))
            end_idx = sec_to_idx(float(end), end_time, len(f0))

            if self.f0_stress_localizer == "celex":
                syllables = syllabify(word)
                stressed_syllable_idx = self.celex_manager.get_stress_index(word)
                stress_syllable_time = find_stress_syllable_start(
                    syllables, stressed_syllable_idx, phoneme_lines, float(start), float(end))

                if stress_syllable_time:
                    stress_syllable_idx = sec_to_idx(stress_syllable_time, end_time, len(f0))
                    new_start = max(start_idx, stress_syllable_idx - self.f0_window // 2)
                    new_end = min(end_idx, stress_syllable_idx + self.f0_window // 2)
                else:
                    cnt_not_found += 1
                    new_start, new_end = start_idx, end_idx
            else:
                new_start, new_end = start_idx, end_idx

            f0_per_word.append(signal.resample(f0[new_start:new_end], self.f0_resampling_length))

        return f0_per_word, cnt_not_found