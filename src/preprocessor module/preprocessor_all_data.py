"""
preprocessor_all_data.py

This module loads raw EEG data (from MATLAB .mat files), performs enhanced preprocessing
(including baseline correction, bandpass and notch filtering, ICA-based artifact removal,
bad channel interpolation, downsampling, epoching, and normalization),
and saves the preprocessed data into a specified output directory.

Expected Directory structure:
    data/
      raw/
        Sub1/
          EEG/
            cnt.mat  or cnt*.mat (e.g., cnt1.mat, cnt2.mat, etc.)
            mrk.mat  or mrk*.mat (e.g., mrk1.mat, mrk2.mat, etc.)
      preprocessed/
        Sub1/
          preprocessed_epochs-epo.fif

Usage:
    Run this module directly to process all subject folders in the raw directory.
    Alternatively, import and call preprocess_subject() for custom processing.

Author: [Your Name]
Date: [Date]
"""

import os
import glob
import numpy as np
import mne
import scipy.io as sio
import h5py
from mne.preprocessing import ICA

# Global Preprocessing Parameters
NOTCH_FREQ = 50  # Hz
BANDPASS_LOW = 1.0  # Hz
BANDPASS_HIGH = 40.0  # Hz
DOWNSAMPLE_SFREQ = 200  # Hz
EPOCH_TMIN = 0.0  # seconds
EPOCH_TMAX = 5.0  # seconds

############################################
# Utility Functions for Loading .mat Files #
############################################


def load_mat_file(file_path):
    """
    Load a .mat file using scipy.io.loadmat; if that fails, load with h5py.
    """
    try:
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        print(f"Loaded {file_path} using scipy.io.loadmat")
        return mat
    except NotImplementedError:
        try:
            mat = h5py.File(file_path, mode="r")
            print(f"Loaded {file_path} using h5py")
            return mat
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")


def get_alternative_files(eeg_folder, base_name):
    """
    Find all files in eeg_folder starting with the prefix of base_name and ending with .mat.
    Returns a sorted list of file paths.
    """
    base_path = os.path.join(eeg_folder, base_name)
    if os.path.exists(base_path):
        return [base_path]
    prefix = base_name.split(".")[0]
    pattern = os.path.join(eeg_folder, f"{prefix}*.mat")
    file_list = sorted(glob.glob(pattern))
    if file_list:
        print(f"Using alternative files for {base_name}: {file_list}")
        return file_list
    else:
        raise FileNotFoundError(f"No files found for base {base_name} in {eeg_folder}.")


######################################
# Functions for Extracting Data      #
######################################


def extract_cnt_data(cnt_mat):
    """
    Extract EEG data, sampling frequency, and channel labels from cnt.mat.
    Expected keys in cnt_mat["cnt"]: 'EEG', 'fs', 'clab'
    """
    if isinstance(cnt_mat, dict):
        cnt_struct = cnt_mat["cnt"]
        eeg_data = np.array(
            cnt_struct.EEG if hasattr(cnt_struct, "EEG") else cnt_struct["EEG"]
        )
        fs = np.array(
            cnt_struct.fs if hasattr(cnt_struct, "fs") else cnt_struct["fs"]
        ).item()
        clab = np.array(
            cnt_struct.clab if hasattr(cnt_struct, "clab") else cnt_struct["clab"]
        )
        if clab.dtype.kind == "S":
            clab = [s.decode("utf-8") for s in clab]
        else:
            clab = list(clab)
    else:
        cnt_struct = cnt_mat["cnt"]
        eeg_data = np.array(cnt_struct["EEG"][()])
        fs = np.array(cnt_struct["fs"][()]).item()
        clab_refs = cnt_struct["clab"][()]
        decoded_labels = []
        for ref in clab_refs:
            if isinstance(ref, np.ndarray):
                ref = ref.item()
            if isinstance(ref, tuple):
                ref = ref[0]
            key = ref.decode("utf-8") if isinstance(ref, bytes) else ref
            try:
                label_ds = cnt_mat[key]
            except Exception:
                key = str(key)
                label_ds = cnt_mat[key]
            label_val = label_ds[()]
            if isinstance(label_val, bytes):
                decoded_labels.append(label_val.decode("utf-8"))
            elif isinstance(label_val, np.ndarray):
                try:
                    decoded_labels.append(
                        "".join(chr(int(x)) for x in label_val.flatten())
                    )
                except Exception:
                    decoded_labels.append(str(label_val))
            else:
                decoded_labels.append(str(label_val))
        clab = decoded_labels

    print("EEG data shape:", eeg_data.shape)
    print("Sampling frequency:", fs)
    print("Channel labels:", clab)
    return eeg_data, fs, clab


def safe_get(dataset, key):
    """
    Safely get an object from an h5py group given a key.
    If key is a tuple, use its first element.
    If key is bytes, decode it.
    """
    if isinstance(key, tuple):
        key = key[0]
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    try:
        obj = dataset[key]
        if isinstance(obj, tuple):
            obj = obj[0]
        return obj[()]
    except Exception as e:
        print(f"Error retrieving key '{key}': {e}")
        return None


def extract_mrk_info(mrk_mat):
    """
    Extract marker information from mrk.mat.
    If the file is loaded via h5py, then if the "event" key is a Group,
    retrieve the event codes from its "desc" subkey.
    Returns an events array of shape (n_events, 3): [time, 0, event code].
    """
    if isinstance(mrk_mat, dict):
        if "mrk" in mrk_mat:
            mrk_struct = mrk_mat["mrk"]
        else:
            keys = [
                k
                for k in mrk_mat.keys()
                if isinstance(k, str) and k.lower().startswith("mrk")
            ]
            if keys:
                mrk_struct = mrk_mat[keys[0]]
            else:
                raise ValueError("No valid marker key found in mrk.mat")
        try:
            mrk_time = np.array(
                mrk_struct.time if hasattr(mrk_struct, "time") else mrk_struct["time"]
            )
            try:
                mrk_event = np.array(
                    mrk_struct.event
                    if hasattr(mrk_struct, "event")
                    else mrk_struct["event"]
                )
            except Exception:
                mrk_event = None
            if mrk_event is None and "event" in mrk_struct:
                event_obj = mrk_struct["event"]
                if isinstance(event_obj, dict) and "desc" in event_obj:
                    mrk_event = np.array(event_obj["desc"])
            mrk_className = np.array(
                mrk_struct.className
                if hasattr(mrk_struct, "className")
                else mrk_struct["className"]
            )
        except Exception as e:
            raise ValueError(f"Error extracting marker data from dictionary: {e}")
    else:
        top_keys = list(mrk_mat.keys())
        if "mrk" in top_keys:
            mrk_group = mrk_mat["mrk"]
        else:
            keys = [
                k
                for k in top_keys
                if isinstance(k, str) and k.lower().startswith("mrk")
            ]
            if keys:
                mrk_group = mrk_mat[keys[0]]
            else:
                raise ValueError("No valid marker key found in mrk.mat")
        mrk_time = np.array(safe_get(mrk_group, "time"))
        # mrk_y = np.array(safe_get(mrk_group, "y"))
        if "event" in mrk_group:
            mrk_event_obj = mrk_group["event"]
            if isinstance(mrk_event_obj, h5py.Group):
                try:
                    mrk_event = np.array(mrk_event_obj["desc"][()])
                    print("Retrieved event data from 'desc' subkey in 'event' group.")
                except Exception as e:
                    print(f"Error retrieving 'desc' from 'event' group: {e}")
                    mrk_event = None
            else:
                mrk_event = np.array(safe_get(mrk_group, "event"))
        else:
            mrk_event = None
        mrk_className = np.array(safe_get(mrk_group, "className"))

    mrk_time = mrk_time.flatten()
    if mrk_event is None:
        raise ValueError("Could not retrieve marker 'event' data from mrk.mat.")
    mrk_event = mrk_event.flatten()
    if mrk_className is not None:
        if mrk_className.dtype.kind == "S":
            mrk_className = [s.decode("utf-8") for s in mrk_className]
        else:
            mrk_className = list(mrk_className)
    else:
        mrk_className = []

    print("Marker times shape:", mrk_time.shape)
    print("Marker event shape:", mrk_event.shape)
    print("Marker class names:", mrk_className)

    try:
        events = np.column_stack(
            (
                mrk_time.astype(int),
                np.zeros(len(mrk_time), dtype=int),
                mrk_event.astype(int),
            )
        )
    except Exception as e:
        raise ValueError(f"Error constructing events array: {e}")
    return events


def create_raw_object(eeg_data, fs, clab):
    """
    Create an MNE RawArray object from EEG data.
    Expects eeg_data shape: (n_samples, n_channels). Transposes if necessary.
    """
    n_channels = eeg_data.shape[1]
    if not clab or len(clab) != n_channels:
        clab = [f"EEG{i}" for i in range(n_channels)]
    if eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T
    info = mne.create_info(ch_names=clab, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data, info)
    print("Created MNE Raw object with shape:", raw.get_data().shape)
    return raw


######################################
# Enhanced Preprocessing Functions   #
######################################


def preprocess_raw(raw):
    """
    Enhanced preprocessing on raw data:
      - Notch filtering (to remove powerline interference)
      - Bandpass filtering (1-40 Hz)
      - Baseline correction (subtract mean)
      - ICA-based artifact removal
      - Bad channel detection and interpolation
      - Downsampling to DOWNSAMPLE_SFREQ
    Returns a cleaned Raw object.
    """
    raw.notch_filter(freqs=[NOTCH_FREQ], picks="eeg")
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, picks="eeg")
    raw.apply_function(lambda x: x - np.mean(x), picks="eeg")
    ica = ICA(n_components=20, random_state=42, max_iter="auto")
    ica.fit(raw)
    try:
        eog_indices, _ = ica.find_bads_eog(raw)
        print("Detected EOG-related ICA components:", eog_indices)
        ica.exclude = eog_indices
    except RuntimeError as e:
        if "No EOG channel(s) found" in str(e):
            print("No EOG channels found; skipping ICA-based EOG removal.")
        else:
            raise e
    raw_clean = ica.apply(raw.copy())
    try:
        bads = mne.preprocessing.find_bad_channels_maxwell(raw_clean)
    except Exception as e:
        print("Error in find_bad_channels_maxwell:", e)
        bads = []
    if bads:
        raw_clean.info["bads"] = bads
        raw_clean.interpolate_bads(reset_bads=True)
    raw_clean.resample(DOWNSAMPLE_SFREQ)
    print("Preprocessing complete. New sampling frequency:", raw_clean.info["sfreq"])
    return raw_clean


def epoch_raw(raw_clean, events, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX):
    """
    Create epochs from cleaned raw data using the events array.
    """
    if events is not None:
        epochs = mne.Epochs(
            raw_clean, events=events, tmin=tmin, tmax=tmax, baseline=None, preload=True
        )
        print("Created", len(epochs), "epochs.")
        return epochs
    else:
        print("No events provided; cannot epoch data.")
        return None


def normalize_epochs(epochs):
    """
    Normalize each epoch channel-wise (zero-mean, unit variance).
    Returns a new Epochs object with normalized data.
    """
    data = epochs.get_data()
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True)
    norm_data = (data - mean) / (std + 1e-8)
    epochs_normalized = mne.EpochsArray(
        norm_data, info=epochs.info, events=epochs.events
    )
    print("Normalization complete. Data shape:", epochs_normalized.get_data().shape)
    return epochs_normalized


######################################
# Saving and Processing Functions    #
######################################


def save_preprocessed_epochs(
    epochs, subject_folder, output_root="./data/preprocessed/"
):
    """
    Save preprocessed epochs in FIF format.
    """
    subject_output = os.path.join(output_root, subject_folder)
    os.makedirs(subject_output, exist_ok=True)
    output_file = os.path.join(subject_output, "preprocessed_epochs-epo.fif")
    epochs.save(output_file, overwrite=True)
    print("Saved preprocessed epochs to:", output_file)


def preprocess_subject(
    subject_folder, raw_root="./data/raw/", output_root="./data/preprocessed/"
):
    """
    Preprocess data for one subject.
    Loads MATLAB .mat files for EEG (cnt*.mat) and markers (mrk*.mat),
    concatenates data from multiple files (adjusting marker times for subsequent files),
    and performs enhanced preprocessing, epoching, normalization, and saving.
    """
    eeg_folder = os.path.join(raw_root, subject_folder, "EEG")

    try:
        cnt_files = get_alternative_files(eeg_folder, "cnt.mat")
    except FileNotFoundError as e:
        print(f"Error in subject {subject_folder}: {e}")
        return
    try:
        mrk_files = get_alternative_files(eeg_folder, "mrk.mat")
    except FileNotFoundError as e:
        print(f"Error in subject {subject_folder}: {e}")
        return

    # Load and combine cnt files
    eeg_data_list = []
    fs_list = []
    clab = None
    for cnt_file in cnt_files:
        cnt_mat = load_mat_file(cnt_file)
        eeg_data, fs, clab_local = extract_cnt_data(cnt_mat)
        eeg_data_list.append(eeg_data)
        fs_list.append(fs)
        if clab is None:
            clab = clab_local
        else:
            if clab != clab_local:
                print(f"Warning: Channel labels differ in {cnt_file}")
    if not all([f == fs_list[0] for f in fs_list]):
        print(
            f"Warning: Sampling frequencies differ in subject {subject_folder}. Using the first one."
        )
    fs = fs_list[0]
    eeg_data_combined = np.concatenate(eeg_data_list, axis=0)

    # Load and combine mrk files
    mrk_list = []
    total_offset = 0
    for idx, mrk_file in enumerate(mrk_files):
        mrk_mat = load_mat_file(mrk_file)
        events = extract_mrk_info(mrk_mat)
        if idx > 0:
            events[:, 0] += total_offset
        mrk_list.append(events)
        total_offset += eeg_data_list[idx].shape[0]
    if len(mrk_list) == 1:
        combined_events = mrk_list[0]
    else:
        combined_events = np.concatenate(mrk_list, axis=0)

    # Create Raw object
    raw = create_raw_object(eeg_data_combined, fs, clab)

    # Enhanced preprocessing
    raw_clean = preprocess_raw(raw)

    # Epoching
    epochs = epoch_raw(raw_clean, combined_events)
    if epochs is None:
        print("Skipping subject", subject_folder, "due to missing events.")
        return

    # Normalization
    epochs_normalized = normalize_epochs(epochs)

    # Save preprocessed data
    save_preprocessed_epochs(epochs_normalized, subject_folder, output_root=output_root)


def process_all_subjects(raw_root="./data/raw/", output_root="./data/preprocessed/"):
    """
    Loop over all subject folders in the raw directory and preprocess each subject.
    """
    subjects = [
        d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))
    ]
    print("Found subjects:", subjects)
    for subj in subjects:
        try:
            print(f"\nProcessing subject: {subj}")
            preprocess_subject(subj, raw_root=raw_root, output_root=output_root)
        except Exception as e:
            print(f"Error processing {subj}: {e}")


if __name__ == "__main__":
    process_all_subjects()
