"""
preprocessor.py

This module loads raw EEG data (from MATLAB .mat files), performs preprocessing
(including filtering, ICA-based artifact removal, downsampling, epoching, and normalization),
and saves the preprocessed data into a specified output directory.

Directory structure expected:
    data/
      raw/
        Sub1/
          EEG/
            cnt.mat
            mrk.mat
          ...
      preprocessed/
        Sub1/
          preprocessed_epochs-epo.fif
        ...

Usage:
    Run this module directly to process all subject folders in the raw directory.
    Alternatively, import and call preprocess_subject() for custom processing.
"""

import os
import numpy as np
import mne
import scipy.io as sio
import h5py
from mne.preprocessing import ICA

# Set global parameters
NOTCH_FREQ = 50  # Hz
BANDPASS_LOW = 1.0  # Hz
BANDPASS_HIGH = 40.0  # Hz
DOWNSAMPLE_SFREQ = 200  # Hz
EPOCH_TMIN = 0.0  # seconds
EPOCH_TMAX = 5.0  # seconds


def load_mat_file(file_path):
    """
    Try loading a .mat file using scipy.io.loadmat.
    If that fails (e.g., for MATLAB v7.3 files), load with h5py.
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


def extract_cnt_data(cnt_mat):
    """
    Extract EEG data, sampling frequency, and channel labels from cnt.mat.
    Expected keys in cnt_mat["cnt"]: 'EEG', 'fs', 'clab'
    """
    # Depending on how the file was loaded, accessing data differs:
    # if isinstance(cnt_mat, dict):
    #     cnt_struct = cnt_mat["cnt"]
    #     eeg_data = np.array(cnt_struct.EEG if hasattr(cnt_struct, 'EEG') else cnt_struct["EEG"])
    #     fs = np.array(cnt_struct.fs if hasattr(cnt_struct, 'fs') else cnt_struct["fs"]).item()
    #     clab = np.array(cnt_struct.clab if hasattr(cnt_struct, 'clab') else cnt_struct["clab"])
    # else:
    #     # h5py File object
    #     cnt_struct = cnt_mat["cnt"]
    #     eeg_data = np.array(cnt_struct["EEG"][()])
    #     fs = np.array(cnt_struct["fs"][()]).item()
    #     clab = np.array(cnt_struct["clab"][()])
    #
    # # Decode channel labels if stored as bytes
    # if clab.dtype.kind == 'S':
    #     clab = [s.decode('utf-8') for s in clab]
    # else:
    #     clab = list(clab)

    # Extract channel labels
    if isinstance(cnt_mat, dict):
        cnt_struct = cnt_mat["cnt"]
        eeg_data = np.array(cnt_struct.EEG if hasattr(cnt_struct, 'EEG') else cnt_struct["EEG"])
        fs = np.array(cnt_struct.fs if hasattr(cnt_struct, 'fs') else cnt_struct["fs"]).item()
        clab = np.array(cnt_struct.clab if hasattr(cnt_struct, 'clab') else cnt_struct["clab"])
        # If stored as bytes, decode them
        if clab.dtype.kind == 'S':
            clab = [s.decode('utf-8') for s in clab]
        else:
            clab = list(clab)
    else:
        # cnt_mat is an h5py.File object
        cnt_struct = cnt_mat["cnt"]
        eeg_data = np.array(cnt_struct["EEG"][()])
        fs = np.array(cnt_struct["fs"][()]).item()
        # Get the channel labels as references
        clab_refs = cnt_struct["clab"][()]
        decoded_labels = []
        for ref in clab_refs:
            # If ref is stored inside an array, get its item
            if isinstance(ref, np.ndarray):
                ref = ref.item()
            # Dereference the HDF5 object to get the dataset
            label_ds = cnt_mat[ref]
            label_val = label_ds[()]
            # If the returned value is bytes, decode it; if it's an array, convert to string.
            if isinstance(label_val, bytes):
                decoded_labels.append(label_val.decode('utf-8'))
            elif isinstance(label_val, np.ndarray):
                # If label_val is a char array, join its characters.
                # It might be stored as a numpy array of ASCII codes.
                try:
                    decoded_labels.append(''.join(chr(int(x)) for x in label_val.flatten()))
                except Exception:
                    decoded_labels.append(str(label_val))
            else:
                decoded_labels.append(str(label_val))
        clab = decoded_labels

    print("EEG data shape:", eeg_data.shape)
    print("Sampling frequency:", fs)
    print("Channel labels:", clab)
    return eeg_data, fs, clab


def extract_mrk_info(mrk_mat):
    """
    Extract marker information from mrk.mat.
    Expected keys in mrk_mat["mrk"]: 'time', 'y', 'event', 'className'
    """
    if isinstance(mrk_mat, dict):
        mrk_struct = mrk_mat["mrk"]
        mrk_time = np.array(mrk_struct.time if hasattr(mrk_struct, 'time') else mrk_struct["time"])
        mrk_y = np.array(mrk_struct.y if hasattr(mrk_struct, 'y') else mrk_struct["y"])
        # For event, check if it's a group or dataset
        event_field = mrk_struct.event if hasattr(mrk_struct, 'event') else mrk_struct["event"]
    else:
        mrk_struct = mrk_mat["mrk"]
        mrk_time = np.array(mrk_struct["time"][()])
        mrk_y = np.array(mrk_struct["y"][()])
        event_field = mrk_struct["event"]

    # Process event field (using h5py approach if needed)
    if isinstance(event_field, h5py.Dataset):
        mrk_event = np.array(event_field[()])
    elif isinstance(event_field, h5py.Group):
        event_keys = list(event_field.keys())
        print("Event group keys:", event_keys)
        # Choose the first key for data extraction
        mrk_event = np.array(event_field[event_keys[0]][()])
    else:
        mrk_event = None

    # Flatten the arrays
    mrk_time = mrk_time.flatten()
    if mrk_event is not None:
        mrk_event = mrk_event.flatten()

    # Extract className info if available
    if isinstance(mrk_mat, dict):
        mrk_className = np.array(mrk_struct.className if hasattr(mrk_struct, 'className') else mrk_struct["className"])
    else:
        mrk_className = np.array(mrk_struct["className"][()])

    if mrk_className.dtype.kind == 'S':
        mrk_className = [s.decode('utf-8') for s in mrk_className]
    else:
        mrk_className = list(mrk_className)

    print("Marker times shape:", mrk_time.shape)
    print("Marker y shape:", mrk_y.shape)
    if mrk_event is not None:
        print("Marker event shape:", mrk_event.shape)
    print("Marker class names:", mrk_className)

    # Construct the MNE events array: [sample, 0, event_code]
    if mrk_event is not None:
        events = np.column_stack((mrk_time.astype(int), np.zeros(len(mrk_time), dtype=int), mrk_event.astype(int)))
    else:
        events = None

    return events


def create_raw_object(eeg_data, fs, clab):
    """
    Create an MNE RawArray object from the EEG data.
    MNE expects data shape as (n_channels, n_times); if needed, transpose the data.
    """
    n_channels = eeg_data.shape[1]
    # Create channel names if clab is empty or not valid
    if not clab or len(clab) != n_channels:
        clab = [f"EEG{i}" for i in range(n_channels)]
    # Transpose if data is (n_samples, n_channels)
    if eeg_data.shape[0] > eeg_data.shape[1]:
        eeg_data = eeg_data.T
    info = mne.create_info(ch_names=clab, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    print("Created MNE Raw object:", raw)
    return raw


def preprocess_raw(raw):
    """
    Apply preprocessing to the raw data:
      - Notch filtering
      - Bandpass filtering
      - ICA artifact removal (with handling for missing EOG channels)
      - Downsampling
    Returns a cleaned raw object.
    """
    # Notch filter
    raw.notch_filter(freqs=[NOTCH_FREQ], picks='eeg')
    # Bandpass filter
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, picks='eeg')

    # ICA artifact removal
    ica = ICA(n_components=20, random_state=42)
    ica.fit(raw)
    try:
        eog_indices, scores = ica.find_bads_eog(raw)
        print("Blink components detected:", eog_indices)
        ica.exclude = eog_indices
    except RuntimeError as e:
        if "No EOG channel(s) found" in str(e):
            print("No EOG channels found; skipping EOG artifact removal.")
        else:
            raise e
    raw_clean = ica.apply(raw.copy())

    # Downsampling
    raw_clean.resample(DOWNSAMPLE_SFREQ)
    print("Downsampled raw data to:", raw_clean.info['sfreq'], "Hz")
    return raw_clean


def epoch_raw(raw_clean, events, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX):
    """
    Create epochs from the cleaned raw data using the events array.
    """
    if events is not None:
        epochs = mne.Epochs(raw_clean, events=events, tmin=tmin, tmax=tmax,
                            baseline=None, preload=True)
        print("Created", len(epochs), "epochs.")
        return epochs
    else:
        print("No events available; cannot epoch data.")
        return None


def normalize_epochs(epochs):
    """
    Normalize each epoch channel-wise (zero-mean, unit variance).
    Return a new Epochs object with normalized data.
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    for i in range(data.shape[0]):
        ch_means = np.mean(data[i], axis=1, keepdims=True)
        ch_stds = np.std(data[i], axis=1, keepdims=True)
        data[i] = (data[i] - ch_means) / (ch_stds + 1e-8)
    epochs_normalized = mne.EpochsArray(data, info=epochs.info, events=epochs.events)
    print("Normalization complete. Normalized epochs shape:", epochs_normalized.get_data().shape)
    return epochs_normalized


def save_preprocessed_epochs(epochs, subject_folder, output_root="./data/preprocessed/"):
    """
    Save the preprocessed epochs (e.g. normalized epochs) to the output directory.
    The file is saved in FIF format.
    """
    subject_output = os.path.join(output_root, subject_folder)
    os.makedirs(subject_output, exist_ok=True)
    output_file = os.path.join(subject_output, "preprocessed_epochs-epo.fif")
    epochs.save(output_file, overwrite=True)
    print("Saved preprocessed epochs to:", output_file)


def preprocess_subject(subject_folder, raw_root="./data/raw/", output_root="./data/preprocessed/"):
    """
    Preprocess the data for one subject given the subject folder name.
    """
    eeg_folder = os.path.join(raw_root, subject_folder, "EEG")
    cnt_file = os.path.join(eeg_folder, "cnt.mat")
    mrk_file = os.path.join(eeg_folder, "mrk.mat")

    # Load files
    cnt_mat = load_mat_file(cnt_file)
    mrk_mat = load_mat_file(mrk_file)

    # Extract data
    eeg_data, fs, clab = extract_cnt_data(cnt_mat)
    events = extract_mrk_info(mrk_mat)

    # Create Raw object
    raw = create_raw_object(eeg_data, fs, clab)

    # Preprocess raw data
    raw_clean = preprocess_raw(raw)

    # Epoching
    epochs = epoch_raw(raw_clean, events)
    if epochs is None:
        print("Skipping subject", subject_folder, "due to missing events.")
        return

    # Normalisation
    epochs_normalized = normalize_epochs(epochs)

    # Save preprocessed data
    save_preprocessed_epochs(epochs_normalized, subject_folder, output_root=output_root)


def process_all_subjects(raw_root="./../data/raw/", output_root="./../data/preprocessed/"):
    """
    Loop over all subject folders in the raw directory and preprocess them.
    """
    subjects = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
    print("Found subjects:", subjects)
    for subj in subjects:
        try:
            print("\nProcessing subject:", subj)
            preprocess_subject(subj, raw_root=raw_root, output_root=output_root)
        except Exception as e:
            print(f"Error processing {subj}: {e}")


if __name__ == "__main__":
    # When running as a script, process all subjects
    process_all_subjects()