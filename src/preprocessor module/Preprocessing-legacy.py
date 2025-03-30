import os
import h5py
from scipy.signal import butter, filtfilt


# Define preprocessing functions
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def preprocess_data(cnt_data, fs, lowcut=0.5, highcut=50.0):
    cnt_data_filtered = bandpass_filter(cnt_data, lowcut, highcut, fs)
    return cnt_data_filtered


# Path to the main directory containing the folders
main_directory = os.getenv(
    "MAIN_DIRECTORY",
    r"C:\Users\AaVerma\OneDrive - Heidelberg Materials\Desktop\Raw_\Raw Dataset",
)

# Ensure the main directory environment variable is set
if not main_directory:
    raise ValueError("The MAIN_DIRECTORY environment variable is not set.")
# Create preprocessed dataset folder if it doesn't exist
preprocessed_dataset_dir = os.path.join(main_directory, "preprocessed_dataset")
if not os.path.exists(preprocessed_dataset_dir):
    os.makedirs(preprocessed_dataset_dir)

# Iterate over each folders
for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)

    if os.path.isdir(folder_path):
        # Check if EEG subfolder exists
        eeg_folder_path = os.path.join(folder_path, "EEG")
        if os.path.isdir(eeg_folder_path):
            print(f"Processing folder: {folder}")

            # Iterate through the .mat files inside the EEG folder
            for mat_file in os.listdir(eeg_folder_path):
                if mat_file.endswith(".mat"):
                    mat_file_path = os.path.join(eeg_folder_path, mat_file)
                    print(f"Loading {mat_file_path}...")

                    try:
                        # Load the .mat file using h5py
                        mat = h5py.File(mat_file_path, "r")

                        if "cnt" in mat:
                            cnt_group = mat["cnt"]
                            print(
                                f"Available keys in 'cnt' group: {list(cnt_group.keys())}"
                            )
                            if "EEG" in cnt_group:
                                cnt_data = cnt_group["EEG"][:]
                                fs = (
                                    cnt_group.attrs["fs"]
                                    if "fs" in cnt_group.attrs
                                    else 1000
                                )
                                print(
                                    f"Preprocessing cnt data with shape {cnt_data.shape}..."
                                )
                                cnt_data_preprocessed = preprocess_data(cnt_data, fs)

                                # Move the preprocessed data to the new folder
                                new_cnt_file_path = os.path.join(
                                    preprocessed_dataset_dir, f"{folder}_cnt.mat"
                                )
                                with h5py.File(new_cnt_file_path, "w") as new_mat:
                                    new_mat.create_dataset(
                                        "cnt", data=cnt_data_preprocessed
                                    )
                                print(
                                    f"Saved preprocessed cnt data to {new_cnt_file_path}"
                                )
                            else:
                                print(
                                    f"Error: 'EEG' key not found in 'cnt' group in {mat_file_path}."
                                )

                        if "mrk" in mat:
                            mrk_group = mat["mrk"]
                            print(
                                f"Available keys in 'mrk' group: {list(mrk_group.keys())}"
                            )
                            try:
                                for key in mrk_group.keys():
                                    print(f"Inspecting key: {key}")
                                    item = mrk_group[key]
                                    print(f"Type of {key}: {type(item)}")
                                    if isinstance(item, h5py.Dataset):
                                        print(
                                            f"Dataset {key} found with shape {item.shape}"
                                        )
                                        data = item[:]
                                        print(f"Data from {key}: {data[:5]}")
                                    else:
                                        print(f"Group {key} found, not a dataset.")

                                new_mrk_file_path = os.path.join(
                                    preprocessed_dataset_dir, f"{folder}_mrk.mat"
                                )
                                with h5py.File(new_mrk_file_path, "w") as new_mat:
                                    for key in mrk_group.keys():
                                        item = mrk_group[key]
                                        if isinstance(item, h5py.Dataset):
                                            new_mat.create_dataset(key, data=item[:])
                                print(
                                    f"Saved processed mrk data to {new_mrk_file_path}"
                                )
                            except KeyError as e:
                                print(
                                    f"Error: Key {e} not found in 'mrk' group in {mat_file_path}."
                                )
                            except Exception as e:
                                print(
                                    f"Error processing 'mrk' group in {mat_file_path}: {e}"
                                )

                        mat.close()

                    except Exception as e:
                        print(f"Error loading or processing {mat_file_path}: {e}")


print("Preprocessing complete")
