import argparse
import os
import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils.device import get_device
from models.sbi_forward_model import ForwardModel
from models.sbi_posterior import SBIPosterior
from training.train_sbi_forward import train_forward_model, optimize_forward_hyper
from training.train_sbi_posterior import train_sbi_posterior, optimize_sbi_hyperparams


def load_eeg_data():
    """Load a sample EEG dataset from MNE's sample dataset

    Raises:
        Exception: MNE sample dataset not found

    Returns:
        dict: Dictionary containing sources and EEG data
    """

    try:
        data_path = mne.datasets.sample.data_path()
    except Exception as e:
        print(
            "MNE sample dataset not found. Please download it via mne.datasets.sample.data_path()"
        )
        raise e

    raw_fname = os.path.join(data_path, "MEG", "sample", "sample_audvis_raw.fif")
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick("eeg")
    raw.crop(0, 60)
    data = raw.get_data()  # shape: (n_channels, n_times)

    n_samples = data.shape[1] // 100  # 100 time points per sample
    sources = np.random.randn(n_samples, 3).astype(np.float32)

    # Average EEG data over 100 time point segments.
    eeg_samples = np.array(
        [data[:, i * 100 : (i + 1) * 100].mean(axis=1) for i in range(n_samples)],
        dtype=np.float32,
    )
    return {"sources": sources, "eeg": eeg_samples}


def dipole_localization_error(true_sources, predicted_sources):
    """Computes dipole localization error between true and predicted sources"""
    return np.linalg.norm(true_sources - predicted_sources, axis=1).mean()


def hamming_distance(true_labels, predicted_labels, threshold=0.5):
    """Computes Hamming distance between true and predicted binary labels"""
    true_bin = (true_labels > threshold).astype(int)
    pred_bin = (predicted_labels > threshold).astype(int)
    return np.mean(true_bin != pred_bin)


def evaluate_localization(true_sources, predicted_sources):
    """Evaluates source localization performance using two metrics"""
    dle = dipole_localization_error(true_sources, predicted_sources)
    hd = hamming_distance(true_sources, predicted_sources)
    return dle, hd


def main():
    parser = argparse.ArgumentParser(
        description="EEG Source Localization via Simulation-Based Inference (SBI)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train models instead of using saved ones",
    )
    parser.add_argument(
        "--optimize_params",
        action="store_true",
        help="Optimize hyperparameters before training",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save evaluation results to disk"
    )
    args = parser.parse_args()

    # torch.device which will be used throughout
    device = get_device()
    print(f"Using device: {device}")

    # Define paths
    root_dir = os.path.dirname(
        os.path.dirname(__file__)
    )  # root/src/source_localization.py
    model_dir = os.path.join(root_dir, "models")  # root/models
    os.makedirs(model_dir, exist_ok=True)

    # Load the MNE EEG training dataset
    data = load_eeg_data()
    true_sources = data["sources"]
    eeg_data = data["eeg"]

    # Get the number of electrodes and sources from the data
    n_electrodes = eeg_data.shape[1]
    n_sources = true_sources.shape[1]

    # Create torch data loader for training
    dataset = TensorDataset(
        torch.tensor(data["sources"], dtype=torch.float32),
        torch.tensor(data["eeg"], dtype=torch.float32),
    )
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    forward_model_path = os.path.join(model_dir, "sbi_forward_model.pth")
    forward_params = {"n_sources": n_sources, "n_electrodes": n_electrodes}

    # Train if argument is set or model does not exist
    if args.train or not os.path.exists(forward_model_path):
        print("Training forward model...")
        lr = 1e-3

        # Optimize hyperparameters if argument is set
        if args.optimize_params:
            print("Optimizing forward model hyperparameters...")
            # parameter grid for optimization can be adjusted in optimize_forward_hyper()
            hidden_sizes, dropout, lr = optimize_forward_hyper(
                data_loader, fixed_params=forward_params, device=device
            )
            forward_params.update({"hidden_sizes": hidden_sizes, "dropout": dropout})

        # Instantiate and train forward model
        print(f"Using hyperparemeters: {forward_params}")
        model_forward = ForwardModel(**forward_params).to(device)
        model_forward = train_forward_model(
            model_forward, data_loader, num_epochs=50, lr=lr, device=device
        )

        # Save trained model to disk, move to cpu to ensure it can be loaded without cuda/mps
        torch.save(model_forward.cpu().state_dict(), forward_model_path)
        # Move model back to the original device after saving
        model_forward = model_forward.to(device)
        print(f"Forward model saved to {forward_model_path}")
    else:
        # Load pretrained model
        print("Loading pretrained forward model...")
        model_forward = ForwardModel(**forward_params)
        model_forward.load_state_dict(
            torch.load(forward_model_path, map_location="cpu")
        )
        model_forward = model_forward.to(device)
        print("Loaded pretrained forward model.")

    posterior_path = os.path.join(model_dir, "sbi_posterior.pkl")

    # Train if argument is set or model does not exist
    if args.train or not os.path.exists(posterior_path):
        print("Training SBI posterior estimator...")

        sbi_params = {"num_simulations": 5000, "density_estimator": "maf"}

        # Optimize hyperparameters if argument is set
        if args.optimize_params:
            print("Optimizing SBI hyperparameters...")

            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # parameter grid for optimization can be adjusted in optimize_sbi_hyperparams()
            num_simulations, density_estimator = optimize_sbi_hyperparams(
                model_forward, train_loader, test_loader, device=device
            )
            sbi_params.update(
                {
                    "num_simulations": num_simulations,
                    "density_estimator": density_estimator,
                }
            )

        # Train the SBI posterior estimator
        posterior_model = train_sbi_posterior(
            model_forward=model_forward,
            data_loader=data_loader,
            device=device,
            **sbi_params,
        )

        # Unwarp the posterior model from the SBIPosterior wrapper
        posterior = posterior_model.posterior
        # Move to cpu to ensure it can be loaded without cuda/mps
        if hasattr(posterior, "to"):
            posterior = posterior.to("cpu")
        torch.save(posterior, posterior_path)
        print(f"SBI posterior estimator saved to {posterior_path}")
    else:
        # Load the pretrained model
        print("Loading pretrained SBI posterior estimator...")
        posterior = torch.load(posterior_path, map_location="cpu")
        posterior_model = SBIPosterior(posterior)
        posterior_model = posterior_model.to(device)
        print("Loaded pretrained SBI posterior estimator.")

    # Evaluate localization performance
    model_forward.eval()

    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).to(device)
        predicted_sources = (
            posterior_model.infer_sources(eeg_tensor, device=device).cpu().numpy()
        )

    dle, hd = evaluate_localization(true_sources, predicted_sources)
    result = f"Dipole Localization Error: {dle:.4f}\nHamming Distance: {hd:.4f}\n"

    if args.save_results:
        results_dir = os.path.join(root_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "source_localization_results.txt")
        with open(result_file, "w") as f:
            f.write(result)
        print(f"Results saved to {result_file}")
    else:
        print(result)

    return 0


if __name__ == "__main__":
    main()
