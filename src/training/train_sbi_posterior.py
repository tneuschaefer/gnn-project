import torch
import itertools
from sbi.utils import BoxUniform
from sbi.inference import SNPE
from models.sbi_simulator import Simulator
from models.sbi_posterior import SBIPosterior


def train_sbi_posterior(
    model_forward,
    data_loader,
    num_simulations=5000,
    density_estimator="maf",
    device=torch.device("cpu"),
):
    """Train an SBI posterior estimator using the forward model."""
    model_forward.eval()

    # Create simulator from the forward model
    simulator = Simulator(model_forward).to(device)

    source_batch, _ = next(iter(data_loader))
    n_sources = source_batch.shape[1]

    # Define prior over source activations
    prior = BoxUniform(
        low=-2 * torch.ones(n_sources, device=device),
        high=2 * torch.ones(n_sources, device=device),
    )

    # Set logging_level to "WARNING" to suppress SBI log file spam
    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
        show_progress_bars=True,
    )

    # Generate training data
    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    inference = inference.append_simulations(theta, x)
    inference.train()

    posterior = inference.build_posterior()

    # Wrap the posterior in an SBIPosterior object
    sbi_posterior = SBIPosterior(posterior).to(device)

    return sbi_posterior


def optimize_sbi_hyperparams(
    model_forward, train_loader, test_loader, device=torch.device("cpu")
):
    """Optimize hyperparameters for SBI posterior estimation using grid search"""
    model_forward.eval()

    best_params = None
    best_loss = float("inf")

    # Define parameter grid
    param_grid = list(
        itertools.product([5000, 10000], ["maf", "nsf"])
    )  # num_simulations, density_estimator

    for num_simulations, density_estimator in param_grid:
        posterior_model = train_sbi_posterior(
            model_forward=model_forward,
            data_loader=train_loader,
            num_simulations=num_simulations,
            density_estimator=density_estimator,
            device=device,
        )

        posterior_model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for sources, eeg in test_loader:
                sources = sources.to(device)
                eeg = eeg.to(device)
                predicted = posterior_model.infer_sources(eeg, device=device)
                loss = torch.mean((sources - predicted) ** 2).item()
                total_loss += loss

        avg_loss = total_loss / len(test_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = (num_simulations, density_estimator)

    print(
        f"Optimized parameters: num_simulations: {best_params[0]}, density_estimator: {best_params[1]} with loss: {best_loss:.4f}"
    )
    return best_params
