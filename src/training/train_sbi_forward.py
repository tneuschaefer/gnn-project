import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from models.sbi_forward_model import ForwardModel


def train_forward_model(
    model, data_loader, num_epochs=50, lr=1e-3, device=torch.device("cpu")
):
    """Train the forward model to map source activations to EEG data"""
    # Using MSE loss and Adam optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for sources, eeg in data_loader:
            sources = sources.to(device)
            eeg = eeg.to(device)
            optimizer.zero_grad()
            outputs = model(sources)
            loss = criterion(outputs, eeg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader):.4f}")
    return model


def optimize_forward_hyper(data_loader, fixed_params, device=torch.device("cpu")):
    """Optimize hyperparameters for the forward model using grid search"""

    best_params = None
    best_loss = float("inf")

    # Define parameter grid
    param_grid = list(
        itertools.product(([32, 64], [64, 128]), (0.1, 0.2), [1e-3, 1e-4])
    )  # hidden_sizes, dropout, lr

    for hidden_sizes, dropout, lr in param_grid:
        print(
            f"Training with hidden_sizes: {hidden_sizes}, dropout: {dropout}, learning rate: {lr}"
        )
        model = ForwardModel(
            **fixed_params, hidden_sizes=hidden_sizes, dropout=dropout
        ).to(device)
        trained_model = train_forward_model(
            model, data_loader, num_epochs=10, lr=lr, device=device
        )

        trained_model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for sources, eeg in data_loader:
                sources = sources.to(device)
                eeg = eeg.to(device)
                predicted = trained_model(sources)
                loss = criterion(predicted, eeg)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = (hidden_sizes, dropout, lr)

    print(
        f"Optimized parameters: hidden_sizes: {best_params[0]}, dropout: {best_params[1]}, learning rate: {best_params[2]} with loss: {best_loss:.4f}"
    )
    return best_params
