import torch
from torch import nn


class SBIPosterior(nn.Module):
    """SBI posterior wrapper"""

    def __init__(self, posterior=None):
        super().__init__()
        self.posterior = posterior

    def infer_sources(self, eeg, num_samples=100, device=None):
        """
        Infer sources from EEG observations using the trained posterior

        Args:
            eeg: Observed EEG data
            num_samples: Number of posterior samples to draw per observation
            device: torch.device

        Returns:
            Tensor: Predicted source distributions
        """
        # If device is not provided, use the device of the first parameter or the input
        if device is None:
            param = next(self.parameters(), None)
            if param is not None:
                device = param.device
            else:
                device = eeg.device

        eeg = eeg.to(device)
        batch_size = eeg.shape[0]
        inferred = []

        for i in range(batch_size):
            samples = self.posterior.sample((num_samples,), x=eeg[i : i + 1])
            inferred.append(samples.mean(dim=0))

        return torch.stack(inferred)

    def to(self, device):
        """Move model to device"""
        super().to(device)
        if hasattr(self.posterior, "to"):
            self.posterior = self.posterior.to(device)
        return self
