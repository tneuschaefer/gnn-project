import torch


class Simulator:
    """Simulator class for SBI that wraps forward model"""

    def __init__(self, forward_model):
        self.forward_model = forward_model

    def __call__(self, sources_batch):
        """Run simulation from source space to EEG space"""
        # Get device from forward model
        device = next(self.forward_model.parameters()).device
        sources_batch = sources_batch.to(device)

        with torch.no_grad():
            return self.forward_model(sources_batch)

    def to(self, device):
        """Move model to device"""
        self.forward_model = self.forward_model.to(device)
        return self
