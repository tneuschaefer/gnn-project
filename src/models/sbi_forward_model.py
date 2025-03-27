import torch.nn as nn


class ForwardModel(nn.Module):
    def __init__(
        self, n_sources=3, n_electrodes=64, hidden_sizes=[32, 64], dropout=0.1
    ):
        """
        Forward model mapping from source space to EEG space

        Args:
            n_sources: Number of source dimensions
            n_electrodes: Number of EEG electrodes
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(ForwardModel, self).__init__()
        layers = []
        in_features = n_sources

        for hs in hidden_sizes:
            layers.append(nn.Linear(in_features, hs))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hs

        layers.append(nn.Linear(in_features, n_electrodes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
