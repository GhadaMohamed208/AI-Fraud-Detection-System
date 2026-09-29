import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=33):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32, 16),
            nn.LeakyReLU(),

            nn.Linear(16, 8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(),

            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_autoencoder(model_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model = Autoencoder(
        input_dim=checkpoint["input_dim"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    threshold = checkpoint["threshold"]

    return model, threshold, device


def calculate_anomaly_score(model, features, device):
    x = np.asarray(features, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(x)

        scores = torch.mean(
            (x - reconstructed) ** 2,
            dim=1
        )

    return scores.cpu().numpy()