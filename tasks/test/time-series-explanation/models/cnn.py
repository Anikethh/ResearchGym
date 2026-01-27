from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, d_inp, n_classes, dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(d_inp, out_channels=dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, n_classes),
        )
    def forward(self, x, mask=None, timesteps=None, get_embedding=False, captum_input=False, show_sizes=False, return_all=False):
        # Input x shape: (B, T, F)
        # Need to convert to: (B, F, T) for Conv1d
        x = x.transpose(1, 2)  # (B, F, T)

        if x.shape[-1] < 8:
            # pad sequence to at least 8 so two max pools don't fail
            # necessary for when WinIT uses a small window
            x = F.pad(x, (0, 8 - x.shape[-1]), mode="constant", value=0)
        if show_sizes:
            print(f"input {x.shape=}")
        embedding = self.encoder(x)  # (B, dim)
        if show_sizes:
            print(f"embedding {embedding.shape=}")
        out = self.mlp(embedding)  # (B, n_classes)
        if show_sizes:
            print(f"{out.shape=}")

        if get_embedding:
            return out, embedding
        else:
            return out