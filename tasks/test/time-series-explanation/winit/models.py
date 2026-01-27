import abc

import torch
from torch import nn


class TorchModel(nn.Module, abc.ABC):
    """
    Class extends torch.nn.Module. Mainly for user to specify the forward with a ``return_all``
    option. The model is supposed to accept inputs of shape (num_samples, num_features, num_times).
    If return_all is True, the output should be of shape (num_samples, num_states, num_times).
    Otherwise, the output should be of shape (num_samples, num_states)
    """

    def __init__(self, feature_size, num_states, hidden_size, device):
        """
        Constructor

        Args:
            feature_size:
               The number of features the model is accepting.
            num_states:
               The number of output nodes.
            hidden_size:
               The hidden size of the model
            device:
               The torch device the model is on.
        """
        super().__init__()
        self.feature_size = feature_size
        self.num_states = num_states
        self.hidden_size = hidden_size
        if self.num_states > 1:
            activation = torch.nn.Softmax(dim=1)
        else:
            activation = torch.nn.Sigmoid()
        self.activation = activation
        self.device = device

    @abc.abstractmethod
    def forward(self, input, return_all=True):
        """
        Specify the forward function for this torch.nn.Module. The forward function should not
        include the activation function at the end. i.e. the output should be in logit space.

        Args:
            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """

    def predict(self, input, return_all=True):
        """
        Apply the activation after the forward function.

            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """
        return self.activation(self.forward(input, return_all=return_all))


class ConvClassifier(TorchModel):
    def __init__(self, feature_size, num_states, hidden_size, kernel_size, device):
        super().__init__(feature_size, num_states, hidden_size, device)

        if kernel_size % 2 != 0:
            raise Exception("Odd kernel size")
        padding = kernel_size // 2

        # Input to torch Conv
        self.regressor = nn.Sequential(
            torch.nn.Conv1d(
                in_channels=feature_size,
                out_channels=self.hidden_size,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=self.num_states,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode="replicate",
            ),
        )

    def forward(self, input, return_all=True):
        """
        Specify the forward function for this torch.nn.Module. The forward function should not
        include the activation function at the end. i.e. the output should be in logit space.

        Args:
            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """
        num_samples, _, num_times = input.shape
        self.regressor.to(self.device)
        input = input.to(self.device)
        regressor = self.regressor(input)
        if return_all:
            return regressor[:, :, -num_times:]  # (num_samples, nstate, num_times)
        else:
            return regressor[:, :, -1]  # (num_samples, nstate)


class StateClassifier(TorchModel):
    def __init__(
        self, feature_size, num_states, hidden_size, device, rnn="GRU", num_layers=None, dropout=0.5
    ):
        super().__init__(feature_size, num_states, hidden_size, device)
        self.rnn_type = "GRU" if rnn is None else rnn
        self.num_layers = 1 if num_layers is None else num_layers

        # Input to torch LSTM should be of size (num_times, num_samples, num_features)
        self.regres_in_size = self.hidden_size

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(feature_size, self.hidden_size, num_layers=self.num_layers).to(
                self.device
            )
        else:
            self.rnn = nn.LSTM(feature_size, self.hidden_size, num_layers=self.num_layers).to(
                self.device
            )

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.regres_in_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.regres_in_size, self.num_states),
        )

    def forward(self, input, return_all=True, past_state=None):
        """
        Specify the forward function for this torch.nn.Module. The forward function should not
        include the activation function at the end. i.e. the output should be in logit space.

        Args:
            input:
                Shape = (num_samples, num_features, num_times)
            return_all:
                True if we want to get the output of the model only at the last timestep.
            past_state:
                The past state for the rnn. Shape = (num_layers, num_samples, hidden_size).
                Default to be None which will set this to zero.

        Returns:
            A tensor of shape (num_samples, num_states, num_times) if return_all is True. Otherwise,
            a tensor of shape (num_samples, num_states) is returned.
        """
        num_samples, num_features, num_times = input.shape
        input = input.permute(2, 0, 1).to(self.device)
        # input.shape (num_times, num_samples, num_features)
        self.rnn.to(self.device)
        self.regressor.to(self.device)
        if past_state is None:
            #  Size of hidden states: (num_layers, num_samples, hidden_size)
            past_state = torch.zeros([self.num_layers, input.shape[1], self.hidden_size]).to(
                self.device
            )
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(input, past_state)
            # all_encodings.shape = (num_times, num_samples, num_hidden)
            # encoding.shape = (num_layers, num_samples, num_hidden)
        else:
            all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))

        if return_all:
            reshaped_encodings = all_encodings.view(
                all_encodings.shape[1] * all_encodings.shape[0], -1
            )
            # shape = (num_times * num_samples, num_hidden)
            return torch.t(
                self.regressor(reshaped_encodings).view(all_encodings.shape[0], -1)
            ).reshape(num_samples, self.num_states, num_times)
            # (num_samples, n_state, num_times)
        else:
            encoding = encoding[-1, :, :]  # encoding.shape (num_samples, num_hidden)
            return self.regressor(encoding)  # (num_samples, num_states)
