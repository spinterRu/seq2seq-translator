import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = int(hidden_size / 2)

        self.gru = nn.GRU(input_size, self.hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(2, 1, self.hidden_size, device=device)