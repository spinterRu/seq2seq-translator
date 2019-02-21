import torch.nn.functional as F
from torch import nn


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(output_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, 128)
        self.out = nn.Linear(128, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        hidden_activations = F.relu(self.hidden(output[0]))
        output = self.softmax(self.out(hidden_activations))
        return output, hidden

    # def initHidden(self, device):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)
