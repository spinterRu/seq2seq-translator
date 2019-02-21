import torch
import torch.nn.functional as F
from torch import nn


class SimpleAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(SimpleAttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(self.output_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, 128)
        self.out = nn.Linear(128, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.attn_combine = nn.Linear(self.output_size + self.hidden_size, self.output_size)

    def forward(self, input, hidden, attn):
        input_concat = torch.cat((input, attn.unsqueeze(0).unsqueeze(0)), 2)
        input_combined = F.relu(self.attn_combine(input_concat))
        output, hidden = self.gru(input_combined, hidden)
        hidden_activations = F.relu(self.hidden(output[0]))
        output = self.softmax(self.out(hidden_activations))
        return output, hidden

    # def initHidden(self, device):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)
