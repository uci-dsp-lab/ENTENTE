import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension, device):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)
    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float().reshape(dimension, -1).to(device))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float().to(device))

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    # print(self.w.weight.get_device(), self.w.bias.get_device())
    # print(t.get_device())
    output = torch.cos(self.w(t))


    return output
