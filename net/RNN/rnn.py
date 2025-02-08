import torch
import torch.nn as nn

# 1.单向、单层RNN
single_rnn = nn.RNN(4, 3, 1, batch_first=True)
input = torch.randn(1, 2, 4)  # bs * sl * fs
output, h_n = single_rnn(input)

# 2. 双向、单层RNN
bidirectional_rnn = nn.RNN(4, 3, 1, batch_first=True, bidirectional=True)
bi_output, bi_h_n = bidirectional_rnn(input)
