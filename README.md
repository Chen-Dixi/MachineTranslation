# MachineTranslation
没什么技术含量，按照官方tutorial写的

一个词一个词的进去
RNN的维度我自己不好搞清楚,代码里添加了一些维度注释
```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
```
## nn.GRU
#### Parameters:	
>
>- input_size – The number of expected features in the input x
>- hidden_size – The number of features in the hidden state h
>- num_layers – Number of recurrent layers. E.g., setting num_layers=2 would - mean stacking two GRUs together to form a stacked GRU, with the second - GRU taking in outputs of the first GRU and computing the final results. - Default: 1
>- bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
>- batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
>- dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
>- bidirectional – If True, becomes a bidirectional GRU. Default: False

编码器前向传播过程中，经过embedding后有一个矩阵变换操作`embedded = self.embedding(input).view(1, 1, -1)`，这里刚开始有点懵。弄清楚GRU的前向传播过程
首先，GRU继承RNNBase(Module)，下面是RNNBase里面的前向传播过程
```python
    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

    ......
    ......省略
```

第二个函数参数`hx`就是代表第`x`时刻的隐藏层h_x或理解为上一个时刻的隐藏层$h^{(t-1)}$，
PYTORCH文档中:
> - **input** of shape (*seq_len, batch, input_size*): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See `torch.nn.utils.rnn.pack_padded_sequence()` for details.
> - **h_0** of shape (*num_layers * num_directions, batch, hidden_size*): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1.

总结，`self.embedding(input).view(1, 1, -1)`得到的tensor是shape表示序列长度为1，batch为1，后面的`-1`得到的就是词向量的维度，即`nn.Embedding(input_size, hidden_size)`中的`hidden_size`。每次forward就给一个词，所以view(1,1,-1)过后的shape就是（1,1,hidden_size）。

