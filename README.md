# MachineTranslation
没任何技术含量，按照官方[tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)写的代码。这份代码的笔记也记录在我的博客里面:[Seq-to-Seq Translation](https://chen-dixi.github.io/2019/06/08/machineTranslation-tutorial/)

一个词一个词的进RNN
RNN的维度我自己不好搞清楚，在代码里添加了一些维度相关的注释，在main.py文件中
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
PyTorch文档GRU的`__call__`函数参数:
> - **input** of shape (*seq_len, batch, input_size*): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See `torch.nn.utils.rnn.pack_padded_sequence()` for details.
> - **h_0** of shape (*num_layers x num_directions, batch, hidden_size*): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1.

总结，`self.embedding(input).view(1, 1, -1)`得到tensor的shape表示序列长度为1，batch为1，后面的`-1`得到的就是词向量的维度，即`nn.Embedding(input_size, hidden_size)`中的`hidden_size`。每次forward就给一个词，所以view(1,1,-1)过后的shape是(1,1,hidden_size)，转换过后传递给GRU。

## train函数的维度细节
下面是官方教程训练步骤的代码，`train`函数分别从两种语言各自接收代表一句话的`tensor`，我把维度的分析说明加在下面的注释中。
```python
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    #参数input_tensor，target_tensor分别只有一句话，2维tensor，shape是（句子长度， 1）

    #encoder 的 0时刻的隐藏层
    encoder_hidden = encoder.initHidden()

    #梯度置零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #input ,target句子长度 
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #每个单词是RNN中的各个时刻输入，encoder_outputs存放各个时刻的输出
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device = device)

    loss = 0

    for ei in range(input_length):
        #input_tensor[ei]是一个长度为1的1维tensor ,shape是torch.Size([1])，经过embedding后得到的shape是(1,embedding_size)
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        #encoder_output的shape是 (seq_len, batch, input_size),和gru的输入 shape一样
        encoder_outputs[ei] = encoder_output[0, 0]#这种索引方式是tensor的索引方式

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
```

