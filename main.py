from __future__ import unicode_literals, print_function, division


import string
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data import *
from net import *
from dixitool.pytorch.module import functional as dixiF
writer = SummaryWriter()

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


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
        encoder_outputs[ei] = encoder_output[0, 0]#这种索引方式是tensor特有的

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

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for idx in range(1, n_iters + 1):
        training_pair = training_pairs[idx - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if idx % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            #print('%s (%d %d%%) %.4f' % (timeSince(start, idx / n_iters), idx, idx / n_iters * 100, print_loss_avg))

        if idx % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            writer.add_scalar('train/train_loss',plot_loss_avg,idx)
    #showPlot(plot_losses)


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size,device).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,device, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
dixiF.save_model('checkpoints', '', encoder1)
dixiF.save_model('checkpoints', '', attn_decoder1)
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
