import torch
from data import *
from net import *
from dixitool.pytorch.module import functional as dixiF
import random
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
import numpy as np
def indexesFromSentence(lang, sentence):
    return [ lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1,1)

def tensorFromPair(pair):
    input_tensor = tensorFromSentence(input_lang,pair[0])
    target_tensor = tensorFromSentence(output_lang,pair[1])
    return (input_tensor, target_tensor)

#没有GPU的时候，用我的dixitool加载GPU格式的预训练模型到CPU上
if torch.cuda.is_available():
    encoder1 = encoder(args.model.encoder,pretrained=True,input_size=input_lang.n_words, hidden_size=args.model.encoder.hidden_size).to(device)
    attn_decoder = attnDecoder(args.model.decoder,
                                pretrained=True,
                                pretrained_embedding=False,
                                hidden_size=args.model.decoder.hidden_size,
                                output_size=output_lang.n_words).to(device)
else:
    encoder1 = encoder(args.model.encoder,pretrained=False,input_size=input_lang.n_words, hidden_size=args.model.encoder.hidden_size).to(device)
    attn_decoder = attnDecoder(args.model.decoder,
                                pretrained=False,
                                pretrained_embedding=False,
                                hidden_size=args.model.decoder.hidden_size,
                                output_size=output_lang.n_words).to(device)
    encoder1 = dixiF.load_model_cross_device(args.model.encoder.model_path,encoder1, save_location='gpu',load_location='cpu')
    attn_decoder = dixiF.load_model_cross_device(args.model.decoder.model_path,attn_decoder, save_location='gpu',load_location='cpu')

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1] #行是目标语言的单词，列是源语言单词

pair = random.choice(pairs)
print('>', pair[0])
print('=', pair[1])
output_words, attentions = evaluate(encoder1, attn_decoder, pair[0])
output_sentence = ' '.join(output_words)
print('<', output_sentence)
print('')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

showAttention(pair[0],output_words,attentions)

