import random

import spacy
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# 'Attention_PM' means using Packed Padded Sequences and Masking

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        # input_dim = source vocab size
        # emb_dim = embedding dimension  amount to  feature
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=enc_hid_dim,
                          bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src len, batch size]
        # src_len = [batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        # need to explicitly put lengths on cpu!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))

        packed_outputs, hidden = self.rnn(packed_embedded)
        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs is now a non-packed sequence, all hidden states obtained when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        # this is actually a deviation from the paper.
        # Instead, they feed only the first backward RNN hidden state
        # we use a linear for compression

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.fc_attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.fc_v = nn.Linear(dec_hid_dim, 1, bias=False)  # without bias!

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch size, src len, dec hid dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.fc_attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]

        attention = self.fc_v(energy).squeeze(2)
        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(input_size=(enc_hid_dim * 2) + emb_dim,
                          hidden_size=dec_hid_dim)
        self.fc = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg_last, hidden, encoder_outputs, mask):
        # trg_last = [batch size]
        # hidden = [batch size, hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        trg_last = trg_last.unsqueeze(0)
        # trg_last = [1, batch size]

        embedded = self.dropout(self.embedding(trg_last))
        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        # a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        weighted = torch.bmm(a, encoder_outputs)  # weighted is equivalent to context
        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden

        output = torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1)
        # output = [batch size, dec hid dim + enc hid dim * 2 + emb dim]
        predictions = self.fc(output)
        # predictions = [batch size, output dim]
        return predictions, hidden.squeeze(0), a.squeeze(1)


class Attention_PM(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.name = 'Attention_PM'
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        # mask = [batch size, src len]
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        trg_last = trg[0, :]

        mask = self.create_mask(src)
        # mask = [batch size, src len]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states and mask
            # receive output tensor (predictions) and new hidden state
            predictions, hidden, _ = self.decoder(trg_last, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = predictions

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = predictions.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            trg_last = trg[t] if teacher_force else top1

        return outputs


def init_weights(m):  # simplified version of the weight initialization scheme used in paper
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def get_model(input_dim, output_dim, src_pad_idx, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    SRC_PAD_IDX = src_pad_idx

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Attention_PM(enc, dec, SRC_PAD_IDX, device).to(device)

    model.apply(init_weights)
    return model


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    # tokenize the source sentence if it has not been tokenized (is a string)
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    # numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # src_indexes = [src len]

    # convert it to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    # src_tensor = [src len, batch size = 1]

    # get the length of the source sentence and convert to a tensor
    src_len = torch.LongTensor([len(src_indexes)])
    # src_len = [batch size = 1]

    # feed the source sentence into the encoder
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
        # encoder_outputs = [src len, batch size = 1, enc hid dim * 2]
        # hidden = [batch size = 1, dec hid dim]

    # create the mask for the source sentence
    mask = model.create_mask(src_tensor)
    # mask = [batch size = 1, src len]

    # create a list to hold the output sentence, initialized with an <sos> token
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    # trg_indexes = [trg len (now = 1)]

    # create a tensor to hold the attention values
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    # attentions = [max len, batch size = 1, src len]

    # while we have not hit a maximum length
    for i in range(max_len):
        # get the input tensor, which should be either <sos> or the last predicted token
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        # trg_tensor = trg_last = [batch size = 1]

        # feed the input, all encoder outputs, hidden state and mask into the decoder
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
            # output = [batch size = 1, output dim]
            # hidden = [batch size = 1, dec hid dim]
            # attention = [batch size = 1, src len]

        # store attention values
        attentions[i] = attention

        # get the predicted next token
        pred_token = output.argmax(1).item()

        # add prediction to current output sentence prediction
        trg_indexes.append(pred_token)

        # break if the prediction was an <eos> token
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # convert the output sentence from indexes to tokens
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # return the output sentence (with the <sos> token removed) and the attention values over the sequence
    # trg_tokens has <eos> token at the end
    # attentions = [trg len - 1, batch size = 1, src len]
    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]


def display_attention(sentence, translation, attention, title='Attention Map'):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    x_ticks = ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
    y_ticks = translation

    # Set the font size of the scale label to 15
    ax.tick_params(labelsize=15)
    # 设置x轴刻度, 旋转45度
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticks(range(len(y_ticks)))
    ax.set_yticklabels(y_ticks)

    # 设置主刻度定位器，使得主刻度之间的间隔为1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_title(fontsize=17, label=f'{title} - Attention Map')
    plt.show()
    plt.close()
