import random

import torch
from torch import nn
import torch.nn.functional as F


# paper name: Neural Machine Translation by Jointly Learning to Align and Translate
# arXiv preprint arXiv:1409.0473
# time: 2014-09-01
# paper reference: https://arxiv.org/pdf/1409.0473.pdf

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

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)
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

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch size, src len, dec hid dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.fc_attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src len, dec hid dim]

        attention = self.fc_v(energy).squeeze(2)
        # attention = [batch size, src len]

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

    def forward(self, trg_last, hidden, encoder_outputs):
        # trg_last = [batch size]
        # hidden = [batch size, hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        trg_last = trg_last.unsqueeze(0)
        # trg_last = [1, batch size]

        embedded = self.dropout(self.embedding(trg_last))
        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
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
        return predictions, hidden.squeeze(0)


class Attention_Model(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.name = "Attention"
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
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
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        trg_last = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            predictions, hidden = self.decoder(trg_last, hidden, encoder_outputs)

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


def get_model(input_dim, output_dim, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Attention_Model(enc, dec, device).to(device)

    model.apply(init_weights)
    return model
