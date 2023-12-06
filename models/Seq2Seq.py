import random

import torch
from torch import nn


# paper name: Sequence to Sequence Learning with Neural Networks
# Neural Information Processing Systems (NIPS)
# time: 2014-09-10
# paper reference: https://arxiv.org/pdf/1409.3215.pdf

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # input_dim = source vocab size
        # emb_dim = embedding dimension  amount to  feature
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim,
                           hidden_size=hid_dim,
                           num_layers=n_layers,
                           dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        # output_dim = target vocab size
        # emb_dim = embedding dimension  amount to  feature
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(input_size=emb_dim,
                           hidden_size=hid_dim,
                           num_layers=n_layers,
                           dropout=dropout)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg_last, hidden, cell):
        # trg_last = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        trg_last = trg_last.unsqueeze(0)  # the length of trg_lat is one
        # trg_last = [1, batch size]

        embedded = self.dropout(self.embedding(trg_last))  # emb  amount to  feature
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # outputs = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        output = output.squeeze(0)
        # output = [batch size, hid dim]
        predictions = self.fc(output)
        # predictions = [batch size, output dim]
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.name = "Seq2Seq"
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # sec = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        trg_last = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            predictions, hidden, cell = self.decoder(trg_last, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = predictions

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = predictions.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            trg_last = trg[t] if teacher_force else top1

        # outputs = [trg len, batch size, trg vocab size]
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def get_model(input_dim, output_dim, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    model.apply(init_weights)
    return model
