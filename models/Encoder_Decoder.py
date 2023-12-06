import random

import torch
from torch import nn


# paper name: Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
# arXiv preprint arXiv:1406.1078
# time: 2014-06-03
# paper reference: https://arxiv.org/pdf/1406.1078.pdf

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim

        # input_dim = source vocab size
        # emb_dim = embedding dimension  amount to  feature
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=hid_dim)  # no dropout as only one layer!
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)  # no cell state!
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        # output_dim = target vocab size
        # emb_dim = embedding dimension  amount to  feature
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(input_size=emb_dim + hid_dim,
                          hidden_size=hid_dim)
        self.fc = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg_last, hidden, context):
        # trg_last = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]

        trg_last = trg_last.unsqueeze(0)
        # trg_last = [1, batch size]

        embedded = self.dropout(self.embedding(trg_last))
        # embedded = [1, batch size, emb dim]

        rnn_input = torch.cat((embedded, context), dim=2)
        # emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(rnn_input, hidden)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), output.squeeze(0), context.squeeze(0)), dim=1)
        # output = [batch size, emb dim + hid dim * 2]
        predictions = self.fc(output)
        # predictions = [batch size, output dim]
        return predictions, hidden


class Encoder_Decoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.name = "Encoder_Decoder"
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"

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

        # last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        trg_last = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            predictions, hidden = self.decoder(trg_last, hidden, context)

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


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)


def get_model(input_dim, output_dim, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    model = Encoder_Decoder(enc, dec, device).to(device)

    model.apply(init_weights)
    return model
