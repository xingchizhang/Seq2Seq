import time

import spacy
import torch
from torch import nn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from torchtext.data.metrics import bleu_score
from tqdm import tqdm


# paper name: Attention is All You Need
# Neural Information Processing Systems (NIPS)
# time: 2017-06-12
# paper reference: https://arxiv.org/pdf/1706.03762.pdf

# The differences between the implementation and the paper are:
#   we use a learned positional encoding instead of a static one
#   we use the standard Adam optimizer with a static learning rate instead of one with warm-up and cool-down steps
#   we do not use label smoothing
# We make all of these changes as they closely follow BERT's set-up
#   and the majority of Transformer variants use a similar set-up.

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0, "hid_dim must be a multiple of n_heads"

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        # mask = [batch size, 1, query len or 1, key len]

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        # x = [batch size, n heads, query len, head dim]
        # This means that our attention vector will most probably not sum to 1,
        #   and we may pay full attention to a token but the attention over that token is set to 0 by dropout.
        # It seems strange, but it's not explained, or even mentioned in the paper.

        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        # attention = [batch size, n heads, query len, key len]
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(query=src, key=src, value=src, mask=src_mask)
        # _src = [batch size, query len = src len, hid dim]

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # _src = [batch size, src len, hid dim]

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]
        return src


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _trg, _ = self.self_attention(query=trg, key=trg, value=trg, mask=trg_mask)
        # _trg = [batch size, query len = trg len, hid dim]

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # _trg = [batch size, query len = trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # _trg = [batch size, trg len, hid dim]

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, query len = trg len, key len = src len]

        return trg, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [batch size, trg len, hid dim]

        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]

        return output, attention


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.name = "Transformer"
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_len = trg.shape[1]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention


def init_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def get_model(input_dim, output_dim, src_pad_idx, trg_pad_idx, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    SRC_PAD_IDX = src_pad_idx
    TRG_PAD_IDX = trg_pad_idx

    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
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
    # append the <sos> and <eos> tokens
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    # numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    # src_indexes = [src len]

    # convert it to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    # src_tensor = [batch size = 1, src len]

    # create the source sentence mask
    src_mask = model.make_src_mask(src_tensor)
    # src_mask = [batch size = 1, 1, 1, src len]

    # feed the source sentence and mask into the encoder
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
        # enc_src = [batch size = 1, src len, hid dim]

    # create a list to hold the output sentence, initialized with an <sos> token
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    # trg_indexes = [trg len (now = 1)]

    # while we have not hit a maximum length
    for i in range(max_len):
        # convert the current output sentence prediction into a tensor with a batch dimension
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        # trg_tensor = [batch size = 1, trg len (change)]

        # create a target sentence mask
        trg_mask = model.make_trg_mask(trg_tensor)
        # trg_mask = [batch size = 1, 1, trg len, trg len (change)]

        # place the current output, encoder output and both masks into the decoder
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            # output = [batch size = 1, trg len, output dim]
            # attention = [batch size = 1, n heads, trg len (change), src len]

        # get next output token prediction from decoder along with attention
        pred_token = output.argmax(2)[:, -1].item()

        # add prediction to current output sentence prediction
        trg_indexes.append(pred_token)

        # break if the prediction was an <eos> token
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # convert the output sentence from indexes to tokens
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # return the output sentence (with the <sos> token removed) and the attention from the last layer
    # trg_tokens has <eos> token at the end
    # attention = [batch size = 1, n heads, trg len - 1, src len]
    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention, title, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 35))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)

        x_ticks = ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
        y_ticks = translation

        ax.set_xticks(range(len(x_ticks)))
        ax.set_xticklabels(x_ticks, rotation=45)
        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels(y_ticks)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        ax.set_title(fontsize=14, label=f'{title} - heads_{i+1} - Attention Map')

    plt.show()
    plt.close()


# Calculate bleu quickly
def translate_sentence_vectorized(src_tensor, src_field, trg_field, model, device, max_len=50):
    assert isinstance(src_tensor, torch.Tensor)

    model.eval()
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    # enc_src = [batch_sz, src_len, hid_dim]

    trg_indexes = [[trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))]
    # Even though some examples might have been completed by producing a <eos> token
    # we still need to feed them through the model because other are not yet finished
    # and all examples act as a batch. Once every single sentence prediction encounters
    # <eos> token, then we can stop predicting.
    translations_done = [0] * len(src_tensor)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_tokens = output.argmax(2)[:,-1]
        for i, pred_token_i in enumerate(pred_tokens):
            trg_indexes[i].append(pred_token_i)
            if pred_token_i == trg_field.vocab.stoi[trg_field.eos_token]:
                translations_done[i] = 1
        if all(translations_done):
            break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
        pred_sentence = []
        for i in range(1, len(trg_sentence)):
            if trg_sentence[i] == trg_field.vocab.stoi[trg_field.eos_token]:
                break
            pred_sentence.append(trg_field.vocab.itos[trg_sentence[i]])
        pred_sentences.append(pred_sentence)

    return pred_sentences, attention


def calculate_bleu_alt(iterator, src_field, trg_field, model, device, max_len = 50):
    trgs = []
    pred_trgs = []
    with torch.no_grad():
        time.sleep(1)
        for batch in tqdm(iterator, desc='Predicting', unit='batch'):
            src = batch.src
            trg = batch.trg
            _trgs = []
            for sentence in trg:
                tmp = []
                # Start from the first token which skips the <start> token
                for i in sentence[1:]:
                    # Targets are padded. So stop appending as soon as a padding or eos token is encountered
                    if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:
                        break
                    tmp.append(trg_field.vocab.itos[i])
                _trgs.append([tmp])
            trgs += _trgs
            pred_trg, _ = translate_sentence_vectorized(src, src_field, trg_field, model, device)
            pred_trgs += pred_trg
        time.sleep(1)
    return pred_trgs, trgs, bleu_score(pred_trgs, trgs)
