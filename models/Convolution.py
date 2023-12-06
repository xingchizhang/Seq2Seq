import spacy
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# paper name: Convolutional Sequence to Sequence Learning
# International conference on machine learning (ICML)
# time: 2017-05-08
# paper reference: https://arxiv.org/pdf/1705.03122.pdf

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, max_length=100):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"  # we restrict the kernel size to be odd

        self.max_length = max_length
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)  # the hid_dim is equivalent to the channel in computer vision
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        assert n_layers >= 1, "Encoder must have at least 1 layer!"
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]
        assert src_len <= self.max_length, "Source length exceeds maximum length!"
        # create position tensor
        pos = torch.arange(0, src_len)
        # pos = [0, 1, 2, 3, ..., src len - 1]
        pos = pos.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, src len]

        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        # tok_embedded = pos_embedded = [batch size, src len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        # embedded = [batch size, src len, emb dim]

        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        # conv_input = [batch size, src len, hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, src len]  # hid dim is equivalent to the channel in computer vision

        # begin convolutional blocks...
        conved = None
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            # conved = [batch size, 2 * hid dim, src len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, src len]

            # set conv_input to conved for next loop iteration
            conv_input = conved
        # ...end convolutional blocks

        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        # conved = [batch size, src len, emb dim]

        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        # combined = [batch size, src len, emb dim]

        return conved, combined


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, trg_pad_idx, device,
                 max_length=100):
        super().__init__()
        self.max_length = max_length
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        # attention layer is difference from decoder, so we need another linear
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        # conved_emb = [batch size, trg len, emb dim]

        combined = (conved_emb + embedded) * self.scale
        # combined = [batch size, trg len, emb dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        # energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)
        # attention = [batch size, trg len, src len]

        attended_encoding = torch.matmul(attention, encoder_combined)  # equivalent to context vector
        # attended_encoding = [batch size, trg len, emd dim]

        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)
        # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):
        # trg = [batch size, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        assert trg_len <= self.max_length, "Target length exceeds maximum length!"
        # create position tensor
        pos = torch.arange(0, trg_len)
        # pos = [0, 1, 2, 3, ..., trg len - 1]
        pos = pos.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, trg len]

        # embed tokens and positions
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        # embedded = [batch size, trg len, emb dim]

        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)
        # conv_input = [batch size, trg len, hid dim]
        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        # conv_input = [batch size, hid dim, trg len]  # hid dim is equivalent to the channel in computer vision

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        conved = None
        attention = None
        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = torch.zeros(batch_size,
                                  hid_dim,
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)
            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input)
            # conved = [batch size, 2 * hid dim, trg len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, trg len]

            # calculate attention
            attention, conved = self.calculate_attention(embedded,
                                                         conved,
                                                         encoder_conved,
                                                         encoder_combined)
            # attention = [batch size, trg len, src len]
            # conved = [batch size, hid dim, trg len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale
            # conved = [batch size, hid dim, trg len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        output = self.hid2emb(conved.permute(0, 2, 1))
        # output = [batch size, trg len, emb dim]
        predictions = self.fc_out(self.dropout(output))
        # predictions = [batch size, trg len, output dim]
        return predictions, attention


class Convolution(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.name = "Convolution"
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)
        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # outputs is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for each word in the trg sentence
        outputs, attention = self.decoder(trg, encoder_conved, encoder_combined)
        # outputs = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]
        return outputs, attention


# todo: init_weights function neet design


def get_model(input_dim, output_dim, trg_pad_idx, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    EMB_DIM = 256
    HID_DIM = 512  # each conv. layer has 2 * hid_dim filters, e.i. 1024
    ENC_LAYERS = 10  # number of conv. blocks in encoder
    DEC_LAYERS = 10  # number of conv. blocks in decoder
    ENC_KERNEL_SIZE = 3  # must be odd!
    DEC_KERNEL_SIZE = 3  # can be even or odd
    ENC_DROPOUT = 0.25
    DEC_DROPOUT = 0.25
    TRG_PAD_IDX = trg_pad_idx

    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

    model = Convolution(enc, dec).to(device)
    # todo: neet to init weights
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
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    # src_tensor = [batch size = 1, src len]

    # feed the source sentence into the encoder
    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(src_tensor)
        # encoder_conved = [batch size = 1, src len, emb dim]
        # encoder_combined = [batch size = 1, src len, emb dim]

    # create a list to hold the output sentence, initialized with an <sos> token
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    # trg_indexes = [trg len (now = 1)]

    # while we have not hit a maximum length
    for i in range(max_len):
        # convert the current output sentence prediction into a tensor with a batch dimension
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        # trg_tensor = [batch size = 1, trg len (change)]

        # place the current output and the two encoder outputs into the decoder
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)
            # output = [batch size = 1, trg len, output dim]
            # attention = [batch size = 1, trg len (change), src len]

        # get next output token prediction from decoder
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
    # attentions = [batch size = 1, trg len - 1, src len] --> [trg len - 1, batch size = 1, src len]
    return trg_tokens[1:], attention.permute(1, 0, 2)


def display_attention(sentence, translation, attention, title='Attention Map'):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    x_ticks = ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
    y_ticks = translation

    # Set the font size of the scale label to 15
    ax.tick_params(labelsize=15)

    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticks(range(len(y_ticks)))
    ax.set_yticklabels(y_ticks)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_title(fontsize=17, label=f'{title} - Attention Map')
    plt.show()
    plt.close()
