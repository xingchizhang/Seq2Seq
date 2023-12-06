import time

import spacy
import torch
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets import Multi30k
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

from models import (Seq2Seq,
                    Encoder_Decoder,
                    Attention,
                    Attention_PM,
                    Convolution,
                    Transformer)


def init_data(device, is_reverse=False, is_batch_first=False, use_pm=False):
    # load spacy models
    print('Loading spacy models...')
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    print('Done.')

    # create the tokenizer functions
    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings (tokens)
        """
        if is_reverse:
            return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
        else:
            return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings (tokens)
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # create the fields
    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=is_batch_first,
                include_lengths=use_pm)

    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=is_batch_first)

    # download and load the train, validation and test data
    print('Loading data...')
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(SRC, TRG))
    print('Done.')

    # build the vocabulary for the source and target languages
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    # define the batch size
    BATCH_SIZE = 128

    if use_pm:
        # create the iterators
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device)
        # all elements in a batch are sorted by their non-padded lengths in descending order
    else:
        # create the iterators
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            device=device)

    return SRC, TRG, train_iterator, valid_iterator, test_iterator, train_data, valid_data, test_data


def get_data(model_name, device):
    if model_name in ['Seq2Seq']:
        (SRC, TRG, train_iterator, valid_iterator, test_iterator,
         train_data, valid_data, test_data) = init_data(device, is_reverse=True)
    elif model_name in ['Encoder_Decoder', 'Attention']:
        (SRC, TRG, train_iterator, valid_iterator, test_iterator,
         train_data, valid_data, test_data) = init_data(device)
    elif model_name in ['Attention_PM']:
        (SRC, TRG, train_iterator, valid_iterator, test_iterator,
         train_data, valid_data, test_data) = init_data(device, use_pm=True)
    elif model_name in ['Convolution', 'Transformer']:
        (SRC, TRG, train_iterator, valid_iterator, test_iterator,
         train_data, valid_data, test_data) = init_data(device, is_batch_first=True)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    return SRC, TRG, train_iterator, valid_iterator, test_iterator, train_data, valid_data, test_data


def get_model(model_name, src_field, trg_field, device):
    INPUT_DIM = len(src_field.vocab)
    OUTPUT_DIM = len(trg_field.vocab)

    if model_name == 'Seq2Seq':
        model = Seq2Seq.get_model(INPUT_DIM, OUTPUT_DIM, device)
    elif model_name == 'Encoder_Decoder':
        model = Encoder_Decoder.get_model(INPUT_DIM, OUTPUT_DIM, device)
    elif model_name == 'Attention':
        model = Attention.get_model(INPUT_DIM, OUTPUT_DIM, device)
    elif model_name == 'Attention_PM':
        src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
        model = Attention_PM.get_model(INPUT_DIM, OUTPUT_DIM, src_pad_idx, device)
    elif model_name == 'Convolution':
        trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]
        model = Convolution.get_model(INPUT_DIM, OUTPUT_DIM, trg_pad_idx, device)
    elif model_name == 'Transformer':
        src_pad_idx = src_field.vocab.stoi[src_field.pad_token]
        trg_pad_idx = trg_field.vocab.stoi[trg_field.pad_token]
        model = Transformer.get_model(INPUT_DIM, OUTPUT_DIM, src_pad_idx, trg_pad_idx, device)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip, epoch):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(tqdm(iterator, desc=f'Epoch_{epoch + 1:02}', unit='batch')):
        if model.name == 'Attention_PM':
            src, src_len = batch.src
        else:
            src = batch.src
            src_len = None
        trg = batch.trg
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # if model.name in ['Convolution', 'Transformer']:
        #     src = [batch size, src len]
        #     trg = [batch size, trg len]

        optimizer.zero_grad()

        if model.name in ['Convolution', 'Transformer']:
            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]
        else:
            if model.name == 'Attention_PM':
                output = model(src, src_len, trg)
            else:
                output = model(src, trg)
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]  # trg vocab size

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            # output = [(trg len - 1) * batch size, output dim]
            # trg = [(trg len - 1) * batch size]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if model.name == 'Attention_PM':
                src, src_len = batch.src
            else:
                src = batch.src
            trg = batch.trg
            # src = [src len, batch size]
            # trg = [trg len, batch size]
            # if model_name in ['Convolution', 'Transformer']:
            #     src = [batch size, src len]
            #     trg = [batch size, trg len]

            if model.name in ['Convolution', 'Transformer']:
                output, _ = model(src, trg[:, :-1])
                # output = [batch size, trg len - 1, output dim]

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                # output = [batch size * trg len - 1, output dim]
                # trg = [batch size * trg len - 1]
            else:
                if model.name == 'Attention_PM':
                    output = model(src, src_len, trg, teacher_forcing_ratio=0)
                else:
                    output = model(src, trg, teacher_forcing_ratio=0)  # turn off teacher forcing
                # output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]

                trg = trg[1:].view(-1)
                output = output[1:].view(-1, output_dim)
                # trg = [(trg len - 1) * batch size]
                # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def translate_sentence(model, sentence, src_field, trg_field, device, max_len=50):
    if model.name == 'Attention_PM':
        return Attention_PM.translate_sentence(sentence, src_field, trg_field, model, device, max_len)
    elif model.name == 'Convolution':
        return Convolution.translate_sentence(sentence, src_field, trg_field, model, device, max_len)
    elif model.name == 'Transformer':
        return Transformer.translate_sentence(sentence, src_field, trg_field, model, device, max_len)
    else:
        raise ValueError(f'Invalid model name: {model.name}')


def display_attention(model, sentence, translation, attention, title, n_heads=8, n_rows=4, n_cols=2):
    if model.name == 'Attention_PM':
        Attention_PM.display_attention(sentence, translation, attention, title)
    elif model.name == 'Convolution':
        Convolution.display_attention(sentence, translation, attention, title)
    elif model.name == 'Transformer':
        Transformer.display_attention(sentence, translation, attention, title, n_heads, n_rows, n_cols)
    else:
        raise ValueError(f'Invalid model name: {model.name}')


def calculate_bleu(data_set, test_iterator, src_field, trg_field, model, device, max_len=50):
    if model.name == 'Transformer':
        print('Enable acceleration!')
        _, _, bleu = Transformer.calculate_bleu_alt(test_iterator, src_field, trg_field, model, device, max_len)
        return bleu

    trgs = []
    pred_trgs = []

    time.sleep(1)
    for datum in tqdm(data_set, desc='Predicting'):
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(model, src, src_field, trg_field, device, max_len)

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])
    time.sleep(1)

    return bleu_score(pred_trgs, trgs)
