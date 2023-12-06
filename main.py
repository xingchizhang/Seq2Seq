import math
import random
import time

import numpy as np
import torch
from torch import optim, nn

import utils

# region Set the random number seed
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# endregion

# This implementation utilized the code available at https://github.com/bentrevett/pytorch-seq2seq.
# Special thanks to Ben Trevett for his contributions!


def print_model(model_name, src_field, trg_field, device):
    model = utils.get_model(model_name, src_field, trg_field, device)
    print('\nmodels info:')
    print(model)
    print(f'The models has {utils.count_parameters(model):,} trainable parameters')


def train_model(model_name, train_iterator, valid_iterator, src_field, trg_field, device):
    model = utils.get_model(model_name, src_field, trg_field, device)

    if model_name in ["Transformer"]:
        LEARNING_RATE = 0.0005
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters())

    # ignore the loss whenever the target token is a padding token
    TRG_PAD_IDX = trg_field.vocab.stoi[trg_field.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    print('\nTraining...')
    for epoch in range(N_EPOCHS):
        time.sleep(1)  # to avoid printing errors on the same line
        start_time = time.time()
        train_loss = utils.train(model, train_iterator, optimizer, criterion, CLIP, epoch)
        valid_loss = utils.evaluate(model, valid_iterator, criterion)
        end_time = time.time()
        time.sleep(1)  # to avoid printing errors on the same line

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'.\\checkpoints\\{model_name}-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print()
    print('Finished Training')


def test_model(model_name, test_iterator, src_field, trg_field, device):
    model = utils.get_model(model_name, src_field, trg_field, device)

    # ignore the loss whenever the target token is a padding token
    TRG_PAD_IDX = trg_field.vocab.stoi[trg_field.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    model.load_state_dict(torch.load(f'.\\checkpoints\\{model_name}-model.pt'))
    test_loss = utils.evaluate(model, test_iterator, criterion)
    print(f'\n| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


def case_test(model_name, data_set, data_set_name, example_idx, src_field, trg_field, device):
    model = utils.get_model(model_name, src_field, trg_field, device)
    model.load_state_dict(torch.load(f'.\\checkpoints\\{model_name}-model.pt'))

    print(f'\nCase Test - {model_name} - {data_set_name} - {example_idx}')
    src = vars(data_set.examples[example_idx])['src']
    trg = vars(data_set.examples[example_idx])['trg']
    print(f'\tsrc = {src}')
    print(f'\ttrg = {trg}')

    translation, attention = utils.translate_sentence(model, src, src_field, trg_field, device)
    print(f'\tpredicted trg = {translation}')

    title = f'{model_name} - {data_set_name} - {example_idx}'
    utils.display_attention(model, src, translation, attention, title)


def print_bleu(model_name, test_data, test_iterator, src_field, trg_field, device):
    model = utils.get_model(model_name, src_field, trg_field, device)
    model.load_state_dict(torch.load(f'.\\checkpoints\\{model_name}-model.pt'))

    print('\nCalculating BLEU score...')
    bleu_score = utils.calculate_bleu(test_data, test_iterator, src_field, trg_field, model, device)
    print(f'BLEU score = {bleu_score * 100:.2f}')
    print('done.')


if __name__ == '__main__':
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 'Attention_PM' means using Packed Padded Sequences and Masking
    Model_name_list = ['Seq2Seq',           # 0
                       'Encoder_Decoder',   # 1
                       'Attention',         # 2
                       'Attention_PM',      # 3
                       'Convolution',       # 4
                       'Transformer']       # 5
    # Model_name__ = [5, 3, 2]

    Model_name = Model_name_list[5]

    (SRC, TRG, Train_iterator, Valid_iterator, Test_iterator,
     Train_data, Valid_data, Test_data) = utils.get_data(Model_name, Device)

    print(f'\nModel name: {Model_name}')

    print_model(Model_name, SRC, TRG, Device)

    train_model(Model_name, Train_iterator, Valid_iterator, SRC, TRG, Device)

    test_model(Model_name, Test_iterator, SRC, TRG, Device)

    case_test(Model_name, Train_data, 'Train_data', 12, SRC, TRG, Device)

    case_test(Model_name, Valid_data, 'Valid_data', 14, SRC, TRG, Device)

    case_test(Model_name, Test_data, 'Test_data', 16, SRC, TRG, Device)

    print_bleu(Model_name, Test_data, Test_iterator, SRC, TRG, Device)
