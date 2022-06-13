from keras.preprocessing import text, sequence
import torch
import numpy as np
from nltk import TweetTokenizer


def tokenize_v1(x_train, x_test, max_len):
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    x_train_sequences = tokenizer.texts_to_sequences(x_train)
    x_test_sequences = tokenizer.texts_to_sequences(x_test)
    x_train_pdd_sequences = sequence.pad_sequences(x_train_sequences, maxlen=max_len)
    x_test_pdd_sequences = sequence.pad_sequences(x_test_sequences, maxlen=max_len)

    return x_train_pdd_sequences, x_test_pdd_sequences, tokenizer.word_index


def tokenize_v2(x_train, x_test):
    """
    For SequenceBucketCollator
    """
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    x_train_sequences = tokenizer.texts_to_sequences(x_train)
    x_test_sequences = tokenizer.texts_to_sequences(x_test)

    lengths = torch.from_numpy(np.array([len(x) for x in x_train_sequences]))
    test_lengths = torch.from_numpy(np.array([len(x) for x in x_test_sequences]))

    maxlen = lengths.max()

    x_train_pdd_sequences = sequence.pad_sequences(x_train_sequences, maxlen=maxlen)
    x_test_pdd_sequences = sequence.pad_sequences(x_test_sequences, maxlen=maxlen)

    return x_train_pdd_sequences, x_test_pdd_sequences, lengths, test_lengths, maxlen, tokenizer.word_index


def tokenize_v3(x_train, x_test):
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

    train_word_sequences = []
    test_word_sequences = []
    word_dict = {}
    word_index = 1

    for doc in list(x_train):
        word_seq = []
        for token in tknzr.tokenize(doc):
            if token not in word_dict:
                word_dict[token] = word_index
                word_index += 1
            word_seq.append(word_dict[token])
        train_word_sequences.append(word_seq)

    for doc in list(x_test):
        word_seq = []
        for token in tknzr.tokenize(doc):
            if token not in word_dict:
                word_dict[token] = word_index
                word_index += 1
            word_seq.append(word_dict[token])
        test_word_sequences.append(word_seq)

    lengths = torch.from_numpy(np.array([len(x) for x in train_word_sequences]))
    test_lengths = torch.from_numpy(np.array([len(x) for x in test_word_sequences]))

    maxlen = lengths.max()

    x_train_pdd_sequences = sequence.pad_sequences(train_word_sequences, maxlen=maxlen)
    x_test_pdd_sequences = sequence.pad_sequences(test_word_sequences, maxlen=maxlen)

    return x_train_pdd_sequences, x_test_pdd_sequences, lengths, test_lengths, maxlen, word_dict