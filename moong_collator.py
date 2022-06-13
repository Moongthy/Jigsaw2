import numpy as np
import math
from keras.preprocessing import text, sequence
import torch


class SimpleCollator(object):
    """
    train_collate = Collator(percentile=100)
    train_dataset = TextDataset(x_train, lengths, y_train_torch.numpy())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=Collator)
    """
    def __init__(self, test=False, percentile=100, max_len=220):
        self.test = test
        self.percentile = percentile
        self.max_len = max_len

    def __call__(self, batch):
        if self.test:
            texts, lens = zip(*batch)
        else:
            texts, lens, target = zip(*batch)

        lens = np.array(lens)
        # 배치 안의 문장 들의 최대 길이가 MAX_LEN 보다 작다면, 그 값으로 패딩
        batch_max_len = min(int(np.percentile(lens, self.percentile)), self.max_len)
        texts = torch.tensor(sequence.pad_sequences(texts, maxlen=batch_max_len), dtype=torch.long)

        if self.test: return texts

        return texts, torch.tensor(target, dtype=torch.float32)


class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset using sequence bucketing to pad each batch individually.

    Arguments:
        sequences (list): A list of variable length tokens (e. g. from keras tokenizer.texts_to_sequences)
        choose_length (function): A function which receives a numpy array of sequence lengths of one batch as input
                                  and returns the length this batch should be padded to.
        other_features (list, optional): A list of tensors with other features that should be fed to the NN alongside the sequences.
        labels (Tensor, optional): A tensor with labels for the samples.
        indices (np.array, optional): A numpy array consisting of indices to iterate over.
        shuffle (bool): Whether to shuffle the dataset or not.  Default false.
        batch_size (int): Batch size of the samples. Default 512.

    implements:
        train_dataset = SequenceDataset(x_train, lambda lengths: lengths.max(), other_features=[lengths], shuffle=False, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    """

    def __init__(self, sequences, choose_length, other_features=None, labels=None,
                 indices=None, shuffle=False, batch_size=512):

        super(SequenceDataset, self).__init__()

        self.sequences = np.array(sequences)
        self.lengths = np.array([len(x) for x in sequences])
        self.n_samples = len(sequences)
        self.choose_length = choose_length
        self.other_features = other_features
        self.labels = labels

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(sequences))

        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            self._shuffle()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def _shuffle(self):
        self.indices = np.random.permutation(self.indices)

    def __getitem__(self, i):
        idx = self.indices[(self.batch_size * i):(self.batch_size * (i + 1))]

        if self.shuffle and i == len(self) - 1:
            self._shuffle()

        pad_length = math.ceil(self.choose_length(self.lengths[idx]))
        padded_sequences = sequence.pad_sequences(self.sequences[idx], maxlen=pad_length)

        x_batch = [torch.tensor(padded_sequences, dtype=torch.long)]

        if self.other_features is not None:
            x_batch += [x[idx] for x in self.other_features]

        if self.labels is not None:
            out = x_batch, self.labels[idx]
        else:
            out = x_batch

        return out


class SequenceBucketCollator():
    """
    implements:
        batch_size = BATCH_SIZE

        test_dataset = data.TensorDataset(x_test_padded, test_lengths)
        train_dataset = data.TensorDataset(x_train_padded, lengths, y_train_torch)
        valid_dataset = data.Subset(train_dataset, indices=[0, 1])

        train_collator = SequenceBucketCollator(torch.max,
                                                sequence_index=0,
                                                length_index=1,
                                                label_index=2,
                                                maxlen=maxlen)
        test_collator = SequenceBucketCollator(torch.max,
                                               sequence_index=0,
                                               length_index=1,
                                               maxlen=maxlen)

        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
        valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collator)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collator)

        databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader, collate_fn=train_collator)

        def custom_loss(data, targets):
            ''' Define custom loss function for weighted BCE on 'target' column '''
            bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
            bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
            return (bce_loss_1 * loss_weight) + bce_loss_2

            all_test_preds = []

        for model_idx in range(NUM_MODELS):
            print('Model ', model_idx)
            seed_everything(1234 + model_idx)
            model = NeuralNet(LSTM_UNITS, DENSE_HIDDEN_UNITS, embedding_matrix, y_aux_train.shape[-1])
            learn = Learner(databunch, model, loss_func=custom_loss)
            test_preds = train_model(learn,test_dataset,output_dim=7)
            all_test_preds.append(test_preds)
    """
    def __init__(self, choose_length, sequence_index, length_index, label_index=None, maxlen=220):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index
        self.maxlen = maxlen

    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]

        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]

        length = self.choose_length(lengths)
        mask = torch.arange(start=self.maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]

        batch[self.sequence_index] = padded_sequences

        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]

        return batch