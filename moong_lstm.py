import torch
from torch import nn
from torch.nn import functional as F


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class JigsawNeuralNetV1(nn.Module):
    """
    This is from Jigsaw2:https://www.kaggle.com/code/bminixhofer/simple-lstm-pytorch-version
    """

    def __init__(self, lstm_units, dense_hidden_units, embedding_matrix, num_aux_targets, max_features):
        super(JigsawNeuralNetV1, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)

        self.linear_out = nn.Linear(dense_hidden_units, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out


class JigsawNeuralNetV2(nn.Module):
    """
    For SequenceBucketCollator:
        The NN itself is also only slightly different.
        It also accepts an optional lengths parameter because lengths are part of the dataset now.
    """
    def __init__(self, lstm_units, dense_hidden_units, embedding_matrix, num_aux_targets):
        super(JigsawNeuralNetV2, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)

        self.linear_out = nn.Linear(dense_hidden_units, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units, num_aux_targets)

    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x.long())
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out


class JigsawNeuralNetV3(nn.Module):
    def __init__(self, lstm_units, dense_hidden_units, embedding_matrix, output_aux_sub=11):
        super(JigsawNeuralNetV3, self).__init__()
        embed_size = embedding_matrix.shape[1]

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)

        self.lstm1 = nn.LSTM(embed_size, lstm_units, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(dense_hidden_units, dense_hidden_units)
        self.linear2 = nn.Linear(dense_hidden_units, dense_hidden_units)

        self.linear_out = nn.Linear(dense_hidden_units + 6 + output_aux_sub, 1)
        self.linear_aux_out = nn.Linear(dense_hidden_units, 6)
        self.linear_sub_out = nn.Linear(dense_hidden_units, output_aux_sub)

    def forward(self, x, lengths=None):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        avg_pool1 = torch.mean(h_lstm1, 1)
        avg_pool2 = torch.mean(h_lstm2, 1)
        max_pool2, _ = torch.max(h_lstm2, 1)

        h_conc = torch.cat((avg_pool1, max_pool2, avg_pool2), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        aux_result = self.linear_aux_out(hidden)
        sub_result = self.linear_sub_out(hidden)
        result = self.linear_out(torch.cat((hidden, aux_result, sub_result), 1))
        # out = torch.cat([result, aux_result, sub_result], 1)
        out = torch.cat([result, aux_result], 1)
        return out