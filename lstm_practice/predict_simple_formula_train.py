import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

class PredictSimpleFormulaNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_first, dropout):
        super(PredictSimpleFormulaNet, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = batch_first,
                            dropout = dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden=None):
        output, (hidden, cell) = self.rnn(inputs, hidden)
        output = self.output_layer(output[:, -1, :])

        return output

class Train():
    def __init__(self):
        input_size = 1
        output_size = 1
        hidden_size = 64
        num_layers = 1
        batch_first = True
        dropout = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device：", self.device)
        self.net = PredictSimpleFormulaNet(input_size, output_size, hidden_size, num_layers, batch_first, dropout).to(self.device)

    def make_dataset(self, dataset_num, input_length, t_start):
        dataset_inputs = []
        dataset_labels = []
        for t in range(dataset_num):
            dataset_inputs.append([np.exp(t_start + t + i) for i in range(input_length)])
            dataset_labels.append([np.exp(t_start + t + input_length)])

        return np.array(dataset_inputs), np.array(dataset_labels)

if __name__ == '__main__':
    '''
    定数
    '''
    dataset_num = 100
    input_length = 50
    t_start = -100.0
    '''
    学習用のデータセットを用意
    '''
    train = Train()
    dataset_inputs, dataset_labels = train.make_dataset(dataset_num, input_length, t_start)
    print("dataset_inputs = {}, dataset_labels = {}".format(dataset_inputs.shape, dataset_labels.shape))
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(dataset_inputs, dataset_labels, test_size=0.2, shuffle=False)
    print("train_inputs = {}, train_labels = {}, test_inputs = {}, test_labels = {}".format(train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape))

