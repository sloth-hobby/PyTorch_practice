import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

class PredictSimpleFormulaNet(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, dropout=0.5):
        super(PredictSimpleFormulaNet, self).__init__()
        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = dropout)
        self.output_layer = nn.Linear(hiddenDim, outputDim)

    def forward(self, inputs, hidden=None):
        output, (hidden, cell) = self.rnn(inputs, hidden)
        output = self.output_layer(output[:, -1, :])

        return output

class Train():
    def __init__(self):
        a = 0

    def makeDataSet(self, dataset_num, input_length, t_start):
        dataset_inputs = []
        dataset_labels = []
        for t in range(dataset_num):
            dataset_inputs.append([np.exp(t_start + t + i) for i in range(input_length)])
            dataset_labels.append([np.exp(t_start + t + input_length)])

        return dataset_inputs, dataset_labels

if __name__ == '__main__':
    '''
    定数
    '''
    dataset_num = 100
    input_length = 50
    t_start = -150.0
    '''
    学習
    '''
    train = Train()
    dataset_inputs, dataset_labels = train.makeDataSet(dataset_num, input_length, t_start)
    print("dataset_inputs = {}, dataset_labels = {}".format(len(dataset_inputs), len(dataset_labels)))
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(dataset_inputs, dataset_labels, test_size=0.2, shuffle=False)
    print("train_inputs = {}, train_labels = {}, test_inputs = {}, test_labels = {}".format(len(train_inputs), len(train_labels), len(test_inputs), len(test_labels)))

