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

        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

    def forward(self, inputs):
        # output, (hidden, cell) = self.rnn(inputs)
        # output = self.output_layer(output[:, -1, :])
        output, _= self.rnn(inputs)
        output = self.output_layer(output[:, -1])

        return output

class Train():
    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_first, dropout):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device：", self.device)
        self.net = PredictSimpleFormulaNet(input_size, output_size, hidden_size, num_layers, batch_first, dropout).to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.net.parameters(),
                                        lr=0.001,
                                        betas=(0.9, 0.999), amsgrad=True)

    def make_dataset(self, dataset_num, sequence_length, t_start):
        dataset_inputs = []
        dataset_labels = []
        for t in range(dataset_num):
            dataset_inputs.append([np.sin(t_start + t + i) for i in range(sequence_length)])
            dataset_labels.append([np.sin(t_start + t + sequence_length)])

        # return np.array(dataset_inputs).reshape(-1, sequence_length, 1), np.array(dataset_labels).reshape(-1, 1)
        return np.array(dataset_inputs),  np.array(dataset_labels)

    def train_step(self, inputs, labels):
        # print("inputs = {}, labels = {}".format(inputs.shape, labels.shape))
        inputs = torch.Tensor(inputs).to(self.device)
        labels = torch.Tensor(labels).to(self.device)
        # print("tensor_ver: inputs = {}, labels = {}".format(inputs, labels))
        self.net.train()
        preds = self.net(inputs)
        loss = self.criterion(preds, labels)
        self.optimizer.zero_grad()
        loss.backward()
        # # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
        # nn.utils.clip_grad_value_(self.net.parameters(), clip_value=2.0)
        self.optimizer.step()

        return loss, preds

    def test_step(self, inputs, labels):
        inputs = torch.Tensor(inputs).to(self.device)
        labels = torch.Tensor(labels).to(self.device)
        self.net.eval()
        preds = self.net(inputs)
        loss = self.criterion(preds, labels)

        return loss, preds

    def train(self, train_inputs, train_labels, test_inputs, test_labels, epochs, batch_size, sequence_length, input_size):
        torch.backends.cudnn.benchmark = True   # ネットワークがある程度固定であれば、高速化させる

        n_batches_train = int(train_inputs.shape[0] / batch_size)
        n_batches_test = int(test_inputs.shape[0] / batch_size)
        for epoch in range(epochs):
            print('-------------')
            print('Epoch {}/{}'.format(epoch+1, epochs))
            print('-------------')
            train_loss = 0.
            test_loss = 0.
            train_inputs_shuffle, train_labels_shuffle = shuffle(train_inputs, train_labels)
            # print("train_inputs_shuffle = {}, train_labels_shuffle = {}".format(train_inputs_shuffle.shape, train_labels_shuffle.shape))
            for batch in range(n_batches_train):
                # print("batch = {}, n_batches_train = {}".format(batch, n_batches_train))
                start = batch * batch_size
                end = start + batch_size
                # print("start = {}, end = {}".format(start, end))
                # print("start = {}, end = {}".format(np.array(train_inputs_shuffle[start:end]).reshape(-1, sequence_length, input_size).shape, np.array(train_labels_shuffle[start:end]).reshape(-1, input_size).shape))
                # print("start = {}, end = {}".format(np.array(train_inputs_shuffle[start:end]).reshape(-1, sequence_length, input_size), np.array(train_labels_shuffle[start:end]).reshape(-1, input_size)))
                loss, _ = self.train_step(np.array(train_inputs_shuffle[start:end]).reshape(-1, sequence_length, input_size), np.array(train_labels_shuffle[start:end]).reshape(-1, input_size))
                train_loss += loss.item()

            for batch in range(n_batches_test):
                start = batch * batch_size
                end = start + batch_size
                loss, _ = self.test_step(np.array(test_inputs[start:end]).reshape(-1, sequence_length, input_size), np.array(test_labels[start:end]).reshape(-1, input_size))
                test_loss += loss.item()

            train_loss /= float(n_batches_train)
            test_loss /= float(n_batches_test)
            print('loss: {:.3}, test_loss: {:.3}'.format(train_loss, test_loss))

if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    '''
    定数
    '''
    dataset_num = 110
    sequence_length = 3
    t_start = -100.0
    # model pram
    input_size = 1
    output_size = 1
    hidden_size = 64
    num_layers = 1
    batch_first = True
    dropout = 0.0
    # train pram
    epochs = 10
    batch_size = 3
    '''
    学習用のデータセットを用意
    '''
    train = Train(input_size, output_size, hidden_size, num_layers, batch_first, dropout)
    dataset_inputs, dataset_labels = train.make_dataset(dataset_num, sequence_length, t_start)
    print("dataset_inputs = {}, dataset_labels = {}".format(dataset_inputs.shape, dataset_labels.shape))
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(dataset_inputs, dataset_labels, test_size=0.2, shuffle=False)
    print("train_inputs = {}, train_labels = {}, test_inputs = {}, test_labels = {}".format(train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape))
    # print("train_inputs = {}, train_labels = {}, test_inputs = {}, test_labels = {}".format(train_inputs, train_labels, test_inputs, test_labels))
    train.train(train_inputs, train_labels, test_inputs, test_labels, epochs, batch_size, sequence_length, input_size)

