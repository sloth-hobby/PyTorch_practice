import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

class SimpleFormula():
    def __init__(self, sin_a=2.0, cos_a = 2.0, sin_t=25.0, cos_t = 25.0):
        self.sin_a = sin_a
        self.cos_a = cos_a
        self.sin_t = sin_t
        self.cos_t = cos_t

    def sin(self, input):
        return self.sin_a * np.sin(2.0 * np.pi / self.sin_t * (input))

    def cos(self, input):
        return self.cos_a * np.cos(2.0 * np.pi / self.cos_t * (input))

class PredictSimpleFormulaNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_first):
        super(PredictSimpleFormulaNet, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            batch_first = batch_first)
        self.output_layer = nn.Linear(hidden_size, output_size)

        nn.init.xavier_normal_(self.rnn.weight_ih_l0)
        nn.init.orthogonal_(self.rnn.weight_hh_l0)

    def forward(self, inputs):
        h, _= self.rnn(inputs)
        output = self.output_layer(h[:, -1])

        return output

class Train():
    def __init__(self, input_size, output_size, hidden_size, batch_first, lr):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device：", self.device)
        self.net = PredictSimpleFormulaNet(input_size, output_size, hidden_size, batch_first).to(self.device)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)

    def set_formula_const_arg(self, sin_a, cos_a, sin_t, cos_t):
        self.f = SimpleFormula(sin_a, cos_a, sin_t, cos_t)

    def make_dataset(self, dataset_num, sequence_length, t_start, calc_mode="sin"):
        dataset_inputs = []
        dataset_labels = []
        dataset_times = []
        for t in range(dataset_num):
            if calc_mode == "sin":
                dataset_inputs.append([self.f.sin(t_start + t + i) for i in range(sequence_length)])
                dataset_labels.append([self.f.sin(t_start + t + sequence_length)])
            elif calc_mode == "cos":
                dataset_inputs.append([self.f.cos(t_start + t + i) for i in range(sequence_length)])
                dataset_labels.append([self.f.cos(t_start + t + sequence_length)])
            dataset_times.append(t_start + t + sequence_length)
        print("test = {}, {}, {}".format(np.array(dataset_inputs).shape,  np.array(dataset_labels).shape, np.array(dataset_times).shape))
        return np.array(dataset_inputs),  np.array(dataset_labels), np.array(dataset_times)

    def train_step(self, inputs, labels):
        inputs = torch.Tensor(inputs).to(self.device)
        labels = torch.Tensor(labels).to(self.device)
        self.net.train()
        preds = self.net(inputs)
        loss = self.criterion(preds, labels)
        self.optimizer.zero_grad()
        loss.backward()
        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
        nn.utils.clip_grad_value_(self.net.parameters(), clip_value=2.0)
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
            for batch in range(n_batches_train):
                start = batch * batch_size
                end = start + batch_size
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

    def pred_result_plt(self, test_inputs, test_labels, test_times, sequence_length, input_size):
        print('-------------')
        print("start predict test!!")
        self.net.eval()
        preds = []
        for i in range(len(test_inputs)):
            input = np.array(test_inputs[i]).reshape(-1, sequence_length, input_size)
            input = torch.Tensor(input).to(self.device)
            pred = self.net(input).data.cpu().numpy()
            preds.append(pred[0].tolist())
        preds = np.array(preds)
        test_labels = np.array(test_labels)
        pred_epss = test_labels - preds
        print("pred_epss = {}".format(pred_epss))
        #以下グラフ描画
        plt.plot(test_times, preds)
        plt.plot(test_times, test_labels, c='#00ff00')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend(['label', 'pred'])
        plt.title('compare label and pred')
        plt.show()

if __name__ == '__main__':
    '''
    定数
    '''
    dataset_num = 250
    sequence_length = 3
    t_start = -100.0
    sin_a = 2.0
    cos_a = 2.0
    sin_t = 25.0
    cos_t = 25.0
    calc_mode = "sin"
    # model pram
    input_size = 1
    output_size = 1
    hidden_size = 64
    batch_first = True
    # train pram
    lr = 0.001
    epochs = 15
    batch_size = 4
    test_size = 0.2
    '''
    学習用の関数を呼び出す
    '''
    train = Train(input_size, output_size, hidden_size, batch_first, lr)
    train.set_formula_const_arg(sin_a, cos_a, sin_t, cos_t)
    dataset_inputs, dataset_labels, dataset_times = train.make_dataset(dataset_num, sequence_length, t_start, calc_mode=calc_mode)
    print("dataset_inputs = {}, dataset_labels = {}".format(dataset_inputs.shape, dataset_labels.shape))
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(dataset_inputs, dataset_labels, test_size=test_size, shuffle=False)
    print("train_inputs = {}, train_labels = {}, test_inputs = {}, test_labels = {}".format(train_inputs.shape, train_labels.shape, test_inputs.shape, test_labels.shape))
    train.train(train_inputs, train_labels, test_inputs, test_labels, epochs, batch_size, sequence_length, input_size)
    train_times, test_times = train_test_split(dataset_times, test_size=test_size, shuffle=False)
    train.pred_result_plt(test_inputs, test_labels, test_times, sequence_length, input_size)

