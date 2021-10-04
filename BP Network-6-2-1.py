import pickle
import argparse
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

def nomorlize(data):
    y = (data.max() - data.min()) * (data - data.min()) / (data.max() - data.min()) + data.min()
    return y



def initial_weights():
    W21 =np.array([[-1, -0.014, 0.005, 0.142, -0.365, 0.850], [-0.455, 0.881, -0.504, 0.167, -1.096, 0.015]] )#%对输入层和隐藏层之间的权值进行初始赋值
    B21 = np.array([[1],[-1]] )  # 对输入层和隐藏层之间的偏置进行初始赋值
    W32 = np.array([[-0.1], [-0.1]]).T #  %对隐藏层和输出层之间的权值进行初始赋值
    B32 = np.array([-0.2])    #   %对隐藏层和输出层之间的偏置进行初始赋值
    return W21, B21, W32, B32

class ActivationFunction():
    # 参考博客实现：https://www.jiqizhixin.com/articles/2021-02-24-7
    def Sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    def Tanh (x):
        return (2.0 / (1.0 + np.exp(-2 * x)))-1
    def ReLu(x):
        # np.where(x>0,x,0)
        return np.maximum(0, x)

    import numpy as np
    def softmax(x):
        orig_shape = x.shape
        if len(x.shape) > 1:
            # Matrix
            exp_minmax = lambda x: np.exp(x - np.max(x))
            denom = lambda x: 1.0 / np.sum(x)
            x = np.apply_along_axis(exp_minmax, 1, x)
            denominator = np.apply_along_axis(denom, 1, x)

            if len(denominator.shape) == 1:
                denominator = denominator.reshape((denominator.shape[0], 1))

            x = x * denominator
        else:
            # Vector
            x_max = np.max(x)
            x = x - x_max
            numerator = np.exp(x)
            denominator = 1.0 / np.sum(numerator)
            x = numerator.dot(denominator)

        assert x.shape == orig_shape
        return x


class Network():
    def initial_weights(self):
        self.lastdW21 = 0
        self.lastdW32 = 0
        self.W21 = np.array([[-1, -0.014, 0.005, 0.142, -0.365, 0.850],
                     [-0.455, 0.881, -0.504, 0.167, -1.096, 0.015]])  # %对输入层和隐藏层之间的权值进行初始赋值
        self.B21 = np.array([[1], [-1]])  # 对输入层和隐藏层之间的偏置进行初始赋值
        self.W32 = np.array([[-0.1], [-0.1]]).T  # %对隐藏层和输出层之间的权值进行初始赋值
        self.B32 = np.array([-0.2])  # %对隐藏层和输出层之间的偏置进行初始赋值

    def forward(self, traininput):
        self.traindatanumber = traininput.shape[1]
        self.HideOut = logistic(np.dot(self.W21, traininput) + np.tile(self.B21, (1, self.traindatanumber)))
        trainoutput = np.dot(self.W32, self.HideOut) + np.tile(self.B32, (1, self.traindatanumber))

        return trainoutput

    def backward(self,trainoutput, trainstandoutput):
        error = trainstandoutput - trainoutput


        delta2 = error
        A = np.dot(self.W32.T, delta2)
        delta1 = A * (self.HideOut * (1 - self.HideOut))
        dW32 = np.dot(delta2, self.HideOut.T)
        dB32 = np.dot(delta2, np.ones([ self.traindatanumber, 1]))
        dW21 = np.dot(delta1, x_train.T)
        dB21 = np.dot(delta1, np.ones([ self.traindatanumber, 1]))
        self.W32 = self.W32 + LearningRate * dW32 + trainmomfactor * self.lastdW32
        self.lastdW32 = LearningRate * dW32 + trainmomfactor * self.lastdW32
        self.B32 = self.B32 + LearningRate * dB32
        self.W21 = self.W21 + LearningRate * dW21 + trainmomfactor * self.lastdW21
        self.lastdW21 = LearningRate * dW21 + trainmomfactor * self.lastdW21
        self.B21 = self.B21 + LearningRate * dB21
def args():
    # 参数化学习
    # https://docs.python.org/zh-cn/3/library/argparse.html
    parser = argparse.ArgumentParser(description="need to change to fit your own data")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--trainmomfactor", type=float, default=0.9, help='Momentum factor')
    parser.add_argument("--errorprecision", type=float, default=10000)
    parser.add_argument("--TrainDataPath", type=str, default=r'F:\研究生课程安排\MyBPnetwork(6-2-1)_matlab/traindata.xlsx')
    parser.add_argument("--TestDataPath", type=str, default=r'F:\研究生课程安排\MyBPnetwork(6-2-1)_matlab/testdata.xlsx')
    opt = parser.parse_args()
    return opt

def dataloader(DataPath):
    data = pd.read_excel(DataPath)
    # test_list = pd.read_excel(TestDataPath)
    ex_x = np.array(data.iloc[:, :6])
    ex_y = np.array(data.iloc[:, 6])
    # traindatanumber = extraininput.shape[0]
    x = ex_x.T
    y = ex_y
    return x, y

def savemodel(model, savepath):
    with open(savepath,'wb') as fw:
        pickle.dump(model,fw)

def loadmodel(path):
    with open(path,'rb') as fr:
        model = pickle.load(fr)
    return model

def test(datapath=None):
    if datapath != None:
        x_test, y_test = dataloader(datapath)
    else:
        testdata = [[-1, 5.5, 4, 4, 5.5, -1, 1],
                    [-3.2, 5, 1, 2.6, 5, -1, 0]]
        ex_x = np.array(testdata)[:, :6]
        ex_y = np.array(testdata)[:, 6]
        x_test = ex_x.T
        y_test = ex_y
    return x_test, y_test


if __name__ == '__main__':
    opt = args()
    TrainDataPath = opt.TrainDataPath
    TestDataPath = opt.TestDataPath
    writer = SummaryWriter('./log')
    x_train, y_train = dataloader(TrainDataPath)

    epoch = opt.epoch
    LearningRate = opt.lr
    errorprecision = opt.errorprecision
    trainmomfactor = opt.trainmomfactor

    logistic = ActivationFunction.Sigmoid
    model = Network()
    model.initial_weights()
    for i in range(epoch):
        trainoutput = model.forward(x_train)
        model.backward(trainoutput, y_train)
        # mae
        error = trainoutput- y_train
        energy = sum(error ** 2)
        writer.add_scalar('error', error[0][1], i)

    savemodel(model, 'model.pkl')
    x_test, y_test = test(TestDataPath)
    pred = model.forward(x_test)
    print(pred, y_test)
