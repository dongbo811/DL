import pickle
import argparse
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

def nomorlize(data):
    y = (data.max() - data.min()) * (data - data.min()) / (data.max() - data.min()) + data.min()
    return y



def initial_weights():
    W21 =np.array([[-1, -0.014, 0.005, 0.142, -0.365, 0.850], [-0.455, 0.881, -0.504, 0.167, -1.096, 0.015]] )
    B21 = np.array([[1],[-1]] )
    W32 = np.array([[-0.1], [-0.1]]).T
    B32 = np.array([-0.2])
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
    def auto_initial_weights(self):
        self.lastdW21 = np.random.rand(1)
        self.lastdW32 = np.random.rand(1)
        self.W21 = np.random.rand(2, 6)  # 对输入层和隐藏层之间的权值进行初始赋值
        self.B21 = np.random.rand(2, 1)  # 对输入层和隐藏层之间的偏置进行初始赋值
        self.W32 = np.random.rand(1, 2)  # 对隐藏层和输出层之间的权值进行初始赋值
        self.B32 = np.random.rand(1)  # 对隐藏层和输出层之间的偏置进行初始赋值
    def fixed_initial_weights(self):
        self.lastdW21 = 0
        self.lastdW32 = 0
        self.W21 = np.array([[-1, -0.014, 0.005, 0.142, -0.365, 0.850],
                     [-0.455, 0.881, -0.504, 0.167, -1.096, 0.015]])  # 对输入层和隐藏层之间的权值进行初始赋值
        self.B21 = np.array([[1], [-1]])  # 对输入层和隐藏层之间的偏置进行初始赋值
        self.W32 = np.array([[-0.1], [-0.1]]).T  # 对隐藏层和输出层之间的权值进行初始赋值
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

def args():
    # 参数化学习
    # https://docs.python.org/zh-cn/3/library/argparse.html
    parser = argparse.ArgumentParser(description="need to change to fit your own data")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epoch", type=int, default=100000)
    parser.add_argument("--trainmomfactor", type=float, default=0.9, help='Momentum factor')
    parser.add_argument("--train", type=str, default=False, help='to train')
    parser.add_argument("--test", type=str, default=True, help='to test')
    parser.add_argument("--TrainDataPath", type=str, default=r'E:\master\EvaluationTools\pythonProject/traindata.xlsx')
    parser.add_argument("--TestDataPath", type=str, default=r'E:\master\EvaluationTools\pythonProject/testdata.xlsx')
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    opt = args()
    TrainDataPath = opt.TrainDataPath
    TestDataPath = opt.TestDataPath
    writer = SummaryWriter('./log3')
    x_train, y_train = dataloader(TrainDataPath)

    epoch = opt.epoch
    LearningRate = opt.lr
    # errorprecision = opt.errorprecision
    trainmomfactor = opt.trainmomfactor

    logistic = ActivationFunction.Sigmoid
    x_val, y_val = test(TestDataPath)
    model = Network()


    if opt.train:
        # 固定参数初始化
        model.fixed_initial_weights()
        # 随机参数初始化
        # model.auto_initial_weights()
        for i in range(epoch):
            trainoutput = model.forward(x_train)
            model.backward(trainoutput, y_train)
            # mae
            error = abs(trainoutput- y_train)
            energy = sum(error ** 2)
            pred_val = model.forward(x_val)
            writer.add_scalar('sigmoid-meanabserror-train-epoch', np.mean(error), i)
            writer.add_scalar('sigmoid-meanabserror-val-epoch', np.mean(abs(pred_val-y_val)), i)
            # writer.add_scalar('meanabserror-val-epoch', np.mean(abs(pred_val - y_val)), i)
            # print(np.mean(pred_val))
        savemodel(model, 'model.pkl')
    if opt.test:
        # 模型加载
        model = loadmodel('model.pkl')
        # 大批量测试
        x_test, y_test = test(TestDataPath)
        # 小批量测试
        # x_test, y_test = test()
        pred = model.forward(x_test)
        pred = np.around(pred, 4)
        print("预测概率：", pred[0])
        pred[pred < 0.5] = 0
        pred[pred > 0.5] = 1
        print("预测结果：", (pred[0]))
        print("标签结果：", y_test)

#
# cmd 可视化代码:tensorboard --logdir=.\log --port 8123
