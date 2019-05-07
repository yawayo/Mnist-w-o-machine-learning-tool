import numpy as np
import matplotlib.pyplot as plt

trainDataFile = open("data/mnist_train.csv", "r")
trainDataList = trainDataFile.readlines()
trainDataFile.close()

testDataFile = open("data/mnist_test.csv", "r")
testDataList = testDataFile.readlines()
testDataFile.close()

inputsize = 784
hiddenLayer = 256
outputsize = 10
learningRate = 0.001

class Net:
    def __init__(self, inputsize, hiddenLayer, outputsize, learningRate):
        self.inputsize  = inputsize
        self.FC = hiddenLayer
        self.outputsize = outputsize
        self.lr = learningRate

        self.intofc = np.random.normal(0, 1, (self.FC, self.inputsize))
        self.fctoout = np.random.normal(0, 1, (self.outputsize, self.FC))
        self.activation = lambda x: sigmoid(x)

    def train(self, inputList, outputList):
        input = np.array(inputList, ndmin=2).T
        target = np.array(outputList, ndmin=2).T

        hiddenoutput = self.activation(np.dot(self.intofc, input))
        output = self.activation(np.dot(self.fctoout, hiddenoutput))

        loss = target - output

        self.fctoout += self.lr * np.dot((loss * grad_sigmoid(output)), np.transpose(hiddenoutput))

        hiddenloss = np.dot(self.fctoout.T, loss)

        self.intofc += self.lr * np.dot((hiddenloss*grad_sigmoid(hiddenloss)), np.transpose(input))

    def test(self, inputList):
        input = np.array(inputList, ndmin=2).T
        hiddenoutput = self.activation(np.dot(self.intofc, input))
        return self.activation(np.dot(self.fctoout, hiddenoutput))

model = Net(inputsize, hiddenLayer, outputsize, learningRate)

# 현재 스텝을 기록한다.
step = [];
# 현재 스텝의 정확도를 기록한다.
stepPerf = [];

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def grad_sigmoid(x):
    g = sigmoid(x)*(1-sigmoid(x))
    return g

for epochs in range(1, 15, 1):
    for e in range(epochs):
        for record in trainDataList:
            # 한줄의 데이터를 읽어 쉼표를 구분자로 하여 여러 셀에 기록한다.
            cells = record.split(',')

            # 색상값이 0~255 사이 숫자이다.
            # 잘 학습할 수 있도록 0은 피하고 큰수를 피하기 위해 0.01 ~ 1.00 사이로 조정한다.
            input = (np.asfarray(cells[1:]) / 255.0 * 0.99) + 0.01

            # 정답인 1개만 아주 크고 나머진 아주 작은 one-hot encoding으로 구성되어 있다.
            target = np.zeros(outputsize) + 0.01
            target[int(cells[0])] = 0.99

            # 학습시킨다.
            model.train(input, target)

    resultCard = []

    for record in testDataList:
        # 한줄의 데이터를 읽어 쉼표를 구분자로 하여 여러 셀에 기록한다.
        cells = record.split(',')
        # 첫번째 항목은 라벨링된 정답지이다.
        target = int(cells[0])

        # 색상값이 0~255 사이 숫자이다.
        # 잘 학습할 수 있도록 0은 피하고 큰수를 피하기 위해 0.01 ~ 1.00 사이로 조정한다.
        input = (np.asfarray(cells[1:]) / 255.0 * 0.99) + 0.01

        # 테스트 쿼리를 실행한다.
        output = model.test(input)
        # 배열중 최대값을 가진 인덱스를 반환한느 argmax를 통해 찾은 최대값과 정답지를 비교한다.
        if (np.argmax(output) == target):
            # 정답이라면 1 표시
            resultCard.append(1)
        else:
            # 오답이라면 0 표시
            resultCard.append(0)

    # resultCard 리스트를 배열형태로 변환한 후 평균을 구한다.
    perf = np.asarray(resultCard).mean() * 100.0

    # 현재 에폭을 step에 저장
    step.append(epochs);
    # 현재 에폭의 결과를 stepPerf에 저장
    print("Accuracy : ", perf)
    stepPerf.append(perf);
    print("epoch : ", epochs)

plt.plot(step, stepPerf)
plt.show()