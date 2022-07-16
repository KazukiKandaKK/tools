"""Create a Mnist of torch"""
import torch as th
import torch.nn as nn  # pylint: disable=R0402
import torch.optim as optim  # pylint: disable=R0402
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch_net import Net


class Execute:  # pylint: disable=R0902R0402
    '''
    Execute Deep Neural Network
    '''
    # Define
    BATCH_SIZE = 100
    WEIGHT_DECAY = 0.005
    LEARNING_RATE = 0.0001
    EPOCH = 10

    def __init__(self) -> None:
        # Tensor形式のオブジェクトを準備
        self.trans = torchvision.transforms.ToTensor()
        # 正規化を行うための準備
        self.trans = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
        # MNISTのデータセットを取得
        self.trainset = torchvision.datasets.MNIST(
            root='path', train=True, download=True, transform=self.trans)
        self.trainloader = th.utils.data.DataLoader(
            self.trainset, batch_size=100, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.MNIST(
            root='~/Documents/work/MNISTDataset/data',
            train=False, download=True, transform=self.trans)
        self.testloader = th.utils.data.DataLoader(
            self.testset, batch_size=100, shuffle=True, num_workers=2)

        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.net = Net()
        self.net = self.net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.0001,
                                   momentum=0.9, weight_decay=0.005)
        # Variable
        self.train_loss_value = []  # trainingのlossを保持するlist
        self.train_acc_value = []  # trainingのaccuracyを保持するlist
        self.test_loss_value = []  # testのlossを保持するlist
        self.test_acc_value = []  # testのaccuracyを保持するlist

    def execute(self):
        '''
        execute to calculate of neural network
        '''

        # Learning
        for epoch in tqdm(range(self.EPOCH)):
            print('epoch', epoch+1)  # epoch数の出力
            for (inputs, labels) in tqdm(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            sum_loss = 0.0  # lossの合計
            sum_correct = 0  # 正解率の合計
            sum_total = 0  # dataの数の合計

            # train dataを使ってテストをする(パラメータ更新がないようになっている)
            for (inputs, labels) in tqdm(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                sum_loss += loss.item()  # lossを足していく
                _, predicted = outputs.max(1)  # 出力の最大値の添字(予想位置)を取得
                sum_total += labels.size(0)  # labelの数を足していくことでデータの総和を取る
                sum_correct += (predicted == labels).sum().item()
            # lossとaccuracy出力
            print(
                f"train mean loss={sum_loss*self.BATCH_SIZE/len(self.trainloader.dataset)},accuracy={float(sum_correct/sum_total)}")  # pylint: disable=C0301
            # traindataのlossをグラフ描画のためにlistに保持
            self.train_loss_value.append(
                sum_loss*self.BATCH_SIZE/len(self.trainloader.dataset))
            # traindataのaccuracyをグラフ描画のためにlistに保持
            self.train_acc_value.append(float(sum_correct/sum_total))

            sum_loss = 0.0
            sum_correct = 0
            sum_total = 0

            # test dataを使ってテストをする
            for (inputs, labels) in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                sum_loss += loss.item()
                _, predicted = outputs.max(1)
                sum_total += labels.size(0)
                sum_correct += (predicted == labels).sum().item()
            print(
                f"test  mean loss={sum_loss*self.BATCH_SIZE/len(self.testloader.dataset)}, accuracy={float(sum_correct/sum_total)}")  # pylint: disable=C0301
            self.test_loss_value.append(
                sum_loss*self.BATCH_SIZE/len(self.testloader.dataset))
            self.test_acc_value.append(float(sum_correct/sum_total))

    def print_result(self):
        '''
        print result
        '''
        plt.figure(figsize=(6, 6))  # グラフ描画用

        # 以下グラフ描画
        plt.plot(range(self.EPOCH), self.train_loss_value)
        plt.plot(range(self.EPOCH), self.test_loss_value, c='#00ff00')
        plt.xlim(0, self.EPOCH)
        plt.ylim(0, 2.5)
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.legend(['train loss', 'test loss'])
        plt.title('loss')
        plt.savefig("./loss_image.png")
        plt.clf()

        plt.plot(range(self.EPOCH), self.train_acc_value)
        plt.plot(range(self.EPOCH), self.test_acc_value, c='#00ff00')
        plt.xlim(0, self.EPOCH)
        plt.ylim(0, 1)
        plt.xlabel('EPOCH')
        plt.ylabel('ACCURACY')
        plt.legend(['train acc', 'test acc'])
        plt.title('accuracy')
        plt.savefig("./accuracy_image.png")
