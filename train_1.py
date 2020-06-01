import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data   # 数据集的抽象类
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import warnings
from PIL import Image
import random
from model import NET     # 从自定义的modle.py文件中导入网络模型
from Visualize import plot_with_labels  # 从自定义的Visualizee.py文件中导入函数

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 利用过滤器来忽略warnings
warnings.filterwarnings('ignore')

# 超参数设置
EPOCH = 10  # 一共训练10次
best_acc = 0.75
BATCH_SIZE = 21  # 21个图片为一个batch 一共1050张图片 共50个batch
LR = 0.01  # learning rate
Data_PATH = 'data/pokemon'  # 图像文件路径
test_num = 150  # 测试集的数量

# 加载数据集
# 数据增强。对训练集进行预处理
transform_train = transforms.Compose([
    transforms.Resize(256),
    # transforms.RandomResizedCrop(224),
    transforms.CenterCrop(224),  # 先四周填充 将图像居中裁剪成224*224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转 默认概率0.5
    transforms.ToTensor(),  # C H W格式   [0,1】数据范围
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 用给定的均值和标准差对每个通道的数据进行正则化
])                                                                      # Normalized_image=(image-mean)/std
# 对测试集进行数据处理
transform_test = transforms.Compose([
    transforms.Resize(256),  # 数据集图像大小不一，加载测试集时也要进行了Resize and Crop ，否则会报错
    # transforms.RandomResizedCrop(224),
    transforms.CenterCrop(224),  # 先四周填充 将图像居中裁剪成224*224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# 用Dataset自带的方法ImageFolder对分好类的文件夹进行读取 作为训练集
train_set = torchvision.datasets.ImageFolder(root=Data_PATH, transform=transform_train)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                           pin_memory=True)  # 将数据保存在pin_memory中
# 测试集没有分好类 所以不能用ImageFolder直接读取
# testset = torchvision.datasets.ImageFolder(root='data/test', transform=transform_test)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
#                                           pin_memory=True)  # 数据加载器在返回前将张量复制到CUDA固定内存中

# 从pokemon文件夹中提取测试集
dataset = ImageFolder(Data_PATH)
# print(dataset.class_to_idx)
test_data = random.sample(dataset.samples, test_num)  # 随机选取150个作为测试集
# print(test_data)
test_inputs = []  # 测试集的输入 ，一个列表 存放图像的路径
test_labels = []  # 测试集的输出，标签，一个列表 存放图像对应的标签（0-4）
for x, y in test_data:  # 遍历test_data -->一个元素是元组的列表，每个元组第一个元素是图片的路径，第二个是该图片对应的标签
    test_inputs.append(x)
    test_labels.append(y)


# 自定义数据读取类  要实现__len__ 和__getitem__方法
class MyDataset(Dataset):
    def __init__(self, file_path, labels, transform):
        self.file_path = file_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):  # 索引数据集中的某一个数据
        image = Image.open(self.file_path[idx]).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(self.labels[idx])


test_loader = torch.utils.data.DataLoader(MyDataset(test_inputs, test_labels, transform_test), #先转化成torch能识别的
                                          batch_size=BATCH_SIZE,                                # dataset 再批处理
                                          shuffle=False, num_workers=2,
                                          pin_memory=True)

# 实例化
net = NET()

# 定义损失函数和优化方式
loss_func = nn.CrossEntropyLoss()  # 损失函数为交叉熵 内置了softmax层
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,   # net.parameters()可迭代的variable指定因优化哪些参数
                      weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，weight_decay并采用L2正则化（权值衰减）

# 开始训练
if __name__ == '__main__':
    with open('model_params.txt', 'w') as f4:   # 将模型参数写入model_params.txt文件
        for parameters in net.parameters():     # 模块参数的迭代器
            f4.write(str(parameters))
            f4.write('\n')
        for name, parameters in net.named_parameters():
            f4.write(name + ':' + str(parameters.size()))
            f4.write('\n')
        f4.flush()
        f4.close()
    with open("acc.txt", "w") as f1:        # 将测试准确率写入acc.txt文件
        with open("log.txt", "w")as f2:     # 训练日志
            save_train_Acc = []
            save_test_Acc = []
            save_train_Loss = []
            save_test_Loss = []
            for epoch in range(EPOCH):
                net.train()  # 开始训练
                train_corr = 0.
                total_loss = 0.
                for i, data in enumerate(train_loader):
                    train_input, target = data
                    # train_input, target = Variable(train_input), Variable(target)
                    train_input, target = train_input.to(device), target.to(device)  # GPU
                    output = net(train_input)
                    loss = loss_func(output, target)
                    total_loss += loss.item()  # 总的损失 用item()累加成数字
                    result = torch.max(output, 1)[1]  # 返回概率最大值位置的索引  one-hot 编码---》标签
                    train_corr += (result == target).sum()  # 训练的结果与原来的标签相等的个数
                    # SGD
                    optimizer.zero_grad()
                    loss.backward()   # 计算梯度
                    optimizer.step()  # 更新参数

                    # 每五个batch打印一下
                    # print(len(train_loader))  # 共50个batch
                    if i % 5 == 0:
                        print('Epoch: %d | step: %d | train_loss: %.4f | train_acc: %.2f '
                              % (epoch, i, loss.item(), float(train_corr) / ((i + 1) * BATCH_SIZE)))
                        f2.write('Epoch: %d | step: %d | train_loss: %.4f | train_acc: %.2f'
                                 % (epoch, i, loss.item(), float(train_corr) / ((i + 1) * BATCH_SIZE)))
                        f2.write('\n')
                        f2.flush()
                # 训练完一个epoch 就打印总的损失值 准确率
                print('Epoch: %d | iteration: %d | total_loss: %.4f | train_acc: %.2f'
                      % (epoch, i, total_loss, float(train_corr) / len(train_set)))
                f2.write('Epoch: %d | step: %d | total_loss: %.4f | train_acc: %.2f'
                         % (epoch, i, total_loss, float(train_corr) / len(train_set)))
                f2.write('\n')
                f2.flush()
                save_train_Acc.append(round(float(train_corr) / len(train_set), 2))
                save_train_Loss.append(round(float(total_loss) / len(train_set), 2))
                with torch.no_grad():  # 显示地取消模型变量的梯度 测试时不要在梯度更新
                    net.eval()  # 开始测试
                    print("Starting testing!")
                    test_loss = 0.
                    correct = 0.
                    accuracy = 0.
                    for i, data in enumerate(test_loader):
                        test_x, target = data
                        # test_x, target = Variable(test_x), Variable(target)
                        test_x, target = test_x.to(device), target.to(device)

                        output = net(test_x)
                        _, pred = torch.max(output.data, 1)  # 或者pred =output.argmax(dim=1)
                        # 计算准确率
                        loss = loss_func(output, target)
                        test_loss += loss.item()
                        correct += pred.eq(target.data).sum()
                    test_loss /= len(test_inputs)
                    accuracy = float(correct) / float(len(test_inputs))  # 注意如果不加float  accuracy 为0.00 ！！！
                    save_test_Acc.append(round(accuracy, 2))
                    save_test_Loss.append(round(test_loss, 2))
                    torch.save(net.state_dict(), 'params.pkl')  # 仅保存和加载模型参数
                    # 打印 总的损失值 正确的个数 和测试准确率 并写入到acc.txt文件中去
                    print('Epoch: ', epoch, '| test_loss: %.4f' % test_loss, '| correct: %d' % correct,
                          '| test accuracy: %.3f' % accuracy)
                    f1.write('Epoch: %d | test_loss: %.4f | correct: %d | test accuracy: %.3f'
                             % (epoch, test_loss, correct, accuracy))
                    f1.write('\n')
                    f1.flush()  # 将缓冲区写入
                    # plot_with_labels(save_train_Loss, save_train_Acc, save_test_Loss, save_test_Acc)
                    if accuracy > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f" % (epoch + 1, accuracy))
                        f3.close()
                        best_acc = accuracy
            print("Congratulating! Training Finished, TotalEPOCH=%d" % EPOCH)
            plot_with_labels(save_train_Loss, save_train_Acc, save_test_Loss, save_test_Acc)