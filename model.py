import torch.nn as nn

# 定义网络模型

class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        # self.features = nn.Sequential(
        self.conv1 = nn.Sequential(  # 利用Sequential 迅速搭建
            nn.Conv2d(3, 64, 11, 4, 2),  # 卷积
            nn.ReLU(inplace=True),  # 激活
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化大小缩小一遍
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(3, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 防止过拟合 默认丢掉0.5
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5)  # 最后分为5类
        )

    def forward(self, x):
        x = self.conv1(x)  # 3*224*224--->64*55*55-->64*27*27 输出图像尺寸=（输入-卷积核大小+2*padding)/步长+1
        x = self.conv2(x)  # 64*27*27-->192*27*27--->192*192*13
        x = self.conv3(x)  # 192*192*13--->384*13*13   如果stride=1,padding=（kernel_size-1)/2，则图像卷积后大小不变
        x = self.conv4(x)  # 384*13*13--->256*13*13
        x = self.conv5(x)  # 256*13*13--->256*13*13--->256*6*6 (池化）
        x = x.view(x.size(0), 256 * 6 * 6)  # 将多维度的Tensor展平成一维,才放入全连接层
        x = self.classifier(x)
        return x