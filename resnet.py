import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        #populate the layers with your custom functions or pytorch
        #functions.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=3, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #ResNet layers
        self.layer1 = self.new_block(64, 64, stride=1)
        self.layer2 = self.new_block(64, 128, stride=2)
        self.layer3 = self.new_block(128, 256, stride=2)
        self.layer4 = self.new_block(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        #TODO: implement the forward function for resnet,
        #use all the functions you've made
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x

    def new_block(self, in_channels, out_channels, stride):

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU()]

        return nn.Sequential(*layers)
