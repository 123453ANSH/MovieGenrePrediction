import torch
import torch.nn as nn
from torchsummary import summary

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.block = block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #ResNet layers
        self.layer1 = self.new_layer(num_blocks=layers[0], in_channels=64, out_channels=64, stride=1)
        self.layer2 = self.new_layer(num_blocks=layers[1], in_channels=64, out_channels=128, stride=2)
        self.layer3 = self.new_layer(num_blocks=layers[2], in_channels=128, out_channels=256, stride=2)
        self.layer4 = self.new_layer(num_blocks=layers[3], in_channels=256, out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        #print("input: ", x.shape)
        x = self.conv1(x)
        #print("conv1: ", x.shape)
        x = self.bn1(x)
        #print("bn1: ", x.shape)
        x = self.relu(x)
        #print("relu: ", x.shape)
        x = self.maxpool(x)
        #print("maxpool: ", x.shape)

        x = self.layer1(x)
        #print("layer 1: ", x.shape)
        x = self.layer2(x)
        #print("layer 2: ", x.shape)
        x = self.layer3(x)
        #print("layer 3: ", x.shape)
        x = self.layer4(x)
        #print("layer 4: ", x.shape)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    def new_layer(self, num_blocks, in_channels, out_channels, stride):
        layers = []
        layers.append(self.block(in_channels, out_channels))
        intermediate_channels = out_channels
        for i in range(num_blocks):
            layers.append(self.block(intermediate_channels, out_channels))
           
        return nn.Sequential(*layers)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        identity = x.clone()
        if self.in_channels != self.out_channels:
            identity = self.identity(identity)
        #print("id1", identity.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        #Skip connection 1
        #print("id2", identity.shape, x.shape)
        x += identity
        identity = x.clone()
        #print("id3", identity.shape)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        #Skip connection 2
        x += identity

        x = self.relu(x)

        return x

def ResNet34():
    return ResNet(Block, [3, 4, 6, 3])

model = ResNet34()
summary(model, input_size=(3, 224, 224), batch_size=1)