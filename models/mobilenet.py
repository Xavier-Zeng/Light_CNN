import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn import init
import math

class MobileNet(nn.Module):
    '''
        1.0 MobileNet-224 实现，加宽度因子alpha,分辨力因子p
    '''
    def __init__(self, alpha=1, p=224, num_classes=1000):
        self.alpha = alpha
        self.p = p
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        # for input size 224x224
        if self.p == 224:
            self.model = nn.Sequential(
                conv_bn(  3,  int(32*self.alpha), 2), # 112x112x32
                conv_dw( int(32*self.alpha),  int(64*self.alpha), 1), # 112x112x64
                conv_dw( int(64*self.alpha), int(128*self.alpha), 2), # 56x56x128
                conv_dw(int(128*self.alpha), int(128*self.alpha), 1), # 56x56x128
                conv_dw(int(128*self.alpha), int(256*self.alpha), 2), # 28x28x256
                conv_dw(int(256*self.alpha), int(256*self.alpha), 1), # 28x28x256
                conv_dw(int(256*self.alpha), int(512*self.alpha), 2), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(1024*self.alpha), 2), # 7x7x1024
                conv_dw(int(1024*self.alpha), int(1024*self.alpha), 1), # 7x7x1024
                nn.AvgPool2d(7), # 1x1x1024, ***************************need to change
            )
         # for input size 192x192
        elif self.p == 192: 
            self.model = nn.Sequential(
                conv_bn(  3,  int(32*self.alpha), 2), # 112x112x32
                conv_dw( int(32*self.alpha),  int(64*self.alpha), 1), # 112x112x64
                conv_dw( int(64*self.alpha), int(128*self.alpha), 2), # 56x56x128
                conv_dw(int(128*self.alpha), int(128*self.alpha), 1), # 56x56x128
                conv_dw(int(128*self.alpha), int(256*self.alpha), 2), # 28x28x256
                conv_dw(int(256*self.alpha), int(256*self.alpha), 1), # 28x28x256
                conv_dw(int(256*self.alpha), int(512*self.alpha), 2), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(1024*self.alpha), 2), # 7x7x1024
                conv_dw(int(1024*self.alpha), int(1024*self.alpha), 1), # 7x7x1024
                nn.AvgPool2d(6), # 1x1x1024, ***************************need to change
            )
        # for input size 160x160
        elif self.p == 160: 
            self.model = nn.Sequential(
                conv_bn(  3,  int(32*self.alpha), 2), # 112x112x32
                conv_dw( int(32*self.alpha),  int(64*self.alpha), 1), # 112x112x64
                conv_dw( int(64*self.alpha), int(128*self.alpha), 2), # 56x56x128
                conv_dw(int(128*self.alpha), int(128*self.alpha), 1), # 56x56x128
                conv_dw(int(128*self.alpha), int(256*self.alpha), 2), # 28x28x256
                conv_dw(int(256*self.alpha), int(256*self.alpha), 1), # 28x28x256
                conv_dw(int(256*self.alpha), int(512*self.alpha), 2), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(1024*self.alpha), 2), # 7x7x1024
                conv_dw(int(1024*self.alpha), int(1024*self.alpha), 1), # 7x7x1024
                nn.AvgPool2d(5), # 1x1x1024, ***************************need to change
            )
        # for input size 128x128
        else :
            self.model = nn.Sequential(
                conv_bn(  3,  int(32*self.alpha), 2), # 112x112x32
                conv_dw( int(32*self.alpha),  int(64*self.alpha), 1), # 112x112x64
                conv_dw( int(64*self.alpha), int(128*self.alpha), 2), # 56x56x128
                conv_dw(int(128*self.alpha), int(128*self.alpha), 1), # 56x56x128
                conv_dw(int(128*self.alpha), int(256*self.alpha), 2), # 28x28x256
                conv_dw(int(256*self.alpha), int(256*self.alpha), 1), # 28x28x256
                conv_dw(int(256*self.alpha), int(512*self.alpha), 2), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(512*self.alpha), 1), # 14x14x512
                conv_dw(int(512*self.alpha), int(1024*self.alpha), 2), # 7x7x1024
                conv_dw(int(1024*self.alpha), int(1024*self.alpha), 1), # 7x7x1024
                nn.AvgPool2d(4), # 1x1x1024, ***************************need to change
            )
        self.fc = nn.Linear(int(1024*self.alpha), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(1024*self.alpha))
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mobilenet_1_0(alpha=1, p=224, num_classes=1000):
    model = MobileNet(alpha, p, num_classes)
    return model

def mobilenet_0_75(alpha=0.75, p=224, num_classes=1000):
    model = MobileNet(alpha, p, num_classes)
    return model

def mobilenet_0_5(alpha=0.5, p=224, num_classes=1000):
    model = MobileNet(alpha, p, num_classes)
    return model

def mobilenet_0_25(alpha=0.25, p=224, num_classes=1000):
    model = MobileNet(alpha, p, num_classes)
    return model

def mobilenet_1_0_224(num_classes=1000):
    model = MobileNet(1, 224, num_classes)
    return model

def mobilenet_0_75_224(num_classes=1000):
    model = MobileNet(1, 224, num_classes)
    return model

def mobilenet_0_5_224(num_classes=1000):
    model = MobileNet(1, 224, num_classes)
    return model

def mobilenet_0_25_224(num_classes=1000):
    model = MobileNet(1, 224, num_classes)
    return model

def speed(model, name, inputX, inputY):
    t0 = time.time()
    input = torch.rand(1,3,inputX,inputY).cuda()
    input = Variable(input, volatile = True)
    t1 = time.time()

    model(input)
    t2 = time.time()
    
    print('=> {} cost: {}'.format(name, t2 - t1))

if __name__ == '__main__':
    #cudnn.benchmark = True # This will make network slow ??
    #  mobilenet_1_0
    # mobilenet = mobilenet_1_0(1, 128, 1000).cuda()
    # print("=> mobilenet_1_0 :\n {}".format(mobilenet))
    # speed(mobilenet, 'mobilenet_224_1_0', 128, 128) # for 224x224

    # mobilenet = mobilenet_1_0(1, 192, 1000).cuda()
    # print("=> mobilenet_1_0 :\n {}".format(mobilenet))
    # speed(mobilenet, 'mobilenet_192_1_0', 192, 192) # for 192x192
    
    # mobilenet = mobilenet_1_0(1, 160, 1000).cuda()
    # print("=> mobilenet_1_0 :\n {}".format(mobilenet))
    # speed(mobilenet, 'mobilenet_160_1_0', 160, 160) # for 160x160
    
    # mobilenet = mobilenet_1_0(1, 128, 1000).cuda()
    # print("=> mobilenet_1_0 :\n {}".format(mobilenet))
    # speed(mobilenet, 'mobilenet_128_1_0', 128, 128) # for 128x128
    # print("=> mobilenet_1_0 param : {}".format(count_parameters(mobilenet)))

    # #  mobilenet_0_25
    # mobilenet = mobilenet_0_25(0.25, 224, 1000).cuda()
    # print("=> MobileNet 0.25x 224 :\n {}".format(mobilenet))
    # speed(mobilenet, 'MobileNet 0.25x 224 ', 224, 224) # for 224x224
    # print("=> MobileNet 0.25x 224  param : {}".format(count_parameters(mobilenet)))

    # #  mobilenet_0_5
    # mobilenet = mobilenet_0_5(0.5, 224, 1000).cuda()
    # print("=> MobileNet 0.5x 224:\n {}".format(mobilenet))
    # speed(mobilenet, 'MobileNet 0.5x 224', 224, 224)
    # print("=> MobileNet 0.5x 224 param : {}".format(count_parameters(mobilenet)))

    # #  mobilenet_0_75
    # mobilenet = mobilenet_0_75(0.75, 224, 1000).cuda()
    # print("=> MobileNet 0.75x 224 :\n {}".format(mobilenet))
    # speed(mobilenet, 'MobileNet 0.75x 224', 224, 224)
    # print("=> MobileNet 0.75x 224 param : {}".format(count_parameters(mobilenet)))

    # #  mobilenet_1_0
    mobilenet = mobilenet_1_0_224(num_classes=1000).cuda()
    # print("=> MobileNetV2 1x 224 :\n {}".format(mobilenet))
    speed(mobilenet, 'MobileNet 1x 224 ', 224, 224) # for 224x224
    print("=> MobileNet 1x 224  param : {}".format(count_parameters(mobilenet)))