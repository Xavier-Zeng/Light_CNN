import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable

class MobileNet(nn.Module):
    '''
        1.0 MobileNet-224 实现，目前还没加宽度因子alpha,分辨力因子p
    '''
    def __init__(self, num_classes=1000):
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

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), # 112x112x32
            conv_dw( 32,  64, 1), # 112x112x64
            conv_dw( 64, 128, 2), # 56x56x128
            conv_dw(128, 128, 1), # 56x56x128
            conv_dw(128, 256, 2), # 28x28x256
            conv_dw(256, 256, 1), # 28x28x256
            conv_dw(256, 512, 2), # 14x14x512
            conv_dw(512, 512, 1), # 14x14x512
            conv_dw(512, 512, 1), # 14x14x512
            conv_dw(512, 512, 1), # 14x14x512
            conv_dw(512, 512, 1), # 14x14x512
            conv_dw(512, 512, 1), # 14x14x512
            conv_dw(512, 1024, 2), # 7x7x1024
            conv_dw(1024, 1024, 1), # 7x7x1024
            nn.AvgPool2d(7), # 1x1x1024
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def mobilenet_224_1_0(num_classes=1000):
    model = MobileNet(num_classes)
    return model

def speed(model, name):
    t0 = time.time()
    input = torch.rand(1,3,224,224).cuda()
    input = Variable(input, volatile = True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()
    
    print('%10s : %f' % (name, t3 - t2))

if __name__ == '__main__':
    #cudnn.benchmark = True # This will make network slow ??
    
    mobilenet = mobilenet_224_1_0(1000).cuda()

    
    speed(mobilenet, 'mobilenet')