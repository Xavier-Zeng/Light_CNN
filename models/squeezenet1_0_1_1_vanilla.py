import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import time
from collections import OrderedDict


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


# model_urls = {
#     'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
#     'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
# }


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes

        self.group1 = nn.Sequential(
            OrderedDict([
                ('squeeze', nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)),
                ('bn1', nn.BatchNorm2d(squeeze_planes)),
                ('squeeze_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group2 = nn.Sequential(
            OrderedDict([
                ('expand1x1', nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)),
                ('bn2', nn.BatchNorm2d(expand1x1_planes)),
                ('expand1x1_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group3 = nn.Sequential(
            OrderedDict([
                ('expand3x3', nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(expand3x3_planes)),
                ('expand3x3_activation', nn.ReLU(inplace=True))
            ])
        )

    def forward(self, x):
        x = self.group1(x)
        return torch.cat([self.group2(x),self.group3(x)], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=10):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=0), # 111x111, 96
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 55x55, 96

                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),

                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 27x27, 256

                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),

                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 13x13, 512

                Fire(512, 64, 256, 256), # 13x13, 512
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=0), # 111x111, 96
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 55x55, 96

                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),

                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 27x27, 256

                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),

                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), # 13x13, 512
                
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1) # 13x13, 1000
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv, # 4x4, 10
            nn.BatchNorm2d(num_classes),
            nn.AvgPool2d(13), # 1x1, 1000
            nn.ReLU(inplace=True),
            # nn.Linear(64*block.expansion, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = 2.0
                if m is final_conv:
                    m.weight.data.normal_(0, 0.01)
                else:
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    u = math.sqrt(3.0 * gain / fan_in)
                    m.weight.data.uniform_(-u, u)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        print("=> x.size() = {}".format(x.size()))
        return x.view(x.size(0), self.num_classes)

def squeezenet1_0(num_classes=1000):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = SqueezeNet(1.0, num_classes)
    # if pretrained:
    #     misc.load_state_dict(model, model_urls['squeezenet1_0'], model_root)
    return model


def squeezenet1_1(num_classes=1000):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = SqueezeNet(1.1, num_classes)
    # if pretrained:
    #     misc.load_state_dict(model, model_urls['squeezenet1_1'], model_root)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def speed(model, name, inputX, inputY):
    t0 = time.time()
    input = torch.rand(1,3,inputX,inputY).cuda()
    input = Variable(input, volatile = True)
    t1 = time.time()

    model(input)
    t2 = time.time()
    
    print('=> {} cost: {}'.format(name, t2 - t1))


if __name__ == '__main__':
    """Testing
    """
    # model = squeezenet1_0(num_classes=1000).cuda()
    # print("=> squeezenet1_0 224:\n {}".format(model))
    # speed(model, 'squeezenet1_0 224', 224, 224) # for 224x224
    # print("=> squeezenet1_0 224 param : {}".format(count_parameters(model)))

    model = squeezenet1_1(num_classes=1000).cuda()
    print("=> squeezenet1_0 227:\n {}".format(model))
    speed(model, 'squeezenet1_0 227', 227, 227) # for 227x227
    print("=> squeezenet1_0 227 param : {}".format(count_parameters(model)))



