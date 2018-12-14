import torch.nn as nn
import torch
import math
import time 
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn import init


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size//32, input_size//32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(self.last_channel, n_class, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            # nn.Linear(self.last_channel, n_class),
        )
        self.n_cass = n_class
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # change n_samplesxn_classx1x1 to n_samplesxn_class
        x = x.view(-1, self.n_cass)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenet_v2_0_4x_224(num_classes=1000):
    model = MobileNetV2(num_classes, input_size=224, width_mult=0.4)
    return model

def mobilenet_v2_0_75x_224(num_classes=1000):
    model = MobileNetV2(num_classes, input_size=224, width_mult=0.75)
    return model

def mobilenet_v2_1x_224(num_classes=1000):
    model = MobileNetV2(num_classes, input_size=224, width_mult=1.)
    return model

def mobilenet_v2_1_4x_224(num_classes=1000):
    model = MobileNetV2(num_classes, input_size=224, width_mult=1.4)
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


if __name__ == "__main__":
    """Testing
    """
    # model = MobileNetV2(n_class=1000, width_mult=0.4).cuda()
    # # print("=> MobileNetV2 0.4x 224 :\n {}".format(model))
    # speed(model, 'MobileNetV2 0.4x 224', 224, 224) # for 224x224
    # print("=> MobileNetV2 0.4x 224 param : {}".format(count_parameters(model)))

    # model = MobileNetV2(n_class=1000, width_mult=0.75).cuda()
    # # print("=> MobileNetV2 0.75x 224 :\n {}".format(model))
    # speed(model, 'MobileNetV2 0.75x 224', 224, 224) # for 224x224
    # print("=> MobileNetV2 0.75x 224 param : {}".format(count_parameters(model)))


    model = MobileNetV2(n_class=1000).cuda()
    # print("=> MobileNetV2 1x 224:\n {}".format(model))
    speed(model, 'MobileNetV2 1x 224', 224, 224) # for 224x224
    print("=> MobileNetV2 1x 224 param : {}".format(count_parameters(model)))

    
    # model = MobileNetV2(n_class=1000, width_mult=1.4).cuda()
    # # print("=> MobileNetV2 1.4x 224 :\n {}".format(model))
    # speed(model, 'MobileNetV2 1.4x 224', 224, 224) # for 224x224
    # print("=> MobileNetV2 1.4x 224 param : {}".format(count_parameters(model)))