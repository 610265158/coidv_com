import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)

    def forward(self, inputs):
        outputs = super(CausalConv1d, self).forward(inputs)
        return outputs[:, :, :-1]


class DilatedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(DilatedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias)

    def forward(self, inputs):
        outputs = super(DilatedConv1d, self).forward(inputs)
        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.filter_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation)
        self.gate_conv = DilatedConv1d(in_channels=res_channels, out_channels=res_channels, dilation=dilation)
        self.skip_conv = nn.Conv1d(in_channels=res_channels, out_channels=skip_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(in_channels=res_channels, out_channels=res_channels, kernel_size=1)

    def forward(self, inputs):
        print(inputs.shape)
        sigmoid_out = F.sigmoid(self.gate_conv(inputs))
        tahn_out = F.tanh(self.filter_conv(inputs))
        output = sigmoid_out * tahn_out


        print(output.shape)

        res_out = self.residual_conv(output)
        res_out = res_out + inputs[:, :, -res_out.size(2):]
        # res
        return res_out


class WaveNet(nn.Module):
    def __init__(self, in_depth=256, res_channels=32, skip_channels=512, dilation_depth=10, n_repeat=5):
        super(WaveNet, self).__init__()
        self.dilations = [2 ** i for i in range(dilation_depth)] * n_repeat
        self.main = nn.ModuleList([ResidualBlock(res_channels, skip_channels, dilation) for dilation in self.dilations])


    def forward(self, inputs):
        #outputs = self.preprocess(inputs)

        outputs=inputs.permute(0,2,1)
        skip_connections = []

        for layer in self.main:
            outputs = layer(outputs)


        print(outputs.shape)
        return outputs

    def preprocess(self, inputs):
        out = self.pre(inputs).transpose(1, 2)
        # out = self.pre_conv(out)
        return out
#-*-coding:utf-8-*-


if __name__=='__main__':
    model=WaveNet()


    test_data=torch.zeros(size=[12,107,32])

    res=model(test_data)

    print(res.shape)