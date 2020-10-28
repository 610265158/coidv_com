
import torch
import torch.nn as nn

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


BN_MOMENTUM=0.02
BN_EPS=1e-5
ACT_FUNCTION=MemoryEfficientSwish


class Attention(nn.Module):

    def __init__(self, input_dim=512, output_dim=512):
        super(Attention, self).__init__()

        self.att=nn.Sequential(nn.Linear(input_dim, output_dim//4,bias=False),
                               nn.BatchNorm1d(output_dim//4,momentum=BN_MOMENTUM,eps=BN_EPS),
                               ACT_FUNCTION(),
                               nn.Linear(output_dim//4, output_dim, bias=False),
                               nn.BatchNorm1d(output_dim, momentum=BN_MOMENTUM,eps=BN_EPS),
                               nn.Sigmoid())

    def forward(self, x):
        xx = self.att(x)

        return x*xx



class ResBlock(nn.Module):

    def __init__(self, input_dim=512, output_dim=512):
        super(ResBlock, self).__init__()

        self.att=nn.Sequential(nn.Linear(input_dim, output_dim//4,bias=False),
                               nn.BatchNorm1d(output_dim//4,momentum=BN_MOMENTUM,eps=BN_EPS),
                               ACT_FUNCTION(),
                               nn.Linear(output_dim//4, output_dim, bias=False),
                               nn.BatchNorm1d(output_dim, momentum=BN_MOMENTUM,eps=BN_EPS),
                               )
        self.act=ACT_FUNCTION()
    def forward(self, x):
        xx = self.att(x)

        return self.act(x+xx)




class Complexer(nn.Module):

    def __init__(self, num_features=875, num_targets=206,num_extra_targets=402, hidden_size=512):
        super(Complexer, self).__init__()

        self.bn_init = nn.BatchNorm1d(num_features, momentum=0.01, eps=BN_EPS)
        self.dense1 =nn.Sequential(nn.Linear(num_features, hidden_size,bias=False),
                                   nn.BatchNorm1d(hidden_size,momentum=BN_MOMENTUM,eps=BN_EPS),
                                   ACT_FUNCTION(),
                                   nn.Dropout(0.5),
                                   )

        self.dense2 =nn.Sequential(ResBlock(hidden_size,hidden_size),
                                   nn.Dropout(0.5),
                                   nn.Linear(hidden_size, hidden_size, bias=False),
                                   nn.BatchNorm1d(hidden_size, momentum=BN_MOMENTUM, eps=BN_EPS),
                                   ACT_FUNCTION(),
                                   nn.Dropout(0.5),
                                   ResBlock(hidden_size, hidden_size),
                                   )


        self.max_p = nn.MaxPool1d(kernel_size=3,stride=1,padding=1)
        self.mean_p = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.att=Attention(hidden_size,hidden_size)


        self.dense3 = nn.Linear(hidden_size, hidden_size)

        self.dense4 = nn.Linear(hidden_size*3, num_targets)

        self.dense5 = nn.Linear(hidden_size * 3, num_extra_targets)
    def forward(self, x):
        x = self.bn_init(x)
        x = self.dense1(x)
        x = self.dense2(x)

        x = self.att(x)
        x = self.dense3(x)


        x=x.unsqueeze(dim=1)
        yy=self.max_p(x)
        zz=self.mean_p(x)
        x=torch.cat([yy,zz,x],dim=2)
        x=x.squeeze(1)


        xx = self.dense4(x)
        yy = self.dense5(x)
        return xx,yy



if __name__=='__main__':
    model=Complexer()
    data=torch.zeros(size=[12,940])
    res=model(data)

    print(res.shape)