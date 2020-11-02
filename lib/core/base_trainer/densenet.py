
import torch
import torch.nn as nn


from lib.core.base_trainer.model_utils import BN_EPS,BN_MOMENTUM,ACT_FUNCTION,Attention



class DenseBlock(nn.Module):

    def __init__(self, input_dim=512, output_dim=512):
        super(DenseBlock, self).__init__()

        self.att=nn.Sequential(nn.Linear(input_dim, output_dim,bias=False),
                               nn.BatchNorm1d(output_dim,momentum=BN_MOMENTUM,eps=BN_EPS),
                               ACT_FUNCTION(),
                               nn.Linear(output_dim, output_dim, bias=False),
                               nn.BatchNorm1d(output_dim, momentum=BN_MOMENTUM,eps=BN_EPS),
                               ACT_FUNCTION()
                               )

    def forward(self, x):
        xx = self.att(x)

        return torch.cat([xx,x],dim=1)


class Denseplexer(nn.Module):

    def __init__(self, num_features=875, num_targets=206,num_extra_targets=402, hidden_size=512):
        super(Denseplexer, self).__init__()

        self.bn_init = nn.BatchNorm1d(num_features, momentum=BN_MOMENTUM, eps=BN_EPS)

        self.dense1 =nn.Sequential(nn.Linear(num_features, hidden_size,bias=False),
                                   nn.BatchNorm1d(hidden_size,momentum=BN_MOMENTUM,eps=BN_EPS),
                                   ACT_FUNCTION(),
                                   nn.Dropout(0.5),
                                   )

        self.dense2 =nn.Sequential(DenseBlock(hidden_size,hidden_size),
                                   nn.Dropout(0.5),
                                   nn.Linear(2*hidden_size, hidden_size, bias=False),
                                   nn.BatchNorm1d(hidden_size, momentum=BN_MOMENTUM, eps=BN_EPS),
                                   ACT_FUNCTION(),
                                   DenseBlock(hidden_size, hidden_size),
                                   nn.Dropout(0.5),
                                   nn.Linear(2 * hidden_size, hidden_size, bias=False),
                                   nn.BatchNorm1d(hidden_size, momentum=BN_MOMENTUM, eps=BN_EPS),
                                   ACT_FUNCTION(),
                                   )

        self.max_p = nn.MaxPool1d(kernel_size=3,stride=1,padding=1)
        self.mean_p = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.att = Attention(hidden_size,hidden_size)

        self.dense3 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size, momentum=BN_MOMENTUM, eps=BN_EPS),
                                    ACT_FUNCTION())

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