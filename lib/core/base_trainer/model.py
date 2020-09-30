import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MemoryEfficientSwish
from train_config import config as cfg


MOMENTUM=0.03
EPS=1e-5
ACT_FUNCTION=MemoryEfficientSwish
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
class ResBlock(nn.Module):
    def __init__(self, input_dim=256,output_dim=256,k_size=5):
        super(ResBlock, self).__init__()


        self.shortcut = nn.Sequential(nn.Conv1d(in_channels=input_dim,
                                            out_channels=output_dim,
                                            kernel_size=1,
                                            stride=1),
                                  nn.BatchNorm1d(output_dim, momentum=MOMENTUM, eps=EPS))


        self.conv=nn.Sequential(nn.Conv1d(in_channels=input_dim,
                                          out_channels=input_dim,
                                          kernel_size=1,
                                          stride=1),
                                nn.BatchNorm1d(input_dim,momentum=MOMENTUM,eps=EPS),
                                ACT_FUNCTION(),
                                nn.Conv1d(in_channels=input_dim,
                                          out_channels=output_dim,
                                          kernel_size=k_size,
                                          padding=((k_size-1)//2),
                                          stride=1),
                                nn.BatchNorm1d(output_dim ,momentum=MOMENTUM,eps=EPS),
                                )


        self.act=ACT_FUNCTION()
    def forward(self,x):


        shortcut=self.shortcut(x)

        bypath=self.conv(x)

        return self.act(shortcut+bypath)

class Attention(nn.Module):
    def __init__(self, input_dim=384,refraction=4):
        super(Attention, self).__init__()

        self.conv=nn.Sequential(nn.Conv1d(in_channels=input_dim, kernel_size=1, out_channels=input_dim//refraction,
                                              stride=1),
                                nn.BatchNorm1d(input_dim//refraction,momentum=MOMENTUM,eps=EPS),
                                nn.Conv1d(in_channels=input_dim//refraction, kernel_size=1, out_channels=input_dim,
                                          stride=1),
                                nn.BatchNorm1d(input_dim,momentum=MOMENTUM,eps=EPS),
                                nn.Sigmoid()
                                )


    def forward(self,x):


        attention=self.conv(x)

        return x*attention



class GRU_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.3, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(GRU_model, self).__init__()
        self.pre_length = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.drop_embed=nn.Dropout(0.3)
        self.preconv = nn.Sequential( nn.Conv1d(in_channels=5+6, kernel_size=5, out_channels=256,
                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(256,momentum=MOMENTUM,eps=EPS),
                                    ACT_FUNCTION(),
                                    nn.Dropout(0.5),
                                    nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                              stride=1,
                                              padding=2, bias=False),
                                    nn.BatchNorm1d(256,momentum=MOMENTUM,eps=EPS),
                                    ACT_FUNCTION(),
                                    nn.Dropout(0.5),

                                      )

        self.gru = nn.GRU(
            input_size=embed_dim * 3+256,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )


        self.post_conv=nn.Sequential( nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(512,momentum=MOMENTUM,eps=EPS),
                                    ACT_FUNCTION(),
                                    nn.Dropout(0.5),
                                    nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                                stride=1,
                                                padding=2, bias=False),
                                    nn.BatchNorm1d(512, momentum=MOMENTUM,eps=EPS),
                                    ACT_FUNCTION(),
                                    nn.Dropout(0.5),
                                    )

    def forward(self, seqs):


        seqs_base=seqs[:,:,0:3].long()

        seqs_extra_fea=seqs[:,:,3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        embed_reshaped = self.drop_embed(embed_reshaped)

        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)
        seqs_extra_fea = self.preconv(seqs_extra_fea)
        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)


        feature=torch.cat([embed_reshaped,seqs_extra_fea],dim=-1)

        output, hidden = self.gru(feature)

        output = output.permute(0, 2, 1)

        output = self.post_conv(output)

        output = output.permute(0, 2, 1)
        output = output[:, :self.pre_length, ...]

        return output

class LSTM_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.3, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(LSTM_model, self).__init__()
        self.pre_length = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.drop_embed = nn.Dropout(0.3)
        self.preconv = nn.Sequential(nn.Conv1d(in_channels=5 + 6, kernel_size=5, out_channels=256,
                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(256, momentum=MOMENTUM, eps=EPS),
                                     ACT_FUNCTION(),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(256, momentum=MOMENTUM, eps=EPS),
                                     ACT_FUNCTION(),
                                     nn.Dropout(0.5),

                                     )

        self.gru = nn.LSTM(
            input_size=embed_dim * 3+256,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )


        self.post_conv=nn.Sequential( nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(512,momentum=MOMENTUM,eps=EPS),
                                    ACT_FUNCTION(),
                                    nn.Dropout(0.5),
                                    nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                                stride=1,
                                                padding=2, bias=False),
                                    nn.BatchNorm1d(512, momentum=MOMENTUM,eps=EPS),
                                    ACT_FUNCTION(),
                                    nn.Dropout(0.5),
                                    )
    def forward(self, seqs):
        seqs_base = seqs[:, :, 0:3].long()

        seqs_extra_fea = seqs[:, :, 3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        embed_reshaped = self.drop_embed(embed_reshaped)

        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)

        seqs_extra_fea = self.preconv(seqs_extra_fea)
        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)
        feature = torch.cat([embed_reshaped, seqs_extra_fea], dim=-1)

        output, hidden = self.gru(feature)

        output = output.permute(0, 2, 1)

        output = self.post_conv(output)

        output = output.permute(0, 2, 1)
        output = output[:, :self.pre_length, ...]

        return output

class LSTM_GRU_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.3, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(LSTM_GRU_model, self).__init__()
        self.pre_length = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.drop_embed = nn.Dropout(0.3)
        self.preconv = nn.Sequential(nn.Conv1d(in_channels=5 + 6, kernel_size=5, out_channels=256,
                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(256, momentum=MOMENTUM, eps=EPS),
                                     ACT_FUNCTION(),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(256, momentum=MOMENTUM, eps=EPS),
                                     ACT_FUNCTION(),
                                     nn.Dropout(0.5),

                                     )

        self.lstm = nn.LSTM(
            input_size=embed_dim * 3+256,
            hidden_size=hidden_dim,
            num_layers=hidden_layers-1,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.gru = nn.GRU(
            input_size=hidden_dim*2,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.post_conv = nn.Sequential(nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(512, momentum=MOMENTUM, eps=EPS),
                                       ACT_FUNCTION(),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(512, momentum=MOMENTUM, eps=EPS),
                                       ACT_FUNCTION(),
                                       nn.Dropout(0.5),
                                       )

    def forward(self, seqs):
        seqs_base = seqs[:, :, 0:3].long()

        seqs_extra_fea = seqs[:, :, 3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        embed_reshaped = self.drop_embed(embed_reshaped)

        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)
        seqs_extra_fea = self.preconv(seqs_extra_fea)
        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)

        feature = torch.cat([embed_reshaped, seqs_extra_fea], dim=-1)


        output, hidden = self.lstm(feature)
        output, hidden = self.gru(output)

        output = output.permute(0, 2, 1)

        output = self.post_conv(output)

        output = output.permute(0, 2, 1)
        output = output[:, :self.pre_length, ...]

        return output


class GRU_LSTM_model(nn.Module):
    def __init__(
            self, seq_length=107, pred_len=68, dropout=0.3, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(GRU_LSTM_model, self).__init__()
        self.pre_length = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.drop_embed = nn.Dropout(0.3)
        self.preconv = nn.Sequential(nn.Conv1d(in_channels=5 + 6, kernel_size=5, out_channels=256,
                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(256, momentum=MOMENTUM, eps=EPS),
                                     ACT_FUNCTION(),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(256, momentum=MOMENTUM, eps=EPS),
                                     ACT_FUNCTION(),
                                     nn.Dropout(0.5),

                                     )

        self.gru = nn.GRU(
            input_size=embed_dim * 3 +256,
            hidden_size=hidden_dim,
            num_layers=hidden_layers - 1,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.post_conv = nn.Sequential(nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(512, momentum=MOMENTUM, eps=EPS),
                                       ACT_FUNCTION(),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(512, momentum=MOMENTUM, eps=EPS),
                                       ACT_FUNCTION(),
                                       nn.Dropout(0.5),
                                       )

    def forward(self, seqs):
        seqs_base = seqs[:, :, 0:3].long()

        seqs_extra_fea = seqs[:, :, 3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        embed_reshaped = self.drop_embed(embed_reshaped)

        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)
        seqs_extra_fea = self.preconv(seqs_extra_fea)
        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)

        feature = torch.cat([embed_reshaped, seqs_extra_fea], dim=-1)



        output, hidden = self.gru(feature)
        output, hidden = self.lstm(output)

        output = output.permute(0, 2, 1)

        output = self.post_conv(output)

        output = output.permute(0, 2, 1)
        output = output[:, :self.pre_length, ...]

        return output

class TRANSFORMER_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.3, embed_dim=128, hidden_dim=256, hidden_layers=2):
        super(TRANSFORMER_model, self).__init__()

        self.pre_length = pred_len
        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.drop_embed = nn.Dropout(0.3)
        self.preconv = nn.Sequential(nn.Conv1d(in_channels=5 + 6, kernel_size=5, out_channels=256,
                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(256, momentum=MOMENTUM, eps=EPS),
                                     ACT_FUNCTION(),
                                     nn.Dropout(0.5),
                                     nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(256, momentum=MOMENTUM, eps=EPS),
                                     ACT_FUNCTION(),
                                     nn.Dropout(0.5),

                                     )
        self.gru = nn.GRU(
            input_size=embed_dim * 3+256,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 64, 1024, dropout=0.3, activation='relu'),
                                            3)

        self.post_conv = nn.Sequential(nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(512, momentum=MOMENTUM, eps=EPS),
                                       ACT_FUNCTION(),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(in_channels=512, kernel_size=5, out_channels=512,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(512, momentum=MOMENTUM, eps=EPS),
                                       ACT_FUNCTION(),
                                       nn.Dropout(0.5),
                                       )


    def forward(self, seqs):
        seqs_base = seqs[:, :, 0:3].long()

        seqs_extra_fea = seqs[:, :, 3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        embed_reshaped = self.drop_embed(embed_reshaped)

        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)
        seqs_extra_fea = self.preconv(seqs_extra_fea)
        seqs_extra_fea = seqs_extra_fea.permute(0, 2, 1)

        feature = torch.cat([embed_reshaped, seqs_extra_fea], dim=-1)


        output, hidden = self.gru(feature)

        encoded = self.encoder(output)

        encoded = encoded.permute(0, 2, 1)
        output=self.post_conv(encoded)
        output = output.permute(0, 2, 1)

        output = output[:, :self.pre_length, ...]

        return output


class OneDirectionResBlock(nn.Module):
    def __init__(self,input_dim,k_size=3,stride=1,expansion=4):
        super().__init__()
        self.expansion=expansion
        self.stride=stride
        output_dim=expansion*input_dim


        if stride==1:
            if expansion!=1:
                self.conv=nn.Sequential(
                                    nn.Conv2d(in_channels=input_dim,
                                              out_channels=output_dim,
                                              kernel_size=[1, 1],
                                              stride=(1, 1),
                                              bias=False),
                                    nn.BatchNorm2d(output_dim, momentum=MOMENTUM, eps=EPS),

                )

            self.bypath = nn.Sequential(
                nn.Conv2d(in_channels=input_dim,
                          out_channels=output_dim,
                          kernel_size=[1, 1],
                          stride=(1, 1),
                          bias=False),
                nn.BatchNorm2d(output_dim, momentum=MOMENTUM, eps=EPS),
                ACT_FUNCTION(),

                nn.Conv2d(in_channels=output_dim,
                          out_channels=output_dim,
                          kernel_size=[k_size, 1],
                          stride=(1, 1),
                          padding=((k_size - 1) // 2, 0)),
                nn.BatchNorm2d(output_dim, momentum=MOMENTUM, eps=EPS),
            )

        else:
            if expansion!=1 or stride!=1:
                self.conv=nn.Sequential(
                                    nn.Conv2d(in_channels=input_dim,
                                              out_channels=output_dim,
                                              kernel_size=[k_size, 1],
                                              stride=(2, 1),
                                              padding=((k_size-1), 0),
                                              bias=False),
                                    nn.BatchNorm2d(output_dim, momentum=MOMENTUM, eps=EPS),

                )

            self.bypath = nn.Sequential(
                nn.Conv2d(in_channels=input_dim,
                          out_channels=output_dim,
                          kernel_size=[1, 1],
                          stride=(1, 1),
                          padding=(0, 0),
                          bias=False),
                nn.BatchNorm2d(output_dim, momentum=MOMENTUM, eps=EPS),
                ACT_FUNCTION(),

                nn.Conv2d(in_channels=output_dim,
                          out_channels=output_dim,
                          kernel_size=[k_size, 1],
                          stride=(2, 1),
                          padding=(k_size - 1, 0)),
                nn.BatchNorm2d(output_dim, momentum=MOMENTUM, eps=EPS))

        self.act=ACT_FUNCTION()
    def forward(self, x):

        if self.expansion!=1 or self.stride!=1:
            short_cut=self.conv(x)
        else:
            short_cut=x

        bypath=self.bypath(x)

        return self.act(short_cut+bypath)



class ImageModel(nn.Module):
    def __init__(self,pred_len=68):
        super().__init__()


        self.pre_length=pred_len
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=6,out_channels=16,kernel_size=[7,1],stride=1,padding=(3,0)),
                                 nn.BatchNorm2d(16,momentum=MOMENTUM,eps=EPS),
                                 ACT_FUNCTION())

        self.conv2=nn.Sequential(OneDirectionResBlock(16,k_size=5,expansion=2,stride=2),
                                 OneDirectionResBlock(32, k_size=5, expansion=1),
                                 )

        self.conv3 =nn.Sequential(OneDirectionResBlock(32,k_size=5,expansion=2,stride=2),
                                 OneDirectionResBlock(64, k_size=5, expansion=1),
                                 )
        self.conv4 = nn.Sequential(OneDirectionResBlock(64,k_size=5,expansion=2,stride=2),
                                 OneDirectionResBlock(128, k_size=5, expansion=1),
                                 )

        self.conv5 = nn.Sequential(OneDirectionResBlock(128,k_size=5,expansion=2,stride=2),
                                 OneDirectionResBlock(256, k_size=5, expansion=1,stride=1),
                                 )
    def forward(self,data):



        x1=self.conv1(data)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        output=torch.mean(x5,dim=2)

        output = output.permute(0, 2, 1)
        output = output[:, :self.pre_length, ...]


        return output

class Complexer(nn.Module):
    def __init__(self,pre_length= cfg.MODEL.pre_length,mtype=0):
        super().__init__()

        self.pre_length=pre_length


        if mtype==0:
            self.data_model = GRU_model(pred_len=self.pre_length)
        elif mtype==1:
            self.data_model = LSTM_model(pred_len=self.pre_length)
        elif mtype==2:
            self.data_model = TRANSFORMER_model(pred_len=self.pre_length)
        elif mtype==3:
            self.data_model = LSTM_GRU_model(pred_len=self.pre_length)
        elif mtype==4:
            self.data_model = GRU_LSTM_model(pred_len=self.pre_length)



        self.image_model=nn.Sequential(ImageModel(pred_len=self.pre_length),
                                       nn.Dropout(0.5))


        self.fc=nn.Sequential(nn.Linear(512+256,5,bias=True))

    def forward(self,image,data):

        image_fm=self.image_model(image)

        rnn_fm=self.data_model(data)

        fm=torch.cat([image_fm,rnn_fm],dim=2)
        out=self.fc(fm)

        return out

if __name__=='__main__':
    model=Complexer()
    images=torch.zeros(size=[12,1,107,107])
    test_data=torch.zeros(size=[12,107,16])

    res=model(images,test_data)

    print(res.shape)