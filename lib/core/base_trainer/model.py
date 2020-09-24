import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MemoryEfficientSwish
from train_config import config as cfg


def gem(x, p=3, eps=1e-5):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-5):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Net(nn.Module):
    def __init__(self, pred_len=68):
        super().__init__()

        # self.mean_tensor=torch.from_numpy(cfg.DATA.PIXEL_MEAN ).float().cuda()
        # self.std_val_tensor = torch.from_numpy(cfg.DATA.PIXEL_STD).float().cuda()
        self.model = EfficientNet.from_pretrained(model_name='efficientnet-b1')
        # self.model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)


        self.decoder_1=nn.Sequential(nn.ConvTranspose2d(40,128,kernel_size=4,stride=2,padding=1),
                                      nn.BatchNorm2d(128),
                                      MemoryEfficientSwish())

        self.decoder_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(256),
                                       MemoryEfficientSwish())

        self.decoder_3 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(256),
                                       MemoryEfficientSwish())



        self.pred_len=pred_len

    def forward(self, inputs):

        inputs=torch.cat([inputs,inputs,inputs],dim=1)
        #do preprocess
        bs = inputs.size(0)
        # Convolution layers
        x,fms = self.model.extract_features(inputs)



        # for k,item in enumerate(fms):
        #
        #     print(k,item.shape)



        fm_used=fms[8]
        decod1=self.decoder_1(fm_used)
        decod2 = self.decoder_2(decod1)
        decod3 = self.decoder_3(decod2)

        x=torch.mean(decod3,dim=2)
        x=torch.transpose(x,2,1)
        x=x[:,:self.pred_len,...]


        return x





token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}



class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]

        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res

class Attention(nn.Module):
    def __init__(self, input_dim=384,refraction=4):
        super(Attention, self).__init__()

        self.conv=nn.Sequential(nn.Conv1d(in_channels=input_dim, kernel_size=1, out_channels=input_dim//refraction,
                                              stride=1),
                                nn.BatchNorm1d(input_dim//refraction,momentum=0.01),
                                nn.Conv1d(in_channels=input_dim//refraction, kernel_size=1, out_channels=input_dim,
                                          stride=1),
                                nn.BatchNorm1d(input_dim, momentum=0.01),
                                nn.Sigmoid()
                                )


    def forward(self,x):


        attention=self.conv(x)

        return x*attention


class GRU_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.4, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(GRU_model, self).__init__()
        self.pre_length = pred_len

        # self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)

        self.preconv = nn.Sequential( nn.Conv1d(in_channels=14+3,
                                                out_channels=384,
                                                kernel_size=5,

                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(384,momentum=0.01),
                                    MemoryEfficientSwish(),
                                      )

        self.gru = nn.GRU(
            input_size=embed_dim * 3,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )


        self.post_conv=nn.Sequential( nn.Conv1d(in_channels=512, kernel_size=5, out_channels=256,
                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(256,momentum=0.01),
                                    MemoryEfficientSwish(),
                                    Attention(256),
                                    nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                stride=1,
                                                padding=2, bias=False),
                                    nn.BatchNorm1d(256, momentum=0.01),
                                    MemoryEfficientSwish(),
                                      )

    def forward(self, seqs):
        seqs = seqs.permute(0, 2, 1)

        cnn_embes = self.preconv(seqs)

        cnn_embes = cnn_embes.permute(0, 2, 1)

        output, hidden = self.gru(cnn_embes)

        output = output.permute(0, 2, 1)

        output = self.post_conv(output)

        output = output.permute(0, 2, 1)
        output = output[:, :self.pre_length, ...]

        return output

class LSTM_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.4, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(LSTM_model, self).__init__()
        self.pre_length = pred_len

        self.preconv = nn.Sequential(nn.Conv1d(in_channels=14 + 3,
                                               out_channels=384,
                                               kernel_size=5,

                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(384, momentum=0.01),
                                     MemoryEfficientSwish(),
                                     )

        self.gru = nn.LSTM(
            input_size=embed_dim * 3+3,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )


        self.post_conv=nn.Sequential( nn.Conv1d(in_channels=512, kernel_size=5, out_channels=256,
                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(256,momentum=0.01),
                                    MemoryEfficientSwish(),
                                    Attention(256),
                                    nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                stride=1,
                                                padding=2, bias=False),
                                    nn.BatchNorm1d(256, momentum=0.01),
                                    MemoryEfficientSwish(),
                                      )

    def forward(self, seqs):
        seqs = seqs.permute(0, 2, 1)

        cnn_embes = self.preconv(seqs)

        cnn_embes = cnn_embes.permute(0, 2, 1)

        output, hidden = self.gru(cnn_embes)

        output = output.permute(0, 2, 1)

        output = self.post_conv(output)

        output = output.permute(0, 2, 1)
        output = output[:, :self.pre_length, ...]

        return output

class LSTM_GRU_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.4, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(LSTM_GRU_model, self).__init__()
        self.pre_length = pred_len

        self.preconv = nn.Sequential(nn.Conv1d(in_channels=14 + 3,
                                               out_channels=384,
                                               kernel_size=5,

                                               stride=1,
                                               padding=2, bias=False),
                                     nn.BatchNorm1d(384, momentum=0.01),
                                     MemoryEfficientSwish(),
                                     )

        self.lstm = nn.LSTM(
            input_size=embed_dim * 3+3,
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

        self.post_conv=nn.Sequential( nn.Conv1d(in_channels=512, kernel_size=5, out_channels=256,
                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(256,momentum=0.01),
                                    MemoryEfficientSwish(),
                                    Attention(256),
                                    nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                stride=1,
                                                padding=2, bias=False),
                                    nn.BatchNorm1d(256, momentum=0.01),
                                    MemoryEfficientSwish(),
                                      )

    def forward(self, seqs):
        seqs = seqs.permute(0, 2, 1)

        cnn_embes = self.preconv(seqs)

        cnn_embes = cnn_embes.permute(0, 2, 1)


        output, hidden = self.lstm(cnn_embes)
        output, hidden = self.gru(output)

        output = output.permute(0, 2, 1)

        output = self.post_conv(output)

        output = output.permute(0, 2, 1)
        output = output[:, :self.pre_length, ...]

        return output

class GRU_LSTM_model(nn.Module):
    def __init__(
            self, seq_length=107, pred_len=68, dropout=0.4, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(GRU_LSTM_model, self).__init__()
        self.pre_length = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)

        # self.preconv = nn.Sequential( nn.Conv1d(in_channels=3, kernel_size=5, out_channels=100,
        #                                       stride=1,
        #                                       padding=2,bias=False),
        #                             nn.BatchNorm1d(384,momentum=0.01),
        #                             MemoryEfficientSwish(),
        #                               )

        self.gru = nn.GRU(
            input_size=embed_dim * 3 + 3,
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

        self.post_conv = nn.Sequential(nn.Conv1d(in_channels=512, kernel_size=5, out_channels=256,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(256, momentum=0.01),
                                       MemoryEfficientSwish(),
                                       Attention(256),
                                       nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(256, momentum=0.01),
                                       MemoryEfficientSwish(),
                                       )

    def forward(self, seqs):
        seqs_base = seqs[:, :, 0:3].long()

        seqs_extra_fea = seqs[:, :, 3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

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
        self, seq_length=107, pred_len=68, dropout=0.4, embed_dim=128, hidden_dim=256, hidden_layers=2):
        super(TRANSFORMER_model, self).__init__()

        self.pre_length = pred_len
        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)

        self.preconv=nn.Sequential( nn.Conv1d(in_channels=384+3, kernel_size=5, out_channels=512,
                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(512,momentum=0.01),
                                    MemoryEfficientSwish())
        self.gru = nn.GRU(
            input_size=embed_dim * 3 + 3,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(512, 64, 1024, dropout=0.2, activation='relu'),
                                            3)

        self.post_conv = nn.Sequential(nn.Conv1d(in_channels=512, kernel_size=5, out_channels=256,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(256, momentum=0.01),
                                       MemoryEfficientSwish(),
                                       Attention(256),
                                       nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(256, momentum=0.01),
                                       MemoryEfficientSwish(),
                                       )
    def forward(self, seqs):

        seqs_base=seqs[:,:,0:3].long()

        seqs_extra_fea=seqs[:,:,3:]

        embed = self.embeding(seqs_base)

        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        feature=torch.cat([embed_reshaped,seqs_extra_fea],dim=-1)
        output, hidden = self.gru(feature)

        encoded = self.encoder(output)

        encoded = encoded.permute(0, 2, 1)
        output=self.post_conv(encoded)
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

        self.fc=nn.Linear(256,5,bias=True)

    def forward(self,data):

        fm=self.data_model(data)
        out=self.fc(fm)

        return out

if __name__=='__main__':
    model=LSTM_GRU_model()

    test_data=torch.zeros(size=[12,107,6])

    res=model(test_data)

    print(res.shape)