import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MemoryEfficientSwish
from train_config import config as cfg



ACT_FUNCTION=MemoryEfficientSwish
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}




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

class InceptionBlock(nn.Module):
    def __init__(self, input_dim=512,output_dim=512):
        super(InceptionBlock, self).__init__()

        self.conv1=nn.Sequential(nn.Conv1d(in_channels=input_dim, kernel_size=1, out_channels=output_dim//4,
                                              stride=1),
                                nn.BatchNorm1d(output_dim//4,momentum=0.01),

                                )

        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=input_dim, kernel_size=1, out_channels=output_dim // 4,
                                               stride=1,
                                             padding=2),
                                     nn.BatchNorm1d(output_dim // 4, momentum=0.01),

        )
        self.conv5 = nn.Sequential(nn.Conv1d(in_channels=input_dim, kernel_size=1, out_channels=output_dim // 4,
                                               stride=1,padding=3),
                                     nn.BatchNorm1d(output_dim // 4, momentum=0.01),

        )
        self.conv7 = nn.Sequential(nn.Conv1d(in_channels=input_dim, kernel_size=1, out_channels=output_dim // 4,
                                               stride=1,padding=4),
                                     nn.BatchNorm1d(output_dim // 4, momentum=0.01),

        )


        self.act=ACT_FUNCTION()
    def forward(self,x):


        eye1 = self.conv1(x)
        eye2 = self.conv2(x)
        eye3 = self.conv3(x)
        eye4 = self.conv4(x)


        x=torch.cat([eye1,eye2,eye3,eye4],dim=1)
        x=self.act(x)
        return x

class GRU_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.3, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(GRU_model, self).__init__()
        self.pre_length = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)

        self.preconv = nn.Sequential( nn.Conv1d(in_channels=3, kernel_size=5, out_channels=128,
                                              stride=1,
                                              padding=2,bias=False),
                                    nn.BatchNorm1d(128,momentum=0.01),
                                    MemoryEfficientSwish(),
                                    nn.Dropout(0.3)
                                      )

        self.gru = nn.GRU(
            input_size=embed_dim * 3+128,
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
                                    ACT_FUNCTION(),
                                    nn.Dropout(0.3),
                                    nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                stride=1,
                                                padding=2, bias=False),
                                    nn.BatchNorm1d(256, momentum=0.01),
                                    ACT_FUNCTION(),
                                    nn.Dropout(0.3)
                                    )

    def forward(self, seqs):


        seqs_base=seqs[:,:,0:3].long()

        seqs_extra_fea=seqs[:,:,3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))



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
        self, seq_length=107, pred_len=68, dropout=0.4, embed_dim=128, hidden_dim=256, hidden_layers=3
    ):
        super(LSTM_model, self).__init__()
        self.pre_length = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)

        # self.preconv = nn.Sequential( nn.Conv1d(in_channels=3, kernel_size=5, out_channels=100,
        #                                       stride=1,
        #                                       padding=2,bias=False),
        #                             nn.BatchNorm1d(384,momentum=0.01),
        #                             MemoryEfficientSwish(),
        #                               )

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
                                    ACT_FUNCTION(),
                                    Attention(256),
                                    nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                stride=1,
                                                padding=2, bias=False),
                                    nn.BatchNorm1d(256, momentum=0.01),
                                    ACT_FUNCTION(),
                                      )

    def forward(self, seqs):


        seqs_base=seqs[:,:,0:3].long()

        seqs_extra_fea=seqs[:,:,3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

        feature=torch.cat([embed_reshaped,seqs_extra_fea],dim=-1)


        output, hidden = self.gru(feature)

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

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)

        # self.preconv = nn.Sequential( nn.Conv1d(in_channels=3, kernel_size=5, out_channels=100,
        #                                       stride=1,
        #                                       padding=2,bias=False),
        #                             nn.BatchNorm1d(384,momentum=0.01),
        #                             MemoryEfficientSwish(),
        #                               )

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
                                    ACT_FUNCTION(),
                                    Attention(256),
                                    nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                stride=1,
                                                padding=2, bias=False),
                                    nn.BatchNorm1d(256, momentum=0.01),
                                    ACT_FUNCTION(),
                                      )

    def forward(self, seqs):


        seqs_base=seqs[:,:,0:3].long()

        seqs_extra_fea=seqs[:,:,3:]

        embed = self.embeding(seqs_base)
        embed_reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

        feature=torch.cat([embed_reshaped,seqs_extra_fea],dim=-1)


        output, hidden = self.lstm(feature)
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
                                       ACT_FUNCTION(),
                                       Attention(256),
                                       nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(256, momentum=0.01),
                                       ACT_FUNCTION(),
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
                                    ACT_FUNCTION())
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
                                       ACT_FUNCTION(),
                                       Attention(256),
                                       nn.Conv1d(in_channels=256, kernel_size=5, out_channels=256,
                                                 stride=1,
                                                 padding=2, bias=False),
                                       nn.BatchNorm1d(256, momentum=0.01),
                                       ACT_FUNCTION(),
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

        self.fc=nn.Sequential(nn.Dropout(0.3),
                              nn.Linear(256,5,bias=True))

    def forward(self,data):

        fm=self.data_model(data)
        out=self.fc(fm)

        return out

if __name__=='__main__':
    model=LSTM_GRU_model()

    test_data=torch.zeros(size=[12,107,6])

    res=model(test_data)

    print(res.shape)