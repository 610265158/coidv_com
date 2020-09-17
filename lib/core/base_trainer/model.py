import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter

from efficientnet_pytorch import EfficientNet
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
                                      nn.ReLU())

        self.decoder_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())

        self.decoder_3 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())



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
        print(fm_used.shape)

        decod1=self.decoder_1(fm_used)
        decod2 = self.decoder_2(decod1)
        decod3 = self.decoder_3(decod2)

        x=torch.mean(decod3,dim=2)
        x=torch.transpose(x,2,1)
        x=x[:,:self.pred_len,...]


        return x





token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}



class Simple1dNet(nn.Module):

    def __init__(self,embed_dim=96):
        super().__init__()

        # self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.encoder1=nn.Sequential(nn.Conv1d(in_channels=3,kernel_size=3,out_channels=128,stride=2),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                )

        self.encoder2 = nn.Sequential(nn.Conv1d(in_channels=128, kernel_size=3,out_channels=128, stride=2),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   )



        self.decoder1=nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128,kernel_size=3,padding=1),
                               nn.BatchNorm1d(128),
                               nn.ReLU(),
                                nn.Dropout(0.5))

        self.decoder2 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(128),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))


        self.head=nn.Conv1d(in_channels=128, out_channels=5, kernel_size=1, padding=0)

    def forward(self,inputs):


        # embed = self.embeding(inputs.long())
        #
        # reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))



        conv1=self.encoder1(inputs)
        conv2 = self.encoder2(conv1)

        inter=nn.functional.interpolate(conv2,size=(107),)

        decode1=self.decoder1(inter)
        #decode2 = self.decoder2(decode1)


        res = self.head(decode1)
        res=res[...,:68]
        return res

class GRU_model(nn.Module):
    def __init__(
        self, seq_length=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128, hidden_layers=3
    ):
        super(GRU_model, self).__init__()
        self.pre_length = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim * 3,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )


    def forward(self, seqs):

        seqs=seqs.long()
        embed = self.embeding(seqs)

        reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        output, hidden = self.gru(reshaped)
        output = output[:, :self.pre_length, ...]

        return output
class LSTM_model(nn.Module):
    def __init__(
        self, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128, hidden_layers=3
    ):
        super(LSTM_model, self).__init__()
        self.pred_len = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.gru = nn.LSTM(
            input_size=embed_dim * 3,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim * 2, 5)

    def forward(self, seqs):
        seqs = seqs.long()
        embed = self.embeding(seqs)
        reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        output, hidden = self.gru(reshaped)
        truncated = output[:, : self.pred_len, :]
        out = self.linear(truncated)
        return out





class Complexer(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.data_model=GRU_model()

        if cfg.MODEL.image or cfg.MODEL.image_only:
            self.image_model=Net()
            self.fc = nn.Linear(256+256, 5, bias=True)
        else:
            self.fc=nn.Linear(256,5,bias=True)


    def forward(self,images,data):

        if cfg.MODEL.image_only:
            image_fm = self.image_model(images)
            out = self.fc(image_fm)
            return out
        data_fm=self.data_model(data)
        print('data',data_fm.shape)
        if cfg.MODEL.image:
            image_fm = self.image_model(images)

            print('image_fm', image_fm.shape)
            fm=torch.cat([data_fm,image_fm],dim=2)

            out = self.fc(fm)

        else:
            out=self.fc(data_fm)



        return out

if __name__=='__main__':
    model=Complexer()

    image_data=torch.zeros(size=[12,1,107,107])
    test_data=torch.zeros(size=[12,107,3])

    res=model(image_data,test_data)

    print(res.shape)