import torch
import torch.nn as nn

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
        self, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128, hidden_layers=3
    ):
        super(GRU_model, self).__init__()
        self.pred_len = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim * 3,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_dim * 2, 5)

    def forward(self, seqs):

        seqs=seqs.long()
        embed = self.embeding(seqs)

        reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        output, hidden = self.gru(reshaped)
        out = self.linear(output)
        out= out[:, : self.pred_len, :]
        return out

if __name__=='__main__':
    model=GRU_model()


    test_data=torch.zeros(size=[12,107,3])

    res=model(test_data)

    print(res.shape)