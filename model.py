import torch
import torch.nn as nn




token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
class Simple1dNet(nn.Module):

    def __init__(self,):
        super().__init__()




        self.embed=nn.Embedding()
        self.conv1=nn.Sequential(nn.Conv1d(in_channels=3,kernel_size=3,out_channels=128,padding=1,stride=2),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=128, kernel_size=3,out_channels=128,padding=1, stride=1),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())


        self.out=nn.Conv1d(in_channels=128,kernel_size=3,stride=2,padding=4, out_channels=5)


    def forward(self,inputs):


        torch.embedding()
        conv1=self.conv1(inputs)
        conv2 = self.conv2(conv1)
        res = self.out(conv2)

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
        self.gru2 = nn.GRU(
            input_size=embed_dim * 3,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.gru3 = nn.GRU(
            input_size=embed_dim * 3,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim * 2, 5)

    def forward(self, seqs):
        embed = self.embeding(seqs)
        reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        output, hidden = self.gru(reshaped)
        output, hidden = self.gru2(hidden)
        output, hidden = self.gru3(hidden)
        truncated = output[:, : self.pred_len, :]
        out = self.linear(truncated)

        return out

if __name__=='__main__':
    model=GRU_model()


    test_data=torch.zeros(size=[1,107,3]).long()

    res=model(test_data)

    print(res.shape)