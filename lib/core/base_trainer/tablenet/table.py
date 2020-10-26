

import torch
import torch.nn as nn

from lib.core.base_trainer.tablenet import tab_network
class Tableplexe(nn.Module):

    def __init__(self, ):
        super(Tableplexe, self).__init__()

        self.n_d: int = 32
        self.n_a: int = 32
        self.n_steps: int = 3
        self.gamma: float = 1.3
        self.cat_idxs: []=[]
        self.cat_dims: []=[]
        self.cat_emb_dim: int = 1
        self.n_independent: int = 2
        self.n_shared: int = 2
        self.epsilon: float = 1e-15
        self.momentum: float = 0.02
        self.lambda_sparse: float = 1e-3

        self.clip_value: int = 1
        self.verbose: int = 1

        self.mask_type: str = "sparsemax"
        self.input_dim: int = 875
        self.output_dim: int = 512
        self.device_name: str = "auto"
        self.virtual_batch_size=32

        self.network = tab_network.TabNet(
                    self.input_dim,
                    self.output_dim,
                    n_d=self.n_d,
                    n_a=self.n_a,
                    n_steps=self.n_steps,
                    gamma=self.gamma,
                    cat_idxs=self.cat_idxs,
                    cat_dims=self.cat_dims,
                    cat_emb_dim=self.cat_emb_dim,
                    n_independent=self.n_independent,
                    n_shared=self.n_shared,
                    epsilon=self.epsilon,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=self.momentum,
                    device_name=self.device_name,
                    mask_type=self.mask_type,
                )
    def forward(self, x):

        x,model_loss = self.network(x)

        return x,model_loss