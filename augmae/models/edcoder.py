from typing import Optional
from itertools import chain
from functools import partial
import random
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .dot_gat import DotGAT
from .loss_func import sce_loss, neg_sce_loss ,uniformity_loss
from augmae.utils import create_norm, drop_edge


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim, 
            num_hidden=num_hidden, 
            out_dim=out_dim, 
            num_layers=num_layers, 
            dropout=dropout, 
            activation=activation, 
            residual=residual, 
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            uniformity_dim: int = 64
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
            self.uniformity_layer = nn.Linear(dec_in_dim * num_layers, uniformity_dim,bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
            self.uniformity_layer = nn.Linear(dec_in_dim, uniformity_dim, bias=False)

        # * setup loss function
        self.criterion, self.mask_criterion = self.setup_loss_fn(loss_fn, alpha_l)
    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
            mask_criterion = -nn.MSELoss()

        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
            mask_criterion = partial(neg_sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion, mask_criterion
    
    def encoding_mask_noise(self, g, x, args, mask_rate, mask_prob,epoch):

        num_nodes = g.num_nodes()
        alpha_adv = args.alpha_0 + ((epoch / args.max_epoch)**args.gamma) * (args.alpha_T - args.alpha_0)
        
        # random masking
        perm = torch.randperm(num_nodes, device=x.device)
        num_random_mask_nodes = int(mask_rate * num_nodes * (1.-alpha_adv))
        random_mask_nodes = perm[: num_random_mask_nodes]
        random_keep_nodes = perm[num_random_mask_nodes: ]

        # adversarial masking
        mask_ = mask_prob[:, 1]
        perm_adv = torch.randperm(num_nodes, device=x.device)
        adv_keep_token = perm_adv[:int(num_nodes*(1.-alpha_adv))]
        mask_[adv_keep_token] = 1.
        Mask_ = mask_.reshape(-1, 1)

        adv_keep_nodes =mask_.nonzero().reshape(-1)
        adv_mask_nodes = (1-mask_).nonzero().reshape(-1)

        mask_nodes = torch.cat((random_mask_nodes,adv_mask_nodes),dim=0).unique()
        keep_nodes = torch.tensor(np.intersect1d(random_keep_nodes.cpu().numpy(),adv_keep_nodes.cpu().numpy())).to(x.device)

        num_mask_nodes = mask_nodes.shape[0]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x = out_x * Mask_
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            out_x = out_x * Mask_
            token_nodes = mask_nodes
            out_x[token_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes),alpha_adv

    def forward(self, g, x,epoch,args,mask_prob,pooler):
        # ---- attribute reconstruction ----
        loss, loss_mask = self.mask_attr_prediction(g, x,epoch,args,mask_prob,pooler)
        loss_item = {"loss": loss.item()}
        return loss, loss_mask, loss_item
    
    def mask_attr_prediction(self, g, x,epoch,args,mask_prob,pooler):

        pre_use_g, use_x, (mask_nodes, keep_nodes),alpha_adv = self.encoding_mask_noise(g, x ,args,self._mask_rate,mask_prob,epoch)
        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0.0

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init_mask = x[mask_nodes]
        x_rec_mask = recon[mask_nodes]
        lamda = args.lamda*(1.0-alpha_adv)

        if args.dataset in ("IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "MUTAG","REDDIT-BINERY", "COLLAB"):
            sub_g = use_g.subgraph(keep_nodes)
            enc_rep_sub_g = enc_rep[keep_nodes]
            graph_emb = pooler(sub_g,enc_rep_sub_g)
            graph_emb = F.relu(self.uniformity_layer(graph_emb))
            u_loss = uniformity_loss(graph_emb,lamda)
        else:
            node_eb = F.relu(self.uniformity_layer(enc_rep))
            u_loss = uniformity_loss(node_eb,lamda)

        loss = self.criterion(x_rec_mask, x_init_mask) + u_loss
        num_all_noeds = mask_prob[:,1].sum() + mask_prob[:,0].sum()
        loss_mask = -self.mask_criterion(x_rec_mask, x_init_mask) + args.belta*(torch.tensor([1.]).to(g.device)/torch.sin(torch.pi/num_all_noeds*(mask_prob[:,0].sum())))

        return loss, loss_mask

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
