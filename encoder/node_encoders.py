"""
Node encoders for Exphormer_Max.
Merges: linear_node_encoder, equivstable_laplace_pos_encoder, laplace_pos_encoder, voc_superpixels_encoder.
All graphgym/cfg dependencies replaced by explicit arguments.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Linear node encoder
# ---------------------------------------------------------------------------

class LinearNodeEncoder(nn.Module):
    """Projects raw node features linearly to dim_emb."""

    def __init__(self, dim_in, dim_emb):
        super().__init__()
        self.encoder = nn.Linear(dim_in, dim_emb)

    def forward(self, batch):
        batch.x = self.encoder(batch.x.float())
        return batch


# ---------------------------------------------------------------------------
# VOC / COCO node encoder (14-dim node features)
# ---------------------------------------------------------------------------

VOC_NODE_INPUT_DIM = 14


class VOCNodeEncoder(nn.Module):
    def __init__(self, dim_emb):
        super().__init__()
        self.encoder = nn.Linear(VOC_NODE_INPUT_DIM, dim_emb)

    def forward(self, batch):
        batch.x = self.encoder(batch.x.float())
        return batch


# ---------------------------------------------------------------------------
# EquivStable LapPE encoder
# ---------------------------------------------------------------------------

class EquivStableLapPENodeEncoder(nn.Module):
    """
    Transforms precomputed k-dim LapPE eigenvectors to dim_emb.
    Stores result as batch.pe_EquivStableLapPE (used by GatedGCN).
    Does NOT concatenate to batch.x.
    """

    def __init__(self, max_freqs, dim_emb, norm_type='none'):
        super().__init__()
        if norm_type.lower() == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None
        self.linear_encoder_eigenvec = nn.Linear(max_freqs, dim_emb)

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("EigVals/EigVecs not found in batch. "
                             "Set posenc_EquivStableLapPE.enable = True.")
        pos_enc = batch.EigVecs.float()
        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0.0

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_encoder_eigenvec(pos_enc)
        batch.pe_EquivStableLapPE = pos_enc
        return batch


# ---------------------------------------------------------------------------
# LapPE encoder (DeepSet or Transformer)
# ---------------------------------------------------------------------------

class LapPENodeEncoder(nn.Module):
    """
    Laplacian Positional Encoding node encoder.
    Appends dim_pe to existing node features (x is expanded linearly).
    """

    def __init__(self, dim_in, dim_emb, dim_pe, max_freqs,
                 model_type='DeepSet', n_layers=2, n_heads=4,
                 post_n_layers=0, norm_type='none',
                 pass_as_var=False, expand_x=True):
        super().__init__()
        self.model_type = model_type
        self.pass_as_var = pass_as_var
        self.expand_x = expand_x

        if dim_emb - dim_pe < 1:
            raise ValueError(f"LapPE size {dim_pe} is too large for embedding size {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)

        # Project (eigvec_i, eigenval_i) → dim_pe
        if model_type == 'Transformer':
            self.linear_A = nn.Linear(2, dim_pe)
        else:
            # DeepSet
            if n_layers == 1:
                self.linear_A = nn.Linear(2, dim_pe)
            else:
                self.linear_A = nn.Linear(2, 2 * dim_pe)

        if norm_type.lower() == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        if model_type == 'Transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim_pe, nhead=n_heads, batch_first=True)
            self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            # DeepSet MLP
            layers = []
            if n_layers == 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.ReLU())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(nn.ReLU())
            self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            layers = []
            if post_n_layers == 1:
                layers.extend([nn.Linear(dim_pe, dim_pe), nn.ReLU()])
            else:
                layers.extend([nn.Linear(dim_pe, 2 * dim_pe), nn.ReLU()])
                for _ in range(post_n_layers - 2):
                    layers.extend([nn.Linear(2 * dim_pe, 2 * dim_pe), nn.ReLU()])
                layers.extend([nn.Linear(2 * dim_pe, dim_pe), nn.ReLU()])
            self.post_mlp = nn.Sequential(*layers)

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("EigVals/EigVecs not found. Enable posenc_LapPE.")
        EigVals = batch.EigVals.float()
        EigVecs = batch.EigVecs.float()

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2)  # (N, k, 2)
        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0.0

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_A(pos_enc)  # (N, k, dim_pe)

        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc,
                                      src_key_padding_mask=empty_mask[:, :, 0])
        else:
            pos_enc = self.pe_encoder(pos_enc)

        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2), 0.0)
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (N, dim_pe)

        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)

        if self.expand_x:
            h = self.linear_x(batch.x.float())
        else:
            h = batch.x.float()
        batch.x = torch.cat((h, pos_enc), 1)

        if self.pass_as_var:
            batch.pe_LapPE = pos_enc
        return batch


# ---------------------------------------------------------------------------
# Combined encoders (factory)
# ---------------------------------------------------------------------------

def build_node_encoder(cfg, dim_in):
    """
    Return the correct node encoder given cfg.dataset.node_encoder_name.
    This replaces graphgym's register.node_encoder_dict lookup.
    """
    name = cfg.dataset.node_encoder_name
    dim_h = cfg.gnn.dim_inner

    if name == 'LinearNode':
        return LinearNodeEncoder(dim_in, dim_h)

    elif name == 'LinearNode+EquivStableLapPE':
        # Two-step encoder: (1) Linear projection of raw features, (2) EquivStableLapPE
        pecfg = cfg.posenc_EquivStableLapPE
        max_freqs = pecfg.eigen.max_freqs
        norm_type = pecfg.raw_norm_type

        class _CombinedLinearEquiv(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_enc = LinearNodeEncoder(dim_in, dim_h)
                self.pe_enc = EquivStableLapPENodeEncoder(max_freqs, dim_h, norm_type)

            def forward(self, batch):
                batch = self.linear_enc(batch)
                batch = self.pe_enc(batch)
                return batch

        return _CombinedLinearEquiv()

    elif name == 'VOCNode':
        return VOCNodeEncoder(dim_h)

    elif name == 'VOCNode+LapPE':
        pecfg = cfg.posenc_LapPE
        dim_pe = pecfg.dim_pe
        max_freqs = pecfg.eigen.max_freqs
        model_type = pecfg.model
        n_layers = pecfg.layers
        n_heads = pecfg.n_heads
        post_n_layers = pecfg.post_layers
        norm_type = pecfg.raw_norm_type
        pass_as_var = pecfg.pass_as_var

        class _CombinedVOCLapPE(nn.Module):
            def __init__(self):
                super().__init__()
                # VOCNode projects 14 → (dim_h - dim_pe)
                self.node_enc = nn.Linear(VOC_NODE_INPUT_DIM, dim_h - dim_pe)
                self.pe_enc = LapPENodeEncoder(
                    dim_in=dim_h - dim_pe, dim_emb=dim_h, dim_pe=dim_pe,
                    max_freqs=max_freqs, model_type=model_type,
                    n_layers=n_layers, n_heads=n_heads,
                    post_n_layers=post_n_layers, norm_type=norm_type,
                    pass_as_var=pass_as_var, expand_x=True)

            def forward(self, batch):
                batch.x = self.node_enc(batch.x.float())
                # LapPE encoder expects batch.x to already be (dim_h - dim_pe)
                # it will expand to dim_h internally
                batch = self.pe_enc(batch)
                return batch

        return _CombinedVOCLapPE()

    else:
        raise ValueError(f"Unknown node encoder: '{name}'. "
                         f"Supported: LinearNode, LinearNode+EquivStableLapPE, "
                         f"VOCNode, VOCNode+LapPE")
