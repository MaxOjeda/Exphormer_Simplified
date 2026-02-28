"""
Standalone YACS config for Exphormer_Max (no graphgym dependency).
"""
from yacs.config import CfgNode as CN

cfg = CN()

cfg.out_dir = 'results'
cfg.metric_best = 'auto'
cfg.metric_agg = 'argmax'
cfg.seed = 0
cfg.device = 'auto'
cfg.num_threads = 6
cfg.run_multiple_splits = []
cfg.name_tag = ''
cfg.round = 5
cfg.run_id = 0
cfg.params = 0

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
cfg.dataset = CN()
cfg.dataset.format = 'PyG'
cfg.dataset.name = 'none'
cfg.dataset.task = 'graph'          # 'graph' or 'node'
cfg.dataset.task_type = 'classification'
cfg.dataset.transductive = False
cfg.dataset.split = [0.8, 0.1, 0.1]
cfg.dataset.split_mode = 'random'   # 'random', 'standard', 'cv'
cfg.dataset.split_index = 0
cfg.dataset.node_encoder = False
cfg.dataset.node_encoder_name = 'LinearNode'
cfg.dataset.node_encoder_bn = True
cfg.dataset.edge_encoder = False
cfg.dataset.edge_encoder_name = 'DummyEdge'
cfg.dataset.edge_encoder_bn = True
cfg.dataset.slic_compactness = 10
cfg.dataset.dir = './datasets'
cfg.dataset.node_encoder_num_types = 0
cfg.dataset.edge_encoder_num_types = 0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
cfg.model = CN()
cfg.model.type = 'MultiModel'
cfg.model.loss_fun = 'cross_entropy'
cfg.model.graph_pooling = 'mean'   # 'mean', 'max', 'add'
cfg.model.edge_decoding = 'dot'

# ---------------------------------------------------------------------------
# Graph Transformer (gt)
# ---------------------------------------------------------------------------
cfg.gt = CN()
cfg.gt.layer_type = 'CustomGatedGCN+Exphormer'
cfg.gt.layers = 3
cfg.gt.n_heads = 8
cfg.gt.dim_hidden = 64
cfg.gt.dropout = 0.0
cfg.gt.attn_dropout = 0.0
cfg.gt.layer_norm = False
cfg.gt.batch_norm = True
cfg.gt.dim_edge = None             # None → will be set equal to dim_hidden
cfg.gt.pna_degrees = []
cfg.gt.bigbird = CN()              # kept for compat but not used
cfg.gt.bigbird.attention_type = 'block_sparse'
cfg.gt.bigbird.chunk_size = 64
cfg.gt.bigbird.num_random_blocks = 1
cfg.gt.bigbird.num_sliding_window_blocks = 3

# ---------------------------------------------------------------------------
# GNN (post-processing)
# ---------------------------------------------------------------------------
cfg.gnn = CN()
cfg.gnn.head = 'default'           # 'default' (graph) or 'inductive_node'
cfg.gnn.layers_pre_mp = 0
cfg.gnn.layers_post_mp = 2
cfg.gnn.dim_inner = 64             # must match gt.dim_hidden
cfg.gnn.batchnorm = True
cfg.gnn.act = 'relu'
cfg.gnn.dropout = 0.0
cfg.gnn.agg = 'mean'
cfg.gnn.normalize_adj = False
cfg.gnn.layer_type = 'generalconv'
cfg.gnn.stage_type = 'stack'

# ---------------------------------------------------------------------------
# Preprocessing (expander edges, etc.)
# ---------------------------------------------------------------------------
cfg.prep = CN()
cfg.prep.exp = True
cfg.prep.exp_algorithm = 'Random-d'
cfg.prep.exp_deg = 5
cfg.prep.exp_count = 1
cfg.prep.exp_max_num_iters = 100
cfg.prep.add_edge_index = True
cfg.prep.num_virt_node = 0
cfg.prep.use_exp_edges = True
cfg.prep.add_self_loops = False
cfg.prep.add_reverse_edges = False
cfg.prep.train_percent = 0.6
cfg.prep.dist_enable = False
cfg.prep.dist_cutoff = None

# ---------------------------------------------------------------------------
# Positional Encodings
# ---------------------------------------------------------------------------

# LapPE
cfg.posenc_LapPE = CN()
cfg.posenc_LapPE.enable = False
cfg.posenc_LapPE.eigen = CN()
cfg.posenc_LapPE.eigen.laplacian_norm = 'none'
cfg.posenc_LapPE.eigen.eigvec_norm = 'L2'
cfg.posenc_LapPE.eigen.max_freqs = 10
cfg.posenc_LapPE.model = 'DeepSet'  # 'DeepSet' or 'Transformer'
cfg.posenc_LapPE.dim_pe = 16
cfg.posenc_LapPE.layers = 2
cfg.posenc_LapPE.n_heads = 4
cfg.posenc_LapPE.post_layers = 0
cfg.posenc_LapPE.raw_norm_type = 'none'
cfg.posenc_LapPE.pass_as_var = False

# EquivStableLapPE
cfg.posenc_EquivStableLapPE = CN()
cfg.posenc_EquivStableLapPE.enable = False
cfg.posenc_EquivStableLapPE.eigen = CN()
cfg.posenc_EquivStableLapPE.eigen.laplacian_norm = 'none'
cfg.posenc_EquivStableLapPE.eigen.eigvec_norm = 'L2'
cfg.posenc_EquivStableLapPE.eigen.max_freqs = 8
cfg.posenc_EquivStableLapPE.raw_norm_type = 'none'

# RWSE
cfg.posenc_RWSE = CN()
cfg.posenc_RWSE.enable = False
cfg.posenc_RWSE.kernel = CN()
cfg.posenc_RWSE.kernel.times_func = ''
cfg.posenc_RWSE.kernel.times = []
cfg.posenc_RWSE.model = 'Linear'
cfg.posenc_RWSE.dim_pe = 16
cfg.posenc_RWSE.raw_norm_type = 'none'

# ERN / ERE (effective resistance — kept minimal for compat)
cfg.posenc_ERN = CN()
cfg.posenc_ERN.enable = False
cfg.posenc_ERN.accuracy = 0.1
cfg.posenc_ERN.dim_pe = 8
cfg.posenc_ERN.model = 'Linear'
cfg.posenc_ERN.layers = 2

cfg.posenc_ERE = CN()
cfg.posenc_ERE.enable = False
cfg.posenc_ERE.dim_pe = 8
cfg.posenc_ERE.model = 'Linear'
cfg.posenc_ERE.layers = 2

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
cfg.optim = CN()
cfg.optim.optimizer = 'adamW'
cfg.optim.base_lr = 0.001
cfg.optim.weight_decay = 0.0
cfg.optim.momentum = 0.9
cfg.optim.max_epoch = 100
cfg.optim.scheduler = 'cosine_with_warmup'
cfg.optim.num_warmup_epochs = 10
cfg.optim.batch_accumulation = 1
cfg.optim.clip_grad_norm = False
cfg.optim.reduce_factor = 0.5
cfg.optim.schedule_patience = 10
cfg.optim.min_lr = 0.0
cfg.optim.steps = []
cfg.optim.lr_decay = 0.1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
cfg.train = CN()
cfg.train.mode = 'custom'
cfg.train.batch_size = 16
cfg.train.eval_period = 1
cfg.train.ckpt_period = 100
cfg.train.auto_resume = False
cfg.train.epoch_resume = -1
cfg.train.enable_ckpt = False
cfg.train.ckpt_best = False
cfg.train.ckpt_clean = True

# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------
cfg.wandb = CN()
cfg.wandb.use = False
cfg.wandb.entity = 'entity'
cfg.wandb.project = 'project'
cfg.wandb.name = ''

# ---------------------------------------------------------------------------
# Shared / internal (used by LapPE encoder for dim_in)
# ---------------------------------------------------------------------------
cfg.share = CN()
cfg.share.dim_in = 1       # set at runtime after dataset is loaded
cfg.share.num_splits = 3

# ---------------------------------------------------------------------------

def load_cfg(cfg_file, opts=None):
    """Load config from yaml file and merge optional list of overrides."""
    cfg.defrost()
    cfg.merge_from_file(cfg_file)
    if opts:
        cfg.merge_from_list(opts)
    # Resolve dim_edge default
    if cfg.gt.dim_edge is None:
        cfg.gt.dim_edge = cfg.gt.dim_hidden
    cfg.freeze()
    return cfg
