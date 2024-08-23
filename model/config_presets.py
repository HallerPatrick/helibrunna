xlstm_cfg_std = """ 
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: 4
slstm_block:
  slstm:
    backend: cuda
    num_heads: 4
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
  feedforward:
    proj_factor: 1.3
    act_fn: gelu
context_length: 512
num_blocks: 7
embedding_dim: 2048
slstm_at: []
"""

xlstm_cfg = """ 
mlstm_block:
  mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: 4
slstm_block:
  slstm:
    backend: cuda
    num_heads: 4
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
  feedforward:
    proj_factor: 1.3
    act_fn: gelu
context_length: 1024
num_blocks: 3
embedding_dim: 3072
slstm_at: [1]
"""

mlstm_only = """
num_blocks: 2
embedding_dim: 256
mlstm_block:
    mlstm:
      num_heads: 1
slstm_block: {}
slstm_at: []
context_length: 512
vocab_size: 0
"""

xlstm_cfg_map = {
    "350m": xlstm_cfg,
    "xlstm_std": xlstm_cfg_std,
    "mlstm_only": mlstm_only,
}
