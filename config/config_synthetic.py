import sys
sys.path.append('../repos/gmcd/src')
from run_config import RunConfig

class PlanarRunConfig(RunConfig):
    def __init__(self,
                 dataset,
                 S, 
                 K) -> None:
        super().__init__()
        self.S = S
        self.K = K
       
        self.eval_freq = 500
        self.dataset = dataset

        self.T = 10
        self.diffusion_steps = self.T
        self.batch_size = 16
        self.encoding_dim = 2
        self.max_iterations = 1000
        self.transformer_dim = 64
        self.input_dp_rate = 0.2
        self.transformer_heads = 8
        self.transformer_depth = 2
        self.transformer_blocks = 1
        self.transformer_local_heads = 4
        self.transformer_local_size = 64

class SBMRunConfig(RunConfig):
    def __init__(self,
                 dataset,
                 S, 
                 K) -> None:
        super().__init__()
        self.S = S
        self.K = K
       
        self.eval_freq = 500
        self.dataset = dataset

        self.T = 10
        self.diffusion_steps = self.T
        self.batch_size = 16
        self.encoding_dim = 2
        self.max_iterations = 1000
        self.transformer_dim = 64
        self.input_dp_rate = 0.2
        self.transformer_heads = 8
        self.transformer_depth = 2
        self.transformer_blocks = 1
        self.transformer_local_heads = 4
        self.transformer_local_size = 64