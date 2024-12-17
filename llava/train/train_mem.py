def mannual_seed(seed=0):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from transformers import set_seed
    set_seed(seed, True)
import os
if os.environ.get("SEED") is not None:
    mannual_seed(int(os.environ.get("SEED")))
else:
    mannual_seed()
from llava.train.train import train

if __name__ == "__main__":
    train()
