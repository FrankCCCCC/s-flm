"""Standalone OWT cache builder (run on a node with internet, no GPU needed).

Downloads jdeschena/openwebtext, GPT-2-tokenizes, groups into 1024-blocks, and
save_to_disk under data.cache_dir — so the multi-GPU training jobs read the cache
instead of holding GPUs idle during the hours-long prep.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import main  # noqa: F401  registers OmegaConf resolvers (device_count, div_up, eval)
import dataloader
from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base=None):
    cfg = compose(
        config_name="config",
        overrides=[
            "data=openwebtext-split",
            f"data.cache_dir={REPO}/data_cache",
            "model=small-hyperbolic-dit",   # only used for model.length=1024
            "algo=hflm",
            "loader.num_workers=32",
        ],
    )

tok = dataloader.get_tokenizer(cfg)
print(f"tokenizer={cfg.data.tokenizer_name_or_path} vocab={tok.vocab_size} "
      f"block_size={cfg.model.length} cache_dir={cfg.data.cache_dir}", flush=True)

for mode in ["valid", "train"]:   # valid first (small) to fail fast
    print(f"=== building {mode} cache ===", flush=True)
    ds = dataloader.get_dataset(cfg, tok, mode=mode)
    try:
        n = len(ds)
    except Exception:
        n = "?"
    print(f"=== {mode} cache READY: {n} examples ===", flush=True)

print("OWT CACHE PREP COMPLETE", flush=True)
