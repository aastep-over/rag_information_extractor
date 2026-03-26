import yaml 
import os
from pathlib import Path

# 1. Dynamically find the Project Root
# This file is in: src/BCP/utils/config_loader.py
# .parent    -> src/BCP/utils
# .parents[1] -> src/BCP
# .parents[2] -> src
# .parents[3] -> PROJECT_ROOT (Project_C4)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def load_config(cfg_path: str="config.yaml"):
    """
    Loads the YAML config and converts relative paths to absolute paths
    so they work on any OS without errors.
    """
    # 2. Define path to config.yaml
    CONFIG_PATH = PROJECT_ROOT / cfg_path
    
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"❌ Config file not found at: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # # 3. Helper: Convert relative paths in YAML to absolute System Paths
    # # This ensures "data/bloodmnist.npz" becomes "D:/Users/.../data/bloodmnist.npz"
    # for key, value in config['paths'].items():
    #     # Only convert if it looks like a path (simple string check)
    #     config['paths'][key] = str(PROJECT_ROOT / value) if value else None
    
    # config['paths']['root'] = str(PROJECT_ROOT)

    return config

# 3. Load configs in this file and import 'cfgs' in others
cfgs = load_config()


if __name__ == "__main__":
    import json
    from copy import deepcopy
    import re

    # cfgs_to_save = deepcopy(cfgs)
    # for k, v in cfgs_to_save['paths'].items():
    #     v = re.sub(re.escape(cfgs['paths']['root']), "", v) if v else ""
    #     while v and (v[0] in ('/', '\\')):
    #         v = v[1:]
    #     cfgs_to_save['paths'][k] = v
    # del cfgs_to_save['paths']['root']

    print(json.dumps(cfgs, indent=4))
    print("\n\n")
    # print(json.dumps(cfgs_to_save, indent=4))