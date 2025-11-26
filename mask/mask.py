import argparse
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from blur import *

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root = os.path.join(args.root, "*")
    imgs = sorted(glob(root, recursive=True))
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    for img in tqdm(imgs):
        blur_map = get_blur_map_torch_unfold(img, win_size=5)
        ak = 1 - blur_map

        if np.any(np.isnan(ak)):
            import pdb; pdb.set_trace()
        if np.any(np.isinf(ak)):
            import pdb; pdb.set_trace()
            
        img = Path(img)
        save_name = os.path.join(save_path, str(img.name).replace('jpg','npy').replace('png','npy'))
        
        print(save_name)
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        np.save(save_name, ak.astype(np.float32))


if __name__ == "__main__":
    main()
