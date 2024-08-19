import os

import numpy as np
import pandas as pd
from tqdm import tqdm

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path_data_ary = pd.read_csv(
    os.path.join(main_dir, "training_data", "link_path_dump.csv"),
    header=0,
    sep="\t",
).values

matidx_sum_arry = pd.read_csv(
    os.path.join(main_dir, "training_data", "matidx_sum_table.csv"),
    header=0,
    sep="\t",
)[["MatrixIndex", "Offset"]].values

offset_data_ary = np.full(path_data_ary.shape, -1).astype(np.int64)

s_loc, t_loc = 0, 0
for n, (s_idx, t_idx, step) in enumerate(tqdm(path_data_ary, desc="Map Offsets")):
    s_loc = np.searchsorted(matidx_sum_arry[:, 0], s_idx)
    t_loc = np.searchsorted(matidx_sum_arry[:, 0], t_idx)
    if s_loc >= matidx_sum_arry.shape[0] or t_loc >= matidx_sum_arry.shape[0]:
        continue
    if matidx_sum_arry[s_loc, 0] != s_idx and matidx_sum_arry[t_loc, 0] != t_idx:
        continue

    s_offset = matidx_sum_arry[s_loc, 1]
    t_offset = matidx_sum_arry[t_loc, 1]
    offset_data_ary[n] = (s_offset, t_offset, step)

offset_data_ary = offset_data_ary[offset_data_ary[:, 0] != -1]

pd.DataFrame(offset_data_ary, columns=["SourceOffset", "TargetOffset", "Step"]).to_csv(
    os.path.join(main_dir, "training_data", "link_path_offset.csv"),
    sep="\t",
    index=False,
)
