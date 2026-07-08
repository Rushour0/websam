"""Dump the checkpoint constants the JS engine ships as small binary blobs.

* tpos_table.npy (7, 64) — memory_temporal_positional_encoding: row k-1 is
  added to a recent memory map at temporal offset k; row 6 (HF's index
  `offset 0 - 1 = -1`) to conditioning maps (modeling_edgetam_video.py
  L2663-2667, param defined L2040-2042).
"""

import pathlib

import numpy as np
import torch
from transformers import EdgeTamVideoModel

HERE = pathlib.Path(__file__).parent
model = EdgeTamVideoModel.from_pretrained("yonigozlan/EdgeTAM-hf", dtype=torch.float32)
tpos = model.memory_temporal_positional_encoding.detach().numpy()  # (7,1,1,64)
np.save(HERE / "activations" / "tpos_table.npy", tpos.reshape(7, 64))
print("wrote", HERE / "activations" / "tpos_table.npy", tpos.shape)
