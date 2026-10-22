# SPDX-License-Identifier: Apache-2.0
"""Frozen scene-level train/validation split.

Validation scenes are never used for data collection or training. Their teacher data is collected
once and reused every iteration, so validation numbers are comparable across iterations. Hold-outs
come from batches that no training scene shares. Change this only between experiment campaigns.
"""

TRAIN_SCENE_IDS = (
    "clipgt-01d503d4-449b-46fc-8d78-9085e70d3554",   # Batch0005 — straightforward
    "clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb",   # Batch0005 — labelled turning scene
    "clipgt-804afc4a-fd1e-4f58-bd39-a4c486a916e5",   # Batch0005 — mixed
    "clipgt-04db1bb2-90fe-47a7-9d09-27d020c618d4",   # Batch0001
    "clipgt-e36cf4c2-03a0-4af2-9146-48a5372efb3d",   # Batch0007
    "clipgt-19f339ba-4f3a-4025-b774-4b271e639f2a",   # Batch0009
)

VAL_SCENE_IDS = (
    "clipgt-1bccdc21-6e9c-425a-a0a1-392e2353f2b3",   # Batch0002 — different lighting/geometry
    "clipgt-6e190b33-73c2-4958-8448-db999b871a78",   # Batch0010 — maximum diversity
)
