"""Published per-class test AUROCs on ChestX-ray14, for context in the results report.

Caveats you should state when presenting:
- Wang et al. (2017) used the official patient-wise test split, the same one we use.
- CheXNet (Rajpurkar et al., 2017) used its own random patient split, so it is NOT directly
  comparable; later work re-evaluating on the official split reports macro AUROC ~0.80-0.82.
Double-check the numbers against the original papers before quoting them.
"""

WANG_2017 = {
    "Atelectasis": 0.716, "Cardiomegaly": 0.807, "Effusion": 0.784, "Infiltration": 0.609,
    "Mass": 0.706, "Nodule": 0.671, "Pneumonia": 0.633, "Pneumothorax": 0.806,
    "Consolidation": 0.708, "Edema": 0.835, "Emphysema": 0.815, "Fibrosis": 0.769,
    "Pleural_Thickening": 0.708, "Hernia": 0.767,
}  # fmt: skip

CHEXNET_2017 = {
    "Atelectasis": 0.8094, "Cardiomegaly": 0.9248, "Effusion": 0.8638, "Infiltration": 0.7345,
    "Mass": 0.8676, "Nodule": 0.7802, "Pneumonia": 0.7680, "Pneumothorax": 0.8887,
    "Consolidation": 0.7901, "Edema": 0.8878, "Emphysema": 0.9371, "Fibrosis": 0.8047,
    "Pleural_Thickening": 0.8062, "Hernia": 0.9164,
}  # fmt: skip

BASELINES = {"Wang et al. 2017": WANG_2017, "CheXNet 2017 (different split)": CHEXNET_2017}
