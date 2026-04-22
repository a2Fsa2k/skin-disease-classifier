"""Dataset label constants."""

DX_TO_CLASS_NAME = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}

# The class order matches the legacy project checkpoints so the existing
# resnet18 weights remain usable after the cleanup.
DX_TO_INDEX = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "bcc": 3,
    "akiec": 4,
    "vasc": 5,
    "df": 6,
}

INDEX_TO_CLASS_NAME = [DX_TO_CLASS_NAME[key] for key, _ in sorted(DX_TO_INDEX.items(), key=lambda item: item[1])]
