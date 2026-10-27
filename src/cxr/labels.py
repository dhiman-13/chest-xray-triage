"""Label space of NIH ChestX-ray14 and the triage policy built on top of it."""

LABELS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(LABELS)

# Findings that should move a study to the top of the radiologist worklist when flagged.
URGENT_FINDINGS: tuple[str, ...] = (
    "Pneumothorax",
    "Effusion",
    "Pneumonia",
    "Edema",
    "Consolidation",
    "Mass",
)

TRIAGE_LEVELS = ("urgent", "review", "routine")


def pretty(label: str) -> str:
    return label.replace("_", " ")
