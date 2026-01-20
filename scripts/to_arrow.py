from datasets import load_dataset, concatenate_datasets, DatasetDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets"

CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

train_parts = [load_dataset("EleutherAI/hendrycks_math", c, split="train") for c in CONFIGS]
test_parts  = [load_dataset("EleutherAI/hendrycks_math", c, split="test")  for c in CONFIGS]

ds = DatasetDict({
    "train": concatenate_datasets(train_parts),
    "test":  concatenate_datasets(test_parts),
})

save_path = OUT / "hendrycks_math_arrow"
ds.save_to_disk(str(save_path))

print("Saved to", save_path)
print(ds)
