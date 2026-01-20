from datasets import load_from_disk

def inspect_disk_gsm8k(path, split="train", idx=78):
    ds = load_from_disk(path)

    print("splits:", list(ds.keys()))
    d = ds[split][idx]

    print("keys:", d.keys())
    print("question:", d.get("question"))
    print("answer:", d.get("answer"))

def inspect_disk_hendrycks_math(path, split="train", idx=6):
    ds = load_from_disk(path)

    print("splits:", list(ds.keys()))
    d = ds[split][idx]

    print("keys:", d.keys())
    print("problem:", d.get("problem"))
    print("level:", d.get("level"))
    print("type:",d.get("type"))
    print("solution:",d.get("solution"))

if __name__ == "__main__":
    inspect_disk_gsm8k("../datasets/gsm8k_arrow")
    inspect_disk_hendrycks_math("../datasets/hendrycks_math_arrow")