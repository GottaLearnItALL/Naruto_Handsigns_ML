import os

for split in ["train", "test"]:
    print(f"\n--- {split} ---")
    for sign in sorted(os.listdir(f"data/{split}")):
        path = f"data/{split}/{sign}"
        if os.path.isdir(path):
            print(f"  {sign}: {len(os.listdir(path))} images")