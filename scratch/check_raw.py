import os

raw_dir = r"d:\Ai Recommendation System\datasets"
print("Raw files list:", os.listdir(raw_dir) if os.path.exists(raw_dir) else "Directory does not exist")

edges_path = os.path.join(raw_dir, "rec-libimseti-dir.edges")
if os.path.exists(edges_path):
    print("First 20 lines of rec-libimseti-dir.edges:")
    with open(edges_path, 'r') as f:
        for _ in range(20):
            line = f.readline()
            if not line:
                break
            print(line.strip())
else:
    print("rec-libimseti-dir.edges does not exist")
