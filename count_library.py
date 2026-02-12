import os

path = "storm_data/library/Dead Internet Theory"
total = 0
valid = 0

for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith('.pdf'):
            total += 1
            fpath = os.path.join(root, f)
            if os.path.getsize(fpath) > 100000:
                valid += 1

print(f"Library Total PDFs: {total}")
print(f"Valid PDFs (>100KB): {valid}")
