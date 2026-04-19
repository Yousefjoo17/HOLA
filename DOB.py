import os

target_files = [
    r".\prime\data1.csv",
    r".\prime\data2.csv"
]

target_files = [os.path.normpath(p) for p in target_files]

for file in prime_files:
    if os.path.normpath(file) in target_files:
        print("MATCH:", file)