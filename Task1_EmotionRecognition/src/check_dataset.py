import glob
import os

files = glob.glob("data/raw/**/*.wav", recursive=True)

print("Total audio files:", len(files))

if files:
    print("First audio file:")
    print(files[0])
else:
    print("No audio files found")