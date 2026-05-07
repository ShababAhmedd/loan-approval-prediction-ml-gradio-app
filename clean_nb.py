import nbformat
import sys

file = sys.argv[1]

with open(file, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# remove widget metadata if present
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

with open(file, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Cleaned:", file)
