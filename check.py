import numpy as np

data = np.load("captures/08_00/mask00.npz")

# List all arrays stored in the file
print(data.files)

# Inspect each array's shape, data type, and values
for key in data.files:
    arr = data[key]
    print(f"\nName: {key}")
    print(f"Shape: {arr.shape}")
    print(f"Data type: {arr.dtype}")
    print(arr)