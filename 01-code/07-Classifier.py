import numpy as np

# Load the document vectors
X = np.load("/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/00-testing/vsm1.npy",allow_pickle=True)  # Adjust path as needed

print("Shape of document vector matrix:", X.shape)
