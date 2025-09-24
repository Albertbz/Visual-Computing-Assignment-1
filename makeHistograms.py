import numpy as np
import matplotlib.pyplot as plt

siftDistances = np.loadtxt("build/Distances SIFT.txt")

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(siftDistances, bins=50, edgecolor="black")
plt.title("Histogram of SIFT Match Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)


orbDistances = np.loadtxt("build/Distances ORB.txt")

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(orbDistances, bins=50, edgecolor="black")
plt.title("Histogram of ORB Match Distances")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
