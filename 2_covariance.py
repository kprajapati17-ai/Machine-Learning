import numpy as np

height = np.array([4.5, 5.0, 5.5, 6.0, 6.5])
weight = np.array([55, 60, 65, 70, 80])

# Mean
height_mean = np.mean(height)
weight_mean = np.mean(weight)

# Variance
height_variance = np.var(height)
weight_variance = np.var(weight)

# Covariance
covariance = np.cov(height, weight)[0, 1]

print("Height Mean:", height_mean)
print("Weight Mean:", weight_mean)

print("Height Variance:", height_variance)
print("Weight Variance:", weight_variance)

print("Height-Weight Covariance:", covariance)