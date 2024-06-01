import numpy as np

# Correct initialization of X and scalar
X = np.array([[1, 1], [1, 1]], dtype=np.float64)
scalar = np.array([[1, 4], [3, 2]], dtype=np.float64)

X *= scalar

scalar_mean = np.mean(scalar)
norm_scalar = scalar /scalar_mean
print(norm_scalar)

k = 0.5
arr_centered = norm_scalar - 1.0
arr_shrunken = arr_centered * k
arr_new = arr_shrunken + 1.0
print(arr_new)

"""
ones_array = np.ones((2, 2))
differences_arr = ones_array - norm_scalar
print(differences_arr)
differences_arr *= 0.8

norm_scalar += differences_arr
print(norm_scalar)
"""