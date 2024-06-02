import numpy as np
from cued_sf2_lab.jpeg import dwtgroup

index_array = np.arange(64)
index_array = index_array.reshape((8, 8))
print(index_array)
print(dwtgroup(index_array, 2))