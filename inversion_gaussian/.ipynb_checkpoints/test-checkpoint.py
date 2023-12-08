import numpy as np


mat=  np.zeros((10,10))

np.save('mat.npy',mat)

mat = mat.reshape((100,1))

np.savetxt('mat.txt', mat)