import numpy as np
from ksvd import ApproximateKSVD
import torch
import multiprocessing

# X ~ gamma.dot(dictionary)
Y = np.random.randn(10, 2)
D = np.random.randn(2, 2)
aksvd = ApproximateKSVD(n_components=2)
dictionary = aksvd.fit(Y, D).components_
X = aksvd.transform(Y)

print(Y.shape, X.shape, D.shape)
# Y = X * D

def main():
    pass

p = multiprocessing.pool.ThreadPool(processes=1)
p.map(main(), [])
