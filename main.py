import numpy as np
from typing import List
from matplotlib import pyplot as plt
import scipy


def inverseParticipationRatio(states: List[np.array]) -> float:
    return np.sum(np.abs(states**4), axis=0)


# Notation matches https://arxiv.org/pdf/1109.2210.pdf
def treeMatrix(K: int, t: complex, l: float, depth: int, disorderDistribution='uniform') -> np.array:
    numOfSites = int((K**depth - 1) / (K - 1))
    tree = np.zeros((numOfSites, numOfSites), dtype=complex)
    innerNodes = range(numOfSites - K**(depth - 1))
    for n in innerNodes:
        for child in range(K * n + 1, K * (n+1) +1):
            tree[n, child] = t
            tree[child, n] = np.conj(t)
    leaves = list(range(numOfSites - K**(depth - 1), numOfSites))
    for n in leaves:
        neighbor = leaves[np.random.randint(0, len(leaves))]
        tree[n, neighbor] = t
        tree[neighbor, n] = np.conj(t)
    for n in list(innerNodes) + leaves:
        if disorderDistribution == 'uniform':
            tree[n, n] = l * np.random.uniform(low=-1, high=1)
    withIngoingBrach = np.zeros((depth**2 + numOfSites, depth**2 + numOfSites), dtype=complex)
    withIngoingBrach[depth**2:, depth**2:] = tree
    for i in range(depth**2):
        withIngoingBrach[i, i+1] = t
        withIngoingBrach[i+1, i] = np.conj(t)
    return withIngoingBrach


def showTree(vec, K, depth):
    amplitudes = [0] * depth
    for d in range(depth):
        amplitudes[d] = sum(vec[int((K**d - 1) / (K - 1)) : int((K**(d+1) - 1) / (K - 1))]**2)
    plt.plot(list(range(depth)), amplitudes)



depths = [7, 8, 9, 10, 11, 12]
K = 2
l = 0
t = 1
iprs = []
Es = []
testE = 0.3
for depth in depths:
    m = treeMatrix(K, t, l, depth)
    vals, vecs = np.linalg.eigh(m)
    ipr = inverseParticipationRatio(vecs)
    iprs.append(ipr)
    showTree(vecs[depth**2:, np.argmin(np.abs(vals - testE))], K, depth)
    Es.append(vals[np.argmax(ipr)])
    b = 1
b = 1




