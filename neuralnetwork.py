import numpy as np

# def create_network(*nodeNumbers):
#     L = len(nodeNumbers)

#     network = np.zeros(L)
#     return network

def create_network(*nodeNumbers):
    w = []
    a = []
    b = []

    for l in range(1, len(nodeNumbers)):
        a.append(np.zeros(nodeNumbers[l]))
        b.append(np.zeros(nodeNumbers[l]))
        w.append(np.zeros(nodeNumbers[l]))

    return (w,a,b)