import numpy as np


# Forward algorithm
def forward(A, B, N, T, x):
    alpha = np.zeros((N, T))
    alpha[1][0] = 1

    for t in range(1, T):
        for j in range(N):
            sumalpha = 0
            for i in range(N):
                sumalpha = sumalpha + alpha[i][t - 1] * A[i][j]
            alpha[j][t] = B[j][x[t-1]] * sumalpha
    return alpha


# Backward algorithm
def backward(A, B, N, T, x, alpha):
    beta = np.zeros((N, T))
    beta[0, T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(N):
            for j in range(N):
                beta[i][t] += beta[j][t+1] * A[i][j] * B[j][x[t]]
    return beta


# Baum-Welch algorithm
def bw(A, B, x, N, T):
    gamma = np.zeros((N, N, T))
    new_a = np.zeros(A.shape)
    new_b = np.zeros(B.shape)

    alpha = forward(A, B, N, T, x)
    beta = backward(A, B, N, T, x, alpha)
    # Compute gamma
    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                gamma[i, j, t] = alpha[i, t-1] * A[i, j] * B[j, x[t-1]] * beta[j, t]

    # Compute a
    for i in range(N):
        for j in range(N):
            denom = 0
            for t in range(1, T):
                new_a[i, j] += gamma[i, j, t]
                denom += np.sum(gamma[i, :, t])
            if denom != 0:
                new_a[i, j] /= denom
    new_a[0, 0] = 1

    # Compute b
    for j in range(N):
        for k in range(N):
            denom = 0
            for t in range(1, T):
                denom += np.sum(gamma[j, :, t])
                if x[t-1] == k:
                    for l in range(N):
                        new_b[j, k + 1] += gamma[j, l, t]
            if denom != 0:
                new_b[j, k+1] /= denom
    new_b[0, 0] = 1
    return new_a, new_b


# Initialisation
A = np.array([[1, 0, 0, 0], [0.2, 0.3, 0.1, 0.4], [0.2, 0.5, 0.2, 0.1], [0.7, 0.1, 0.1, 0.1]])
B = np.array([[1, 0, 0, 0, 0], [0, 0.3, 0.4, 0.1, 0.2], [0, 0.1, 0.1, 0.7, 0.1], [0, 0.5, 0.2, 0.1, 0.2]])
N = 4
T = 5
x = [1, 3, 2, 0]

# Forward
alpha = forward(A, B, N, T, x)

# Decoding
path = np.argmax(alpha, axis=0)

# Backward
beta = backward(A, B, N, T, x, alpha)

# BW
for i in range(10):
    A, B = bw(A, B, x, N, T)

print(np.around(A, 3))
print(np.around(B, 3))
