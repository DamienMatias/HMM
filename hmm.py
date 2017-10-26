import numpy as np


def ordonateb(B, order, N, T):
    new_b = np.zeros((N, T-1))
    for i in range(0, T-1):
        new_b[:, i] = B[:, order[i]]
    return new_b


# Forward algorithm
def forward(A, B, N, T, x):
    alpha = np.zeros((N, T))
    # new_b = ordonateb(B, x, N, T)
    alpha[1][0] = 1
    # print(A)

    for t in range(1, T):
        for j in range(N):
            sumalpha = 0
            for i in range(N):
                sumalpha = sumalpha + alpha[i][t - 1] * A[i][j]
            alpha[j][t] = B[j][x[t-1]] * sumalpha
    # print(np.around(alpha, decimals=3))
    return alpha


# Backward algorithm
def backward(A, B, N, T, x, alpha):
    beta = np.zeros((N, T))
    beta[0, T-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(N):
            for j in range(N):
                beta[i][t] += beta[j][t+1] * A[i][j] * B[j][x[t]]
    print(beta[1, 0])


A = np.array([[1, 0, 0, 0], [0.2, 0.3, 0.1, 0.4], [0.2, 0.5, 0.2, 0.1], [0.7, 0.1, 0.1, 0.1]])
B = np.array([[1, 0, 0, 0, 0], [0, 0.3, 0.4, 0.1, 0.2], [0, 0.1, 0.1, 0.7, 0.1], [0, 0.5, 0.2, 0.1, 0.2]])
N = 4
T = 5
x = [1, 3, 2, 0]

#Forward
alpha = forward(A, B, N, T, x)
print(alpha[0][T-1])

#Decoding
path = np.argmax(alpha, axis=0)
print(path)

backward(A, B, N, T, x, alpha)
