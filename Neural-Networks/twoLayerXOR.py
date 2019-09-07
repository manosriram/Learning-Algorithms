import numpy as np
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


n_hidden = 10
n_inp = 10
n_out = 10
n_samples = 300

learning_rate = 0.01
momentum = 0.9

np.random.seed(0)


def Train(x, t, V, W, bv, bw):
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))

    return loss, (dV, dW, Ew, Ev)


def Predict(x, V, W, bV, bW):
    A = np.dot(x, V) + bV
    B = np.dot(np.tanh(A), W) + bW

    return (sigmoid(B) > 0.5).astype(int)


V = np.random.normal(scale=0.1, size=(n_inp, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bV = np.zeros(n_hidden)
bW = np.zeros(n_out)

params = [V, W, bV, bW]

X = np.random.binomial(1, 0.5, (n_samples, n_inp))
T = X ^ 1

for epoch in range(100):
    err = []
    upd = [0]*4  # len(params)

    t0 = time.clock()

    for i in range(X.shape[0]):
        loss, grad = Train(X[i], T[i], *params)

        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]

        err.append(loss)

    print("Epoch: {}  Loss: {}  Time: {}s".format(
        epoch, np.mean(err), time.clock() - t0))


x = np.random.binomial(1, 0.5, n_inp)
print(Predict(x, *params))
