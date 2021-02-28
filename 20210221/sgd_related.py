import numpy as np
import matplotlib.pyplot as plt

with open('./Data/train.npy', 'rb') as fin:
    X = np.load(fin)

with open('./Data/target.npy', 'rb') as fin:
    y = np.load(fin)


def expand(X):
    """
    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]

    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))
    X_expanded[:, 0], X_expanded[:, 1] = X[:, 0], X[:, 1]
    X_expanded[:, 2] = X_expanded[:, 0] ** 2
    X_expanded[:, 3] = X_expanded[:, 1] ** 2
    X_expanded[:, 4] = np.multiply(X_expanded[:, 0], X_expanded[:, 1])
    X_expanded[:, 5] = 1
    return X_expanded

X_expanded = expand(X)


def probability(X, w):
    """
    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x), see description above

    Don't forget to use expand(X) function (where necessary) in this and subsequent functions.

    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """

    # TODO:<your code here>
    return 1 / (1 + np.exp(-(np.dot(X, w))))

def compute_loss(X,y,w):

    return -np.mean(y*np.log(probability(X,w))+(1-y)*np.log(1-probability(X,w)))


def compute_grad(X, y, w):
    """
    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    """
    return -np.mean(X * (y - probability(X, w))[:, np.newaxis], axis=0)


h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

from IPython import display
def visualize(X, y, w, history):
    """draws classifier prediction with matplotlib magic"""
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    display.clear_output(wait=True)
    plt.show()

# dummy_weights = np.linspace(-1, 1, 6)
# visualize(X, y, dummy_weights, [0.5, 0.5, 0.25])


np.random.seed(42)
w = np.array([0, 0, 0, 0, 0, 1])

eta= 0.1 # learning rate

n_iter = 100
batch_size = 4
loss = np.zeros(n_iter)
plt.figure(figsize=(12, 5))

for i in range(n_iter):
    ind = np.random.choice(X_expanded.shape[0], batch_size)
    loss[i] = compute_loss(X_expanded, y, w)
    if i % 10 == 0:
        visualize(X_expanded[ind, :], y[ind], w, loss)

    # Keep in mind that compute_grad already does averaging over batch for you!
    w=w-eta*compute_grad(X_expanded[ind,:],y[ind],w)

visualize(X, y, w, loss)
plt.clf()
