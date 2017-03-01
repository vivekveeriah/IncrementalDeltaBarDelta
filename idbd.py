### Incremental Delta Bar Delta
# variable names closely match the notations from the original paper

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

num_examples = 250000

input_dim = 20
meta_step_size = 0.05

h = np.zeros((input_dim, 1))
w = np.random.normal(0.0, 1.0, size=(input_dim, 1))
beta = np.ones((input_dim, 1)) * np.log(0.05)
alpha = np.exp(beta)

s = np.zeros((input_dim, 1))  # weights that generate a target
target_num = 5  # only first target_num examples are relevant for prediction

def generate_task():
    for i in range(target_num):
        s[i] = np.random.choice([-1, 1])

def tracking_task(x, examples_seen):
    if examples_seen % 20 == 0:
        s[np.random.choice(np.arange(target_num))] *= -1
    for i in range(target_num, input_dim):
        s[i] = np.random.random()
    return np.dot(s.transpose(), x)[0, 0]

def main():
    alpha_mat = np.zeros((num_examples, 2))
    global h, w, beta, alpha
    generate_task()
    for example_i in range(num_examples):
        x = np.random.rand(input_dim, 1)
        y = tracking_task(x, example_i + 1)
        estimate = np.dot(w.transpose(), x)[0, 0]
        delta = y - estimate
        beta += (meta_step_size * delta * x * h)
        alpha = np.exp(beta)
        w += delta * np.multiply(alpha, x)
        h = h * np.maximum((1.0 - alpha * (x ** 2)), np.zeros((input_dim, 1))) + (alpha * delta * x)

        alpha_mat[example_i, 0] = alpha[0]  # relevant feature
        alpha_mat[example_i, 1] = alpha[19]  # irrelevant feature

    x_length = np.arange(1, num_examples + 1)
    plt.errorbar(x_length, alpha_mat[:, 0].flatten(), label='relevant feature')
    plt.errorbar(x_length, alpha_mat[:, 1].flatten(), label='irrelevant feature')
    plt.legend(loc='best')
    plt.savefig('plot.png')

if __name__ == '__main__':
    main()