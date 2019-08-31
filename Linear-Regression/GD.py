from numpy import *


def step_gradient(b, m, points, learning_rate):


def gradientDescentRunner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]


def run():
    points = genfromtext('data.csv', delimiter=',')
    learning_rate = 0.0001

    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    [b, m] = gradientDescentRunner(
        points, initial_b, initial_m, learning_rate, num_iterations)

    print(b)
    print(m)
