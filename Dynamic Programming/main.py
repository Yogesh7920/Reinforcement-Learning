import numpy as np
from pprint import pprint
from copy import deepcopy


reward = np.array([
    [2, 3, -1, 0],
    [1, 1, 9, 3],
    [4, 5, 7, -6],
    [2, -8, -3, 9]
])

actions = ['L', 'R', 'U', 'D']
values = np.zeros((len(reward), len(reward[0])))
policy = [['L' for _ in range(len(reward[0]))] for _ in range(len(reward))]


def take_action(x, y, action):

    if action == 'L':
        y -= 1
    elif action == 'R':
        y += 1
    elif action == 'U':
        x -= 1
    elif action == 'D':
        x += 1

    xo, yo = x, y

    x = max(min(x, len(reward)-1), 0)
    y = max(min(y, len(reward[0])-1), 0)

    return x, y, (xo != x or yo != y)


def reward_gain(x, y, next_values, hit_penalty=False):

    gains = []
    for action in actions:
        nx, ny, hit = take_action(x, y, action)
        gain = values[nx][ny] + reward[nx][ny] - 500 * hit * hit_penalty
        gains.append(gain)

    gains = np.array(gains)
    best_action = np.argmax(gains)

    next_values[x][y] = gains[best_action]
    best_action = actions[best_action]
    old = policy[x][y]
    policy[x][y] = best_action
    return old == policy[x][y]


if __name__ == '__main__':
    flag = False
    for _ in range(10):
        flag = True
        next_values = deepcopy(values)

        for i in range(len(reward)):
            for j in range(len(reward[0])):
                flag &= reward_gain(i, j, next_values, True)

        values = next_values
        if flag:
            break

    print(np.matrix(policy))
