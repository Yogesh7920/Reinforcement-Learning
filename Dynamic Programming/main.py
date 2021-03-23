import numpy as np
from pprint import pprint
from copy import deepcopy


reward = np.array([
    [-1, -1, 10],
    [-1, -1, -1],
    [0, -1, -1]
])

actions = ['S', 'L', 'R', 'U', 'D']
values = np.zeros((3, 3))
policy = [['S' for _ in range(3)] for _ in range(3)]


def take_action(x, y, action):
    if action == 'L':
        y -= 1
    elif action == 'R':
        y += 1
    elif action == 'U':
        x -= 1
    elif action == 'D':
        x += 1

    x = max(min(x, len(reward)-1), 0)
    y = max(min(y, len(reward[0])-1), 0)

    return x, y


def reward_gain(x, y, next_values):

    gains = []
    for action in actions:
        nx, ny = take_action(x, y, action)
        gain = values[nx][ny] + reward[nx][ny]
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

        for i in range(3):
            for j in range(3):
                flag &= reward_gain(i, j, next_values)

        values = next_values
        if flag:
            break

    print(np.matrix(policy))
