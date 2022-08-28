import random

import matplotlib.pyplot as plt
import numpy as np


class Fruit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.name = f'{x}_{y}'
        self.deployed = False


class GatherEnv:

    def __init__(self, width=10, height=10, n_actions=6, episode_length=100):
        self.counter = 0
        self.width = width
        self.height = height
        self.episode_length = episode_length
        self.field = np.zeros((self.width, self.height))
        self.agent_pos = []
        self.fruits = []
        self.fruits_dict = {}
        self.n_actions = n_actions
        self.cart = []

    def get_next_state(self):
        cart_state = len(self.cart)
        next_state = np.copy(self.field), self.agent_pos[:], cart_state
        return next_state

    def reset(self):
        self.counter = 0
        self.fruits = []
        self.fruits_dict = {}
        self.cart = []
        self.field = np.zeros((self.width, self.height))

        # gather line
        for y in range(self.height):
            self.field[0, y] = -1

        # agent pos
        self.agent_pos = [1, 4]

        # fruits
        for x in range(2, self.width-1):
            for y in [4, 5]:
                fruit = Fruit(x, y)
                self.fruits.append(fruit)
                self.fruits_dict[fruit.name] = fruit

        for fruit in self.fruits:
            self.field[fruit.x, fruit.y] = 1

        return self.get_next_state()

    def update_field(self):
        self.field = np.zeros((self.width, self.height))
        # gather line
        for y in range(self.height):
            self.field[0, y] = -1
        # fruits
        for fruit in self.fruits:
            self.field[fruit.x, fruit.y] = 1

    def exec_action(self, action):

        curr_x, curr_y = self.agent_pos

        # left
        if action == 0:
            curr_x = max(0, curr_x - 1)
        # up
        elif action == 1:
            curr_y = min(self.height - 1, curr_y + 1)
        # right
        elif action == 2:
            curr_x = min(self.width - 1, curr_x + 1)
        # down
        elif action == 3:
            curr_y = max(0, curr_y - 1)
        # pick
        elif action == 4:
            if len(self.cart) == 0:
                possible_fruits = [fruit for fruit in self.fruits if (fruit.x, fruit.y) == self.agent_pos]
                if len(possible_fruits) == 1:
                    fruit = possible_fruits[0]
                    self.cart.append(fruit)
                elif len(possible_fruits) > 1:
                    raise RuntimeError('more than one fruit on the position')
        # drop
        elif action == 5:
            if len(self.cart) == 1:
                # all except current
                curr_fruits = self.fruits[:]
                curr_fruits.remove(self.cart[0])
                # if there is another one on the same location
                possible_fruits = [fruit for fruit in curr_fruits if (fruit.x, fruit.y) == self.agent_pos]
                if len(possible_fruits) == 0:
                    self.cart = []
        else:
            raise RuntimeError('action incorrect')

        self.agent_pos = curr_x, curr_y
        for fruit in self.cart:
            fruit.x = curr_x
            fruit.y = curr_y

    def calc_reward(self):
        possible_fruits = [fruit for fruit in self.fruits if fruit.x == 0]
        if len(possible_fruits) > 1:
            raise RuntimeError('too mach fruits on The line')
        if len(self.cart) == 0:
            if len(possible_fruits) == 1:
                deployed_fruit = possible_fruits[0]
                self.fruits.remove(deployed_fruit)
                return 1
        return -1

    def step(self, action):
        self.counter += 1

        # action
        self.exec_action(action)

        # reward
        reward = self.calc_reward()

        # done
        done = self.counter > self.episode_length

        # updates
        self.update_field()

        return self.get_next_state(), reward, done

    def sample_action(self):
        return random.choice(list(range(self.n_actions)))

    def sample_state(self):
        pass

    def render(self):
        plt.cla()
        # field
        plt.imshow(self.field.T, origin='lower')
        # agent
        plt.scatter(self.agent_pos[0], self.agent_pos[1], s=300, c='red')
        plt.pause(0.05)


def main():
    env = GatherEnv(episode_length=EPS_L)

    # FIRST INIT
    state = env.reset()
    done = False

    for episode in range(EPISODES):

        counter = 0
        total_return = 0
        while not done:
            counter += 1
            action = env.sample_action()
            next_state, reward, done = env.step(action)

            # STATS
            total_return += reward

            # LEARNING
            pass

            # PLOT + PRINT
            print(f'\r[ep. {episode}, step {counter}] return: {total_return}', end='')
            if counter % 10 == 0:
                env.render()

        # END OF EPISODE
        state = env.reset()
        done = False

        # PLOT + PRINT
        print()


if __name__ == '__main__':
    EPISODES = 100
    EPS_L = 200
    main()


