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

    def __init__(self, width=4, height=1, n_actions=4, episode_length=100):
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

        # render
        self.fig, self.axs = plt.subplots(1, 3)  # , figsize=(9, 3), sharey=True

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

        # agent pos
        self.agent_pos = [1, int(self.height/2)]

        # fruits
        for x in range(2, self.width-1):
            for y in [int(self.height/2)]:
                fruit = Fruit(x, y)
                self.fruits.append(fruit)
                self.fruits_dict[fruit.name] = fruit

        self.update_field()

        return self.get_next_state()

    def update_field(self):
        self.field = np.zeros((self.width, self.height))

        # gather line
        # for y in range(self.height):
        #     self.field[0, y] = -1

        for fruit in self.fruits[:]:
            if fruit.x == 0 and len(self.cart) == 0:
                self.fruits.remove(fruit)

        # fruits
        for fruit in self.fruits:
            self.field[fruit.x, fruit.y] = 1

    def exec_action(self, action):

        curr_x, curr_y = self.agent_pos

        # left
        if action == 0:
            curr_x = max(0, curr_x - 1)
        # up
        # elif action == 1:
        #     curr_y = min(self.height - 1, curr_y + 1)
        # right
        elif action == 1:
            curr_x = min(self.width - 1, curr_x + 1)
        # down
        # elif action == 3:
        #     curr_y = max(0, curr_y - 1)
        # pick
        elif action == 2:
            if len(self.cart) == 0:
                possible_fruits = [fruit for fruit in self.fruits if (fruit.x, fruit.y) == self.agent_pos]
                if len(possible_fruits) == 1:
                    fruit = possible_fruits[0]
                    self.cart.append(fruit)
                elif len(possible_fruits) > 1:
                    raise RuntimeError('more than one fruit on the position')
        # drop
        elif action == 3:
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
        fruits_x_0 = [fruit for fruit in self.fruits if fruit.x == 0]
        if len(fruits_x_0) > 1:
            raise RuntimeError('too mach fruits on The line')
        if len(fruits_x_0) == 1:
            if len(self.cart) == 0:
                # deployed_fruit = fruits_x_0.pop()
                # self.fruits.remove(deployed_fruit)
                return 1
        return -1

    def calc_done(self):
        if self.counter >= self.episode_length:
            return True
        if len(self.fruits) == 0:
            return True
        return False

    def step(self, action):
        self.counter += 1

        # action
        self.exec_action(action)

        # reward
        reward = self.calc_reward()

        # updates
        self.update_field()

        # done
        done = self.calc_done()

        return self.get_next_state(), reward, done

    def sample_action(self):
        return random.choice(list(range(self.n_actions)))

    def sample_state(self):
        pass

    def render(self, total_returns=None):
        for ax_i in self.axs:
            ax_i.cla()

        # self.axs[0]
        # field
        self.axs[0].imshow(self.field.T, origin='lower')
        # agent
        self.axs[0].scatter(self.agent_pos[0], self.agent_pos[1], s=300, c='red')

        # self.axs[1]
        if total_returns:
            self.axs[1].plot(total_returns)
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
            if counter % 100 == 0:
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


