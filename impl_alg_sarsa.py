import random

import matplotlib.pyplot as plt
import numpy as np
from impl_env_gather import GatherEnv


def process_state(state):
    field, agent_pos, cart_state = state
    new_state = np.zeros(7)
    new_state[0] = field[0]
    new_state[1] = field[1]
    new_state[2] = field[2]
    new_state[3] = field[3]
    new_state[4] = agent_pos[0]
    new_state[5] = agent_pos[1]
    new_state[6] = cart_state
    return new_state


def get_q_index(state, action):
    q_index = state.tolist()
    q_index.append(action)
    q_index = [int(i) for i in q_index]
    return tuple(q_index)


def epsilon_greedy(q_func, state, actions, epsilon):
    if random.random() > epsilon:
        actions_dict = {}
        for action in actions:
            actions_dict[action] = q_func[get_q_index(state, action)]
        max_value = max(actions_dict.values())
        max_actions = [action for action, value in actions_dict.items() if value == max_value]
        action = random.choice(max_actions)
        return action

    action = random.choice(actions)
    return action


def run_sarsa(env, q_func):
    # FOR PLOTS
    total_returns = []

    # FIRST INIT
    state = env.reset()
    state = process_state(state)
    done = False
    actions = list(range(env.n_actions))
    for episode in range(EPISODES):
        epsilon = 1 - episode/EPISODES
        counter = 0
        total_return = 0
        while not done:
            counter += 1
            # action = env.sample_action()
            action = epsilon_greedy(q_func, state, actions, epsilon)
            next_state, reward, done = env.step(action)
            next_state = process_state(next_state)

            # STATS
            total_return += reward

            # LEARNING
            # field, agent_state, cart_state = state
            # next_field, agent_next_state, next_cart_state = next_state
            if not done:
                next_action = epsilon_greedy(q_func, next_state, actions, epsilon)
                new_q = q_func[get_q_index(state, action)] + ALPHA * (
                            reward + GAMMA * q_func[get_q_index(next_state, next_action)] - q_func[get_q_index(state, action)])
                q_func[get_q_index(state, action)] = new_q
            else:
                new_q = q_func[get_q_index(state, action)] + ALPHA * (reward - q_func[get_q_index(state, action)])
                q_func[get_q_index(state, action)] = new_q
                # for plot
                total_returns.append(total_return)

            # PLOT + PRINT
            print(f'\r[ep. {episode}, step {counter}] return: {total_return}, state: {state}, q_index: {get_q_index(state, action)}', end='')
            if episode > EPS_TO_RENDER and counter % 1 == 0:
                env.render(total_returns=total_returns)
                # plt.pause(1)

            # END OF STEP
            state = next_state

        # END OF EPISODE
        state = env.reset()
        state = process_state(state)
        done = False

        # PLOT + PRINT
        print()


def main():
    env = GatherEnv(episode_length=EPS_L)
    # width x height x cart x actions
    q_func = np.ones((2, 2, 2, 2, 4, 4, 2, env.n_actions)) * -100
    run_sarsa(env, q_func)


if __name__ == '__main__':
    EPS_TO_RENDER = 800
    EPISODES = int(EPS_TO_RENDER * 1.2)
    EPS_L = 200
    # EPSILON = 0.4
    ALPHA = 0.5
    GAMMA = 0.9
    main()


