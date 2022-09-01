import random

import matplotlib.pyplot as plt
import numpy as np
from impl_env_gather import GatherEnv


def epsilon_greedy(q_func, agent_state, cart_state, actions):
    if random.random() > EPSILON:
        actions_dict = {action: q_func[agent_state[0], agent_state[1], cart_state, action] for action in actions}
        action = max(actions_dict, key=actions_dict.get)
        return action

    action = random.choice(actions)
    return action


def run_sarsa(env, q_func):
    # FOR PLOTS
    total_returns = []

    # FIRST INIT
    state = env.reset()
    done = False
    actions = list(range(env.n_actions))
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
            field, agent_state, cart_state = state
            next_field, agent_next_state, next_cart_state = next_state
            if not done:
                next_action = epsilon_greedy(q_func, agent_next_state, cart_state, actions)
                new_q = q_func[agent_state[0], agent_state[1], cart_state, action] + ALPHA * (
                            reward + GAMMA * q_func[agent_next_state[0], agent_next_state[1], next_cart_state, next_action] - q_func[
                        agent_state[0], agent_state[1], cart_state, action])
                q_func[agent_state[0], agent_state[1], cart_state, action] = new_q
            else:
                new_q = q_func[agent_state[0], agent_state[1], cart_state, action] + ALPHA * (reward - q_func[agent_state[0], agent_state[1], cart_state, action])
                q_func[agent_state[0], agent_state[1], cart_state, action] = new_q
                # for plot
                total_returns.append(total_return)

            # PLOT + PRINT
            print(f'\r[ep. {episode}, step {counter}] return: {total_return}, state: {agent_state}, cart-state: {cart_state}', end='')  # , field: \n{field}\n
            if episode > 1000 and counter % 1 == 0:
                env.render(total_returns=total_returns)

            # END OF STEP
            state = next_state

        # END OF EPISODE
        state = env.reset()
        done = False

        # PLOT + PRINT
        print()


def main():
    env = GatherEnv(episode_length=EPS_L)
    # width x height x cart x actions
    q_func = np.zeros((env.width, env.height, 2, env.n_actions))
    run_sarsa(env, q_func)


if __name__ == '__main__':
    EPISODES = 4000
    EPS_L = 200
    EPSILON = 0.1
    ALPHA = 0.5
    GAMMA = 0.9
    main()


