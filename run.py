from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
import numpy as np
from tqdm import tqdm


# Run dqn with Tetris
def dqn():
    env = Tetris()
    episodes = 500
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']
    actions = []

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    # log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # log = CustomTensorBoard(log_dir=log_dir)
    # agent.set_memory(np.load('FILE PATH'))
    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False
        # render = False
        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            if render:
                reward, done, best_action = env.player_play(render=render, render_delay=render_delay)
            else:
                best_action = None
                for action, state in next_states.items():
                    if state == best_state:
                        best_action = action
                        break
                reward, done = env.play(best_action[0], best_action[1], render=render, render_delay=render_delay)

            print(reward, done, best_action)
            if best_action in next_states.items():
                agent.add_to_memory(current_state, next_states[best_action], reward, done)
                current_state = next_states[best_action]
                actions.append(best_action)
                steps += 1

        scores.append(env.get_game_score())
        # enregister dans file --> agent.get_memory()
        memory = np.array([agent.get_memory()]).T
        weights = agent.model.save_weights
        np.save('ia_tetris_weights', weights)
        np.save('ia_tetris_memory', memory)
        print('--------')
        print(memory)
        print(weights)
        print('---------')
        # Train
        agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        # if log_every and episode and episode % log_every == 0:
        #     avg_score = mean(scores[-log_every:])
        #     min_score = min(scores[-log_every:])
        #     max_score = max(scores[-log_every:])
        #
        #     log.log(episode, avg_score=avg_score, min_score=min_score,
        #             max_score=max_score)


if __name__ == "__main__":
    dqn()
    # enregister model
