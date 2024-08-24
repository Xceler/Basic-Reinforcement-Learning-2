import gym
import numpy as np
from ddpg import Agent  # Ensure your DDPG Agent is suitable for discrete action spaces
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    
    # Since Acrobot-v1 has a discrete action space, use `env.action_space.n`
    n_actions = env.action_space.n
    
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=n_actions)
    
    n_games = 250
    figure_file = 'plots/acrobot.png'

    best_score = env.reward_range[0]

    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()

            # For discrete actions, you can sample from the action space
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = True

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            # If your Agent is designed for continuous action spaces, you'll need to adjust this
            # For example, you could modify `choose_action` to return discrete actions
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('Episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)
    
    # Plotting the learning curve
    plot_learning_curve(score_history, figure_file)
