import gym 
import numpy as np 
from rein_tf import Agent 
from utils import plotLearning 

if __name__ == '__main__':
    agent = Agent(alpha = 0.0005, gamma =0.99, n_actions = 2) 

    env = gym.make('CartPole-v1')
    score_history = []

    num_episodes = 200 
    for i in range(num_episodes):
        done = False 
        score = 0 
        observation = env.reset() 
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            agent.store_transition(observation, action, reward)
            observation = observation_ 
            score += reward 
        
        score_history.append(score)

        agent.learn() 

        avg_score = np.mean(score_history[-100:])
        print('episode:', i , 'score:%.1f' % score,
               'average score%.1f' % avg_score)

        
    
    filename = 'cartpole.png'
    plotLearning(score_history, filename =filename, window= 100)