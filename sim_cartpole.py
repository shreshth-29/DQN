import gym
from DQN_atari import Agent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__' :

    #env=gym.make('ALE/Breakout-v5',full_action_space=False) #4 actions instead of 18
    env=gym.make('CartPole-v1') #2 actions
    #agent=Agent(gamma=0.99,epsilon=1.0,batch_size=16,n_actions=4,eps_end=0.01,input_dims=[210*160*3], lr=0.003)
    agent=Agent(gamma=0.99,epsilon=1.0,batch_size=32,n_actions=2,eps_end=0.01,input_dims=[4], lr=0.003)

    scores, eps_history= [],[] #for plotting

    n_episodes=100

    for i in range(n_episodes):
        score=0
        done=False
        observation=env.reset() #state
        observation=observation.flatten()
        observation = np.array(observation, dtype=np.float32)
        while not done:
            
            action=agent.choose_action(observation)
            #observation_,reward,done,truncate,info= env.step(action) #observation_ == new state
            observation_,reward,done,info= env.step(action) #observation_ == new state
            
            observation_=observation_.flatten()
            observation_=np.array(observation_,dtype=np.float32)

            score+=reward
            agent.store_transition(observation,action,reward,observation_,done)
            agent.learn()
            observation=observation_
            epsilon=agent.epsilon
            
        scores.append(score)
        eps_history.append(epsilon)

        avg_score=np.mean(scores[-100:]) #last 100 values

        print('episode: ', i, ' score: ', score, ' avg_score: ', avg_score, ' epsilon: ',epsilon)

    ep_no=[i for i in range(1,n_episodes+1)]
    plt.plot(ep_no, scores)
    plt.show()
    

    
        
