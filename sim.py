import gym
from DQN_atari import Agent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__' :

    env=gym.make('ALE/Breakout-v5',full_action_space=False) #4 actions instead of 18
    #env=gym.make('CartPole-v1') #2 actions
    agent=Agent(gamma=0.99,epsilon=1.0,batch_size=16,n_actions=4,eps_end=0.01,input_dims=[84*84*4], lr=0.003)
    #agent=Agent(gamma=0.99,epsilon=1.0,batch_size=32,n_actions=2,eps_end=0.01,input_dims=[4], lr=0.003)

    scores, eps_history= [],[] #for plotting
    

    n_episodes=100

    for i in range(n_episodes):
        print('episode no. ',i+1)
        
        arrays=np.array([np.zeros((84,84)) for i in range(4)]) #for stacking 4 frames
        score=0
        done=False
        observation=env.reset() #state
        #observation=observation.flatten()
        observation = np.array(observation, dtype=np.float32)
        observation_count=0
        while not done:
          
            observation_modified=agent.preprocess(observation,84,arrays)
            print('observation number ', observation_count)
            
            print(observation_modified)
            print(observation_modified.shape)

            action=agent.choose_action(observation_modified)
            #observation_,reward,done,truncate,info= env.step(action) #observation_ == new state
            observation_,reward,done,info= env.step(action) #observation_ == new state

            observation_=np.array(observation_,dtype=np.float32)
            observation_next_modified=agent.preprocess(observation_,84,arrays)
            
            # observation_=observation_.flatten()
         

            score+=reward
            agent.store_transition(observation_modified,action,reward,observation_next_modified,done)
            agent.learn()
            observation=observation_
            epsilon=agent.epsilon
            observation_count+=1
            
        scores.append(score)
        eps_history.append(epsilon)

        avg_score=np.mean(scores[-100:]) #last 100 values

        print('episode: ', i, ' score: ', score, ' avg_score: ', avg_score, ' epsilon: ',epsilon)

    ep_no=[i for i in range(1,n_episodes+1)]
    plt.plot(ep_no, scores)
    plt.show()
    

    
        
