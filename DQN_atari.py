import gym
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

class DQN(nn.Module):
    def __init__(self, lr,input_dims,fc1_dims,fc2_dims,n_actions):
    
        super(DQN,self).__init__()

        self.input_dims=input_dims
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.n_actions=n_actions
        self.lr=lr
        
        print(self.fc1_dims)
        print(*self.input_dims)

        self.conv1=nn.Conv2d(4,16,8,stride=4)
        self.conv2=nn.Conv2d(16,32,4,stride=2)
        self.fc1=nn.Linear(2592,256)
        self.out=nn.Linear(256,self.n_actions)

    
        # self.fc1=nn.Linear(*self.input_dims,self.fc1_dims)
        # self.fc2=nn.Linear(self.fc1_dims,fc2_dims)
        # self.fc3=nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer=optim.Adam(self.parameters(),lr=lr)
        self.loss= nn.MSELoss()
        self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.device='cpu' #T.device('cpu')
        self.to(self.device)

    def forward(self,state):
        #print(state.shape)
        x=self.conv1(state)
        x=F.relu(x)
        #print(x.shape)
        x=self.conv2(x)
        x=F.relu(x)
        print(x.shape)
        arrays=T.empty(size=(16,32*9*9))
        #print(len(x))
        for i in range(0,len(x)):
          array=x[i]
          #print(array.shape)
          
          arrays[i]=T.flatten(array)
        #print(arrays.shape)
        x=T.Tensor(arrays).to(self.device)


        # x=T.flatten(x)
        #print(x.shape)
        x=self.fc1(x)
        x=F.relu(x)
        #print(x.shape)
        Q_values=self.out(x) #we want the raw estimate of the agent, so no activation function in the final layer
        #print(Q_values.shape)
        #print(Q_values)
        return Q_values
    
class Agent():

    def __init__(self, gamma,epsilon,lr,input_dims,batch_size,n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):

        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_end
        self.eps_dec=eps_dec
        self.lr=lr
        self.input_dims=input_dims
        self.batch_size=batch_size
        self.mem_size=max_mem_size
        self.mem_cntr=0 #memory counter
        self.action_space=[i for i in range(n_actions)]

        self.Q_eval = DQN(self.lr, n_actions=n_actions,input_dims=input_dims, fc1_dims=256, fc2_dims=256)

        #for replay buffer

        self.state_memory= np.zeros( (self.mem_size, *input_dims) , dtype=np.float32).reshape((self.mem_size,4,84,84))
        #print(self.state_memory.shape)
        self.new_state_memory= np.zeros( (self.mem_size, *input_dims) , dtype=np.float32).reshape((self.mem_size,4,84,84))
        self.action_memory= np.zeros(self.mem_size, dtype=np.int32) #discrete actions 0,1
        self.reward_memory= np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory=np.zeros(self.mem_size, dtype=np.bool)


    def preprocess(self,observation,op_dims,arrays):
        
        arrays=arrays.reshape((4,84,84))
        gray=cv2.cvtColor(observation,cv2.COLOR_BGR2GRAY)
        scaled=cv2.resize(gray,(110,84))
        #print(scaled.shape)
        #crop image to (84,84)
        dims1= int(55-(0.5*op_dims))
        dims2= int(55+(0.5*op_dims))
        cropped= scaled[:,dims1:dims2]
        #print(cropped.shape)

        for i in range(3):
            arrays[i]=np.copy(arrays[i+1])

        arrays[3]=np.copy(cropped)
    
        #array_stacked=arrays.reshape((84,84,4))

        return arrays
    
        


    def store_transition(self,state,action,reward,state_ , done): #store transitions in agent's memory. state_=== next state

        index=self.mem_cntr % self.mem_size #resets index to 0 when we hit mem_size. earliest memories are rewritten
        self.state_memory[index]=state
        self.new_state_memory[index]=state_
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done

        self.mem_cntr+=1


    def choose_action(self,observation):
        #epsilon greedy

        if (np.random.random() > self.epsilon): #exploit
            device=self.Q_eval.device
            state=T.tensor([observation]).to(device)
            Q_values= self.Q_eval.forward(state)
            action = T.argmax(Q_values).item() #select action corresponding to max Q value. Will return an index, which is what we want (as only 0,1 are actions)

        else: #explore
            action=np.random.choice(self.action_space)

        return action

    def learn(self):

        #play randomly till you fill up some portion of the memory (=batch size). then start learning from that memory

        if(self.mem_cntr < self.batch_size):
            return

        else:
            
            
            self.Q_eval.optimizer.zero_grad() #initilize gradients

            max_mem=min(self.mem_cntr, self.mem_size) #memory untill mem_size

            batch=np.random.choice(max_mem , self.batch_size, replace=False) #choose a batch of samples from memory. replace=False so that we dont end up choosing the same values
            
            batch_index= np.arange(self.batch_size, dtype=np.int32) # array

            states=self.state_memory[batch]
            actions=self.action_memory[batch]
            new_states=self.new_state_memory[batch]
            rewards = self.reward_memory[batch]
            terminals=self.terminal_memory[batch]

            #eg, for max_mem=40, self.batch_size= 8, then batch will be array of 8 random values in [0,40]
            #self.state_memory[batch] will then select 8 states from memory. this will be stored in state_batch

            device=self.Q_eval.device

            state_batch=T.tensor(states).to(device)
            print(T.tensor(states).shape)
            new_state_batch = T.tensor(new_states).to(device)
            reward_batch = T.tensor(rewards).to(device)
            terminal_batch= T.tensor(terminals).to(device)

            #actions not converted to tensors.

            q_eval_all_actions_state = self.Q_eval.forward(state_batch) #returns q values for all possible actions for the states in our batch
            q_eval_state= q_eval_all_actions_state[batch_index, actions] #Q values correspondig to our actions

            
            q_new_state = self.Q_eval.forward(new_state_batch) #for next state q values we need them for all actions. so no slicing
            q_new_state[terminal_batch]=0.0 #set 0 Q value for terminal states

            q_target = reward_batch + self.gamma*T.max(q_new_state,dim=1)[0] #T.max returns index as well, we just need the value

            loss = self.Q_eval.loss(q_eval_state,q_target).to(device)
            loss.backward() #backprop
            self.Q_eval.optimizer.step()

            if(self.epsilon > self.eps_min): #epsilon decay till minimum value
                self.epsilon = self.epsilon - self.eps_dec
            else:
                self.epslon= self.eps_min
                

            

            

            

            

            
            
            
            
            

        
        
        

        

    

        

    

        
        
        

    
    

    
    
