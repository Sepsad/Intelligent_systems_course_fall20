import numpy as np
import pandas as pd

START_POINT = 1
END_POINT = 3
class ENV(object):
    
    def __init__(self, states = pd.DataFrame(), reward_dict = {}, random_starting_point = False):

        self.states = states
        self.reward_map = states.applymap(lambda x: reward_dict[x])
        self.state_cnt =  states.shape[0] * states.shape[1]
        self.state_row_cnt = states.shape[0]
        self.state_col_cnt = states.shape[1]

        self.reward_dict = reward_dict
        self.actions= ['U','D','L','R']   
        
        if random_starting_point:
            self.agent_curr_state = list(np.random.choice(range(15), size = 2))
        else:
            self.agent_curr_state = [(np.where(np.array(states) == START_POINT))[0][0], (np.where(np.array(states) == START_POINT))[1][0]]
        self.agent_last_state = [(np.where(np.array(states) == END_POINT))[0][0], (np.where(np.array(states) == END_POINT))[1][0]]
    
    def reset(self, random_starting_point = False):
        del self.reward_map
        self.reward_map = self.states.applymap(lambda x: self.reward_dict[x])
        if random_starting_point:
            self.agent_curr_state = list(np.random.choice(range(15), size = 2))
        else:
            self.agent_curr_state = [(np.where(np.array(self.states) == START_POINT))[0][0], (np.where(np.array(self.states) == START_POINT))[1][0]]
        return tuple(self.agent_curr_state)
     

    def getActionItems(self): 
        return len(self.actions) ,self.actions

    def getAgentPosition(self):
        return tuple(self.agent_curr_state)
    
    #just for debugging purpose
    def display_env(self):
        print("Number of states : {}".format(self.state_cnt))
        print("Action list : {}".format(self.actions))
        print("Agent's current position :{}".format(self.agent_curr_state))
        states_for_disp = self.states.copy()
        states_for_disp.loc[self.agent_curr_state[0],self.agent_curr_state[1]] = "A"
        print("Environment dump : \n{}\n".format(pd.DataFrame(states_for_disp).to_string(index=False,header=False)))
        print("Reward dump : \n{}\n".format(self.reward_map))

    
    def isDone(self,state):
        done = False
        if(list(state) == self.agent_last_state):
           done = True
        return done
    
    def render(self):
        states_for_disp = (self.states.copy())
        states_for_disp.loc[self.agent_curr_state[0],self.agent_curr_state[1]] = "A"
        return ("{}\n".format(states_for_disp))      
     
    def step(self,action): 
        done = False
        if (action == 'U'):
            tmp = self.agent_curr_state[0]
            self.agent_curr_state[0] = max(self.agent_curr_state[0] -1,0)
            reward = self.reward_map.loc[self.agent_curr_state[0], self.agent_curr_state[1]]
            if(tmp -1 < 0):
                reward = self.reward_dict[2]
     

        if (action == 'D'):
            tmp = self.agent_curr_state[0]
            self.agent_curr_state[0] = min(self.agent_curr_state[0] +1, self.state_row_cnt-1)
            reward = self.reward_map.loc[self.agent_curr_state[0], self.agent_curr_state[1]]

            if (tmp +1 > self.state_row_cnt-1 ):
                reward = self.reward_dict[2]
            
        if (action == 'L'):
            tmp = self.agent_curr_state[0]
            self.agent_curr_state[1] = max(self.agent_curr_state[1] -1,0)
            reward = self.reward_map.loc[self.agent_curr_state[0], self.agent_curr_state[1]]

            if (tmp -1 < 0):
                reward = self.reward_dict[2]

        if (action == 'R'):
            tmp = self.agent_curr_state[0]
            self.agent_curr_state[1] = min(self.agent_curr_state[1] +1,self.state_col_cnt-1)  
            reward = self.reward_map.loc[self.agent_curr_state[0], self.agent_curr_state[1]]

            if(tmp +1 > self.state_col_cnt - 1):
                reward = self.reward_dict[2]
        
        if(reward > 0):
            self.reward_map.loc[self.agent_curr_state[0], self.agent_curr_state[1]] = -0.01
        
        
        if (self.isDone(self.agent_curr_state) == True):
            done = True
            print('Target reached')
            
        
        next_state = tuple(self.agent_curr_state)
        return next_state, reward, done