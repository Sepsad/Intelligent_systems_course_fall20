import env
import numpy as np
import pandas as pd
import time

ACTION_ERROR_PROB = 0.05
OPPOSIT_ACTION_DICT = {"U":"D", "D":"U", "R":"L","L":"R" }

class Q_table:

    def __init__(self, env, gamma=0.9, n = 15, alpha=0.1, num_episodes=500):

        self.alpha = alpha
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.env1 = env
        self.action_count, self.actions = self.env1.getActionItems()

        self.Q_table = pd.DataFrame(0,index=pd.MultiIndex.from_product([list(range(n)) ,list(range(n))]),\
            columns=self.actions)
    
    def policy(self ,state, beta = 1):
        Qs = dict(self.Q_table.loc[state, :])
        sumQ = sum([ np.exp( Qs[a]/beta ) for a in self.actions ])
        probs = [  np.exp( Qs[a]/beta )/sumQ for a in self.actions ]
        action = np.random.choice(self.actions, size=1, p = probs)
        return action[0]

    def update(self, curr_state, action, reward, next_state):
        curr_Q = self.Q_table.loc[curr_state,action]
        next_Q = self.Q_table.loc[next_state,:].max()

        self.Q_table.loc[curr_state,action] += self.alpha * (reward + self.gamma * next_Q - curr_Q)

        opp_action = OPPOSIT_ACTION_DICT[action[0]]
        self.Q_table.loc[next_state, opp_action] -= 1
    
    def learn(self, Q_table_name  = "Q_table", random_starting_point = False):
        
        total_reward_dict = {}
        for episode_cnt in range(self.num_episodes):
            print('episode: {}'.format(episode_cnt),end='\t')
            state = self.env1.reset(random_starting_point = random_starting_point)
            print(state)
            done = self.env1.isDone(state)
            total_reward = 0
            while not done:
                action = self.policy(state)
                p = np.random.uniform(0,1)
                if ( p < ACTION_ERROR_PROB):
                    actions_wo_a = self.actions.copy()
                    actions_wo_a.remove(action)
                    action = np.random.choice(actions_wo_a, size = 1)
                next_state, reward, done = self.env1.step(action) 
                total_reward += reward
                self.update(state, action, reward, next_state)               
                state = next_state
            total_reward_dict[episode_cnt] = total_reward
        print('\n Final Q table: \n {}'.format(self.Q_table))
        print('\n Final ENV: \n {}'.format(self.env1.render()))
        
        self.Q_table.to_pickle( Q_table_name + '.pkl') 

        return total_reward_dict

    def test(self,random_starting_point = False):
        state = self.env1.reset(random_starting_point = random_starting_point)
        done = self.env1.isDone(state)
        while not done:
            action = self.policy(state)
            p = np.random.uniform(0,1)
            if ( p < ACTION_ERROR_PROB):
                actions_wo_a = self.actions.copy()
                actions_wo_a.remove(action)
                action = np.random.choice(actions_wo_a, size = 1)
            next_state, _, done = self.env1.step(action) 
            state = next_state
            print(self.env1.render())

    
