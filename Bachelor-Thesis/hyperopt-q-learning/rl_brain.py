import pandas as pd 
import numpy as np

class q_table(object):
    def __init__(self, learning_rate=0.01, lstm_search_space=(1, 32), dense_search_space=(1, 64)):

        self.lr = learning_rate

        lstm_search_space = list(range(lstm_search_space[0], lstm_search_space[1]+1))
        dense_search_space = list(range(dense_search_space[0], dense_search_space[1]+1))
        states_layer_0 = [('L', n_units, 0) for n_units in lstm_search_space]
        states_layer_1 = [('L', n_units, 1) for n_units in lstm_search_space]
        states_layer_2 = [('L', n_units, 2) for n_units in lstm_search_space]
        states_layer_3 = [('D', n_units, 3) for n_units in dense_search_space]
        self.table_0 = pd.DataFrame(index=[('Start')], columns=states_layer_0, dtype=np.float64, data=0.5)
        self.table_1 = pd.DataFrame(index=states_layer_0, columns=states_layer_1, dtype=np.float64, data=0.5)
        self.table_2 = pd.DataFrame(index=states_layer_1, columns=states_layer_2, dtype=np.float64, data=0.5)
        self.table_3 = pd.DataFrame(index=states_layer_2, columns=states_layer_3, dtype=np.float64, data=0.5)
    
    def save_q_table(self, save_path='q_table/'):
        self.table_0.to_csv(save_path+'table_0.csv')
        self.table_1.to_csv(save_path+'table_1.csv')
        self.table_2.to_csv(save_path+'table_2.csv')
        self.table_3.to_csv(save_path+'table_3.csv')

    def sample_new_network(self, epsilon):
        # initialize
        S = [('Start')]
        U = []
        # sample layer_0
        if np.random.uniform(0, 1) > epsilon:
            row = list(self.table_0.index).index(S[-1])
            state_action_0 = self.table_0.iloc[row, :]
            state_action_0 = state_action_0.reindex(np.random.permutation(state_action_0.index))
            action_0 = state_action_0.idxmax()
            state_1 = action_0
        else:
            action_0 = np.random.choice(self.table_0.columns)
            state_1 = action_0
        U.append(action_0)
        S.append(state_1)
        # sample layer_1
        if np.random.uniform(0, 1) > epsilon:
            row = list(self.table_1.index).index(S[-1])
            state_action_1 = self.table_1.iloc[row, :]
            state_action_1 = state_action_1.reindex(np.random.permutation(state_action_1.index))
            action_1 = state_action_1.idxmax()
            state_2 = action_1
        else:
            action_1 = np.random.choice(self.table_1.columns)
            state_2 = action_1
        U.append(action_1)
        S.append(state_2)
        # sample layer_2
        if np.random.uniform(0, 1) > epsilon:
            row = list(self.table_2.index).index(S[-1])
            state_action_2 = self.table_2.iloc[row, :]
            state_action_2 = state_action_2.reindex(np.random.permutation(state_action_2.index))
            action_2 = state_action_2.idxmax()
            state_3 = action_2
        else:
            action_2 = np.random.choice(self.table_2.columns)
            state_3 = action_2
        U.append(action_2)
        S.append(state_3)
        # sample layer_3
        if np.random.uniform(0, 1) > epsilon:
            row = list(self.table_3.index).index(S[-1])
            state_action_3 = self.table_3.iloc[row, :]
            state_action_3 = state_action_3.reindex(np.random.permutation(state_action_3.index))
            action_3 = state_action_3.idxmax()
        else:
            action_3 = np.random.choice(self.table_3.columns)
        U.append(action_3)  # S is not extended
        return S, U 

    def update_q_values(self, S, U, accuracy):
        # update table 3
        row3 = list(self.table_3.index).index(S[3])
        col3 = list(self.table_3.columns).index(U[3])
        self.table_3.iloc[row3, col3] = (1 - self.lr)*self.table_3.iloc[row3, col3] + self.lr*accuracy
        # update table 2
        row2 = list(self.table_2.index).index(S[2])
        col2 = list(self.table_2.columns).index(U[2])
        max_val = max(self.table_3.iloc[row3, :])
        self.table_2.iloc[row2, col2] = (1 - self.lr)*self.table_2.iloc[row2, col2] + self.lr*max_val
        # update table 1
        row1 = list(self.table_1.index).index(S[1])
        col1 = list(self.table_1.columns).index(U[1])
        max_val = max(self.table_2.iloc[row2, :])
        self.table_1.iloc[row1, col1] = (1 - self.lr)*self.table_1.iloc[row1, col1] + self.lr*max_val
        # update table 0
        row0 = list(self.table_0.index).index(S[0])
        col0 = list(self.table_0.columns).index(U[0])
        max_val = max(self.table_1.iloc[row1, :])
        self.table_0.iloc[row0, col0] = (1 - self.lr)*self.table_0.iloc[row0, col0] + self.lr*max_val


def uniform(memory_s, memory_u, memory_acc):
    memory_size = len(memory_s)
    sample_index = np.random.randint(0, memory_size)
    return memory_s[sample_index], memory_u[sample_index], memory_acc[sample_index]

def convert_to_tuple(sample_sequence):
    seq_converted = list()
    for item in sample_sequence:
        if item == 'Start':
            seq_converted.append(item)
        else:
            useful_part = item.split('(')[1].split(')')[0].split(',')
            layer_type = useful_part[0][1]
            n_units = int(useful_part[1])
            n_depth = int(useful_part[2])
            seq_converted.append((layer_type, n_units, n_depth))
    return seq_converted

