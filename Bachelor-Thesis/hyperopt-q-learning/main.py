'''
e-greedy 1.0:  samples: 500
         0.9:           40
         0.8:           30
         0.7:           30
         0.6:           50
         0.5:           50
         0.4:           50
         0.3:           50
         0.2:           50
         0.1:           50

         total:         900
'''
import pandas as pd
from network import neural_network
from rl_brain import q_table
from rl_brain import uniform
from load_dataset import load_dataset
from copy import deepcopy

k_replay_update = 10
n_episode = 900

# Load dataset
(X_train,
 y_train), (X_test,
            y_test) = load_dataset('dataset_LSTM/smooth/Limited/memory_30/')
input_shape = (X_train.shape[1], X_train.shape[2])

# Initialize replay memory
Memory_S = []
Memory_U = []
Memory_accuracy = []

# Initialize q table
table = q_table(learning_rate=0.01)

# Main loop
for episode in range(n_episode):
    # define e-greedy's value
    if 0 <= episode < 500:
        e_greedy = 1.0
    elif 500 <= episode < 540:
        e_greedy = 0.9
    elif 540 <= episode < 570:
        e_greedy = 0.8
    elif 570 <= episode < 600:
        e_greedy = 0.7
    elif 600 <= episode < 650:
        e_greedy = 0.6
    elif 650 <= episode < 700:
        e_greedy = 0.5
    elif 700 <= episode < 750:
        e_greedy = 0.4
    elif 750 <= episode < 800:
        e_greedy = 0.3
    elif 800 <= episode < 850:
        e_greedy = 0.2
    else:
        e_greedy = 0.1

    # sample net structure, train and get accuracy
    S, U = table.sample_new_network(epsilon=e_greedy)
    net_structure = deepcopy(U)
    net_structure.append(('T', 4))
    nn = neural_network(net_structure=net_structure, input_shape=input_shape)
    nn.compile_model()
    nn.fit_model(X_train, y_train, batch_size=64, nb_epoch=5, val_X=X_test, val_y=y_test)
    accuracy = nn.evaluate_model()
    print('score :', accuracy)

    # store to replay memory
    Memory_S.append(S)
    Memory_U.append(U)
    Memory_accuracy.append(accuracy)

    # update q-table for k times
    for memory in range(k_replay_update):
        S_sample, U_sample, accuracy_sample = uniform(Memory_S, Memory_U, Memory_accuracy)
        table.update_q_values(S_sample, U_sample, accuracy_sample)
    
    # save log regularly
    if episode == 0 or episode % 10 == 9:
        nn.save_model('models/'+'episode_'+str(episode+1)+'.h5')
        table.save_q_table()
    
    print('Episode '+str(episode+1)+' finished.')
    
# save replay memory
df_save = pd.DataFrame(index=list(range(1, n_episode+1)))
df_save['Structure'] = Memory_U
df_save['Score'] = Memory_accuracy
df_save.to_csv('Replay_Memory.csv')

