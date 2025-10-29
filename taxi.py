import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt
import time
epsilon=1
gamma=0.95
lr=0.1
env = gym.make("Taxi-v3")
state,_=env.reset()
decoded = tuple(env.unwrapped.decode(state))
#taxi_row, taxi_col, passenger_idx, destination_idx = env.decode(state)
"""Passenger locations:

    0: Red

    1: Green

    2: Yellow

    3: Blue

    4: In taxi
    Destinations:

    0: Red

    1: Green

    2: Yellow

    3: Blue

"""
Q_table=np.random.uniform(low=-1,high=1,size=(500,env.action_space.n))
total_reward=0
rewardlist=[]
for iter in range(1000):
    (state,_)=tuple(env.reset())
    done=False 
    while not done:
        if np.random.random()>epsilon:#exploitation
            action = np.argmax(Q_table[state])
        else:#exploration 
            action=env.action_space.sample()
        new_state, reward, done,_,_=env.step(action)
        total_reward+=reward 
        Q_table[state,action]+=lr*(reward+gamma*np.max(Q_table[new_state])-Q_table[state][action])
        state=new_state 
    if iter%100==0:
        rewardlist.append(total_reward)
        print(f"etape : {iter}, score : {total_reward/100}")
        total_reward=0
    epsilon=max(0.1,epsilon-1/10000)
plt.figure(figsize=(12,6))
plt.plot(rewardlist)
plt.show()
env = gym.make("Taxi-v3",render_mode="human")
(state,_)=tuple(env.reset())
done=False 
for _ in range(10):
    (state,_)=tuple(env.reset())
    done=False
    while not done:
        env.render()
        time.sleep(0.01)
        action = np.argmax(Q_table[state])
        new_state, reward, done,_,_=env.step(action)
        state=new_state
    
