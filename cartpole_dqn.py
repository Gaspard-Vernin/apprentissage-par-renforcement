import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque 
import random
epsilon=1
gamma=0.99
max_size_buffer=10000
class Net (nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(4,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,2)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    def forward(self,input):
        c1=torch.relu(self.fc1(input))
        c2=torch.relu(self.fc2(c1))
        c3=self.fc3(c2)
        return c3 
    def backprop(self,target,output):
        error=nn.MSELoss()
        loss=error(output,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def train(self,input,target):
        output=net(input)
        net.backprop(torch.tensor(target),output)
    def train_batches(self,batch_size,buffer):
        if len(buffer.buffer)<batch_size:
            #le buffer n'est pas plein, on ignore l'appel
            return
        #recuperer les transitions du batch du buffer
        transitions = buffer.echantilloner(batch_size)
        #on feed les transitions
        states=np.array([transition.state for transition in transitions],dtype=np.float32)
        feed_state = self.forward(torch.tensor(states))
        #stockage des valeur de Q pour chaque transitions
        q_value=[]
        for i in range(len(transitions)) : 
            q_value.append(feed_state[i,transitions[i].action])
        #on transforme la liste de tensor(Q value) en un tenseur de Q value
        q_value=torch.stack(q_value)
        #calcul des target pour chaque transitions
        with torch.no_grad():
            tab_transi=torch.tensor([transition.prochaine_etat for transition in transitions])
            next_max_q_value = self.forward(tab_transi).max(1)[0]
            targets = [transitions[i].reward+(1-transitions[i].done)*next_max_q_value[i]*gamma 
                       for i in range(len(transitions))]
            targets=torch.stack(targets).float()
        self.backprop(targets,q_value)
class Transition():
    def __init__(self,state,action,reward,prochaine_etat,done):
        self.state=state
        self.action=action 
        self.reward=reward 
        self.prochaine_etat=prochaine_etat
        self.done=done
class Buffer():
    def __init__(self):
        #on crée un buffer (liste circulaire) de taille max 10 000
        self.buffer = deque(maxlen=max_size_buffer)
    def add(self,transi):
        self.buffer.append(transi)
    def echantilloner(self,batch_size):
        batch=random.sample(population=self.buffer,k=batch_size)
        #dezip batch dans une liste de transi
        return batch
    def len(self):
        return len(self.buffer)
        
net=Net()
buffer=Buffer()
env=gym.make("CartPole-v1")
total_reward=0
liste_total_reward=[]
for iter in range(2000):
    if(iter!=0 and iter%50==0):
        liste_total_reward.append(total_reward)
        print(f"iter : {iter} score : {total_reward/50} eps : {epsilon} taille buffer : {buffer.len()}")
        total_reward=0
    etat,_=env.reset()
    done = False 
    while not done:
        if np.random.random()>epsilon:#exploitation
            with torch.no_grad():
                forw = net.forward(torch.tensor(etat,dtype=torch.float32))
                #.item transforme [x] en x
                action = torch.argmax(forw).item()
        else : 
            action = env.action_space.sample()
        etat_suivant, recompense, terminated, truncated, _ = env.step(action)
        done=terminated or truncated
        transi = Transition(etat,action,recompense,etat_suivant,done)
        total_reward+=recompense
        buffer.add(transi)
        net.train_batches(64,buffer)
        etat=etat_suivant
    epsilon=max(epsilon*0.9995,0.01)
plt.figure(figsize=(12,6))
plt.plot(liste_total_reward)
plt.show()
env=gym.make("CartPole-v1",render_mode="human")
for i in range(3):
    etat,_=env.reset()
    done = False 
    points_episode=0
    while not done:
        forw = net.forward(torch.tensor(etat))
        action = torch.argmax(forw).item()
        etat_suivant, recompense, terminated, truncated, info = env.step(action)
        done=terminated or truncated 
        tenseur_etat=torch.tensor(etat,dtype=torch.float32)
        #on force le dtype car etat est un np array donc en float64 et les poids sont en float32
        tenseur_prochain_etat=torch.tensor(etat_suivant,dtype=torch.float32)
        etat=etat_suivant
        points_episode+=recompense
    print(f"points episodes : {points_episode}\n")
    points_episode=0
