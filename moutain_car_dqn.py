import gymnasium as gym 
import numpy as np 
from torch import nn
import torch
from collections import deque 
import random
import matplotlib.pyplot as plt
def shape_reward(position, velocity, done):
    # recompense basée sur la position (on pousse à aller vers la droite et en hauteur)
    reward = position + 0.5
    # bonus pour la vitesse positive (vers la droite)
    reward += abs(velocity) * 10
    # grosse récompense si atteint le sommet
    if done and position >= 0.5:
        reward += 100
    return reward
class Transition():
    def __init__(self,etat,action,reward,prochain_etat,done):
        self.etat=etat 
        self.action=action 
        self.reward=reward 
        self.prochain_etat = prochain_etat 
        self.done = done 
class Buffer():
    def __init__(self,batch_size,taille_buffer):
        self.buffer=deque(maxlen=taille_buffer)
        self.batch_size=batch_size
    def add(self,transition):
        self.buffer.append(transition)
    def define_batch(self):
        return random.sample(population=self.buffer,k=self.batch_size)

class Net(nn.Module):
    def __init__(self,gamma,eps,buffer):
        super(Net,self).__init__()
        self.gamma=gamma
        self.epsilon=eps
        self.buffer=buffer
        self.f1=nn.Linear(2,10)
        self.f2=nn.Linear(10,10)
        self.f3=nn.Linear(10,3)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
    def forward(self,input):
        x=self.f1(input)
        x=torch.relu(x)
        x=self.f2(x)
        x=torch.relu(x)
        x=self.f3(x)
        return x 
    def backprop(self,target,output):
        #on définit la donction d'erreur et le loss
        error_function=nn.MSELoss()
        loss=error_function(output,target)
        #on remet à 0 les gradients
        self.optimizer.zero_grad()
        #lance la backprop
        loss.backward()
        #on actualise l'optimisateur
        self.optimizer.step()
    def train_batches(self,net_objectif):
        if(len(self.buffer.buffer)<self.buffer.batch_size):
            return;
        #recuperer l'échantillon de transitions
        transitions = self.buffer.define_batch()
        #recuperer les etats 
        etats=np.array([t.etat for t in transitions],dtype=np.float32)
        #feed les etats 
        etats_feed = self.forward(torch.tensor(etats))
        #ici, dans etats_feed on a donc un tableau
        #de taille le nb d'actions possibles qui 
        #contient les Q values de chaque action 
        q_value_action=[etats_feed[i,transitions[i].action] for i in range(len(transitions))]
        #on transforme la liste de Q_value en un tenseur de Q_value
        q_value_action = torch.stack(q_value_action)
        #on veut pas changer les gradients
        with torch.no_grad():
            tab_next_step=[t.prochain_etat for t in transitions]
            #on forward les next_step puis on selectionne le max selon
            #la premiere dimension (Q(etat,action)), donc selno action
            #max(1) renvoie un couple (val_max,index_val_max), on veut
            #que la valeur donc on garde uniquement en [0]
            next_step_max_q_value = net_objectif.forward(torch.tensor(tab_next_step)).max(1)[0]
            targets=[transitions[i].reward+(1-transitions[i].done)*self.gamma*next_step_max_q_value[i]
                     for i in range(len(transitions))]
            #on empile ces tensor([valeur]) en un gros tenseur avec tout 
            targets=torch.stack(targets) 
        #on ne veut modifier que les paramètres liés à l'action 
        #qu'on a choisi ici, donc on envoie juste les Qoutput de la 
        #sortie et pytorch comprend tout seul que c'est que cette 
        #composante du vecteur output du forward qu'il 
        #faut backprop (c'est completement trop fort)
        self.backprop(targets,q_value_action)
env=gym.make("MountainCar-v0")
buffer = Buffer(batch_size=64,taille_buffer=10000)
net = Net(gamma=0.99,eps=1,buffer=buffer)
net_objectif=Net(gamma=0.99,eps=1,buffer=buffer)
total_rewards = []
acc_reward=0
step_compteur=0
for iter in range(1000):
    etat,_=env.reset()
    done=False 
    while not done :
        step_compteur+=1
        if(step_compteur%1000==0):
            net_objectif.load_state_dict(net.state_dict())
            step_compteur=0
        if np.random.random() < net.epsilon : 
            action = env.action_space.sample()
        else: 
            feed = net.forward(torch.tensor(etat,dtype = torch.float32))
            action = torch.argmax(feed).item()
        etat_suivant,reward,terminated,truncated,_ = env.step(action)
        done=truncated or terminated
        shaped_reward = shape_reward(etat_suivant[0], etat_suivant[1], done)
        transi = Transition(etat,action,shaped_reward,etat_suivant,done)   
        buffer.add(transi)
        net.train_batches(net_objectif)
        etat=etat_suivant
        acc_reward+=shape_reward(etat_suivant[0], etat_suivant[1], done)
    if (iter%10==0 and iter!=0):
        total_rewards.append(acc_reward)
        print(f"iter : {iter}, reward : {acc_reward/10}, eps : {net.epsilon}")
        acc_reward=0
    net.epsilon=max(net.epsilon*0.995,0.01)
acc_reward=0
plt.figure(figsize=(12,6))
plt.plot(total_rewards)
plt.show()
env=gym.make("MountainCar-v0",render_mode="human")
for i in range(3):
    etat,_=env.reset()
    done=False 
    acc_reward
    while not done :
        forw = net.forward(torch.tensor(etat,dtype=torch.float32))
        action = torch.argmax(forw).item()
        etat_suivant,recompense,terminated,truncated,_=env.step(action)
        done = truncated | terminated
        acc_reward+=recompense 
        etat=etat_suivant 
    if truncated : 
        print(f"fini par victoire, score = {acc_reward}")
    else :
        print (f"fini par défaite, score = {acc_reward}")
