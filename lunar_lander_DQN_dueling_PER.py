import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import math
from gymnasium.wrappers import RecordVideo
import os
import shutil
import time
class Transi:
    def __init__(self,etat,action,reward,prochain_etat,done):
        self.etat=etat 
        self.action=action 
        self.reward=reward 
        self.prochain_etat = prochain_etat 
        self.done=done 
    def __eq__(self, t1):
        assert(isinstance(t1,Transi))
        return (self.etat==t1.etat) and (self.action==t1.action) and (self.reward==t1.reward) and(self.prochain_etat==t1.prochain_etat) and (self.done==t1.done)
class Sumtree:
    def __init__(self,taille):
        self.taille=taille 
        #un arbre parfait qui a x feuilles à 2x-1 noeuds 
        self.tree = np.zeros(taille*2-1)
        #data[i] représente la transition indexée par i
        self.data = np.zeros(taille,dtype=Transi)
        #pointeur pointe sur la prochaine case sur laquelle on va écrire
        self.pointeur=0
        self.taille_actuele=0
        #on initialise max_priorité à 1, pour que la première valeur ajoutée dans le tree ait cette priorité
        #et pas une priorité nulle ainsi elle sera tirée et sa vraie priorité sera mise à la place de 1
        self.max_priorité = 1
    def i_data_to_i_arbre(self,x):
        """renvoie l'indice de l'arbre associé à l'indice de data"""
        return self.taille-1+x
    def i_arbre_to_i_data(self,x):
        """renvoie l'indice de data associé à l'indice de l'arbre"""
        return -(self.taille-1)+x
    def add(self,priorité,transi):
        """ajoute transi/priorité et supprime la plus vieilles transi du sumtree"""
        #on stocke la transition a la bonne case
        self.data[self.pointeur]=transi 
        #on update la priorité dans l'arbre 
        self.update(self.i_data_to_i_arbre(self.pointeur),priorité)
        #on avance write, si il dépasse taille, il repart à 0
        self.pointeur=(self.pointeur+1)%self.taille
        if(self.taille_actuele<self.taille):
            self.taille_actuele+=1
    def update(self,indice_arbre,priorité):
        """prend l'indice dans l'arbre a changer et une priorité et fait les updates nécessaires"""
        if(priorité>self.max_priorité):
            self.max_priorité=priorité
        ancienne_priorité = self.tree[indice_arbre]
        #quand indice_arbre=0, il est ensuite update a -1 donc on quitte la boucle pile après avoir finit toutes les modif
        while(indice_arbre>=0):
            self.tree[indice_arbre]+=(priorité-ancienne_priorité)
            #le parent d'un noeud d'indexe i dans un arbre parfait est a l'index partie entière de(i-1)//2
            indice_arbre = (indice_arbre-1)//2
    def total(self):
        """renvoie la somme des priorités stockées"""
        return self.tree[0]
    def get(self, priorité):
        """renvoie (transition,indice_data) en fonction de la priorité donnée"""
        i=0
        while True:
            #si c'est une feuille
            if(i>=self.taille-1):
                indice_data=self.i_arbre_to_i_data(i)
                return (self.data[indice_data],indice_data)
            i_enfant_gauche=2*i+1
            #si la transi cherchée est à gauche, le fils gauche de i est a l'index 2i+1
            if priorité <=self.tree[i_enfant_gauche]:
                i=i_enfant_gauche
            #si la transi est à droite, le fils droit de i est a l'index 2i+2, 
            #il faut soustraire à la priorité la priorité de gauche car elle est comptée dans priorité
            else :
                priorité-=self.tree[2*i+1]
                i=i_enfant_gauche+1
                
        return 
class Buffer:
    def __init__(self,capacity,alpha,batch_size,beta):
        self.sumtree=Sumtree(capacity)
        self.alpha=alpha 
        self.batch_size=batch_size
        self.beta=beta
    def add(self,transi):
        """ajoute la transition dans le buffer avec la plus grande priorité du buffer existante,
            la priorité sera update quand on aura train au moins une fois dessus et donc qu'on update
            ainsi toutes les priorités (à priori) seront jouées au moins une fois"""
        self.sumtree.add(self.sumtree.max_priorité,transi)
    def tirage(self):
        """renvoie un tirage de taille self.buffer_size respectant les priorités du sumtree sous la forme (transitions,indices,compensations)"""
        #pour respecter les priorités au mieux, on va diviser le "segment" [0,sumtree.total()] en 
        #batch_size segments côte à côte. on tire une transi dans chaque segment. Ainsi on force le
        #tirage à s'étaler sur tout le spectre des priorités et donc on réduit les cas où la faute à
        #pas de chance, on a tiré que des transition nulles
        taille_segment = self.sumtree.total()/self.batch_size 
        """
        version claire de la version vectoriése ci dessous
        tirage = []
        for i in range(taille_segment):
            #on tire la position sur l'interval, puis on décale jusqu'à l'interval
            prio_tirée = (np.random.random()*taille_segment)+(i*taille_segment)
            tirage.append(self.sumtree.get(prio_tirée))
        """
        #on tire les priorités sur les segments et on extrait transitions et indices
        priorités = [(np.random.random()*taille_segment)+(i*taille_segment)for i in range(self.batch_size)]
        tirage = [self.sumtree.get(priorités[i]) for i in range(self.batch_size) ]
        transitions = [tirage[i][0] for i in range(self.batch_size)]
        indices= [tirage[i][1] for i in range(self.batch_size)]
        #on calcule la compensation à ajouter aux TD_errors
        priorités_normalisées = torch.tensor(priorités)/self.sumtree.total()
        compensations = torch.pow(self.sumtree.taille_actuele*priorités_normalisées,-self.beta)
        #on normalise la compensation pour plus de stabilité, N peut être très grand...
        compensations/=compensations.max()
        return transitions,indices,compensations
    def update(self,indices,td_errors):
        td_errors=np.array(td_errors.detach())
        for indice,td_error in zip(indices,td_errors):
            poid=abs(td_error)**self.alpha+1e-5
            self.sumtree.update(self.sumtree.i_data_to_i_arbre(indice),poid)
class Dueling_network(nn.Module):
    def __init__(self,taille_etat,nb_actions,lr):
        super(Dueling_network,self).__init__()
        self.fc1=nn.Linear(taille_etat,128)
        self.fc2=nn.Linear(128,128)
        self.fc3v=nn.Linear(128,128)
        self.fcV=nn.Linear(128,1)
        self.fc3a=nn.Linear(128,128)
        self.fcA=nn.Linear(128,nb_actions)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=lr)
    def forward(self,input):
        """effectue le feedforward en utilisant le dueling"""
        # Si l'input n'a qu'une dimension on ajoute la dimension 
        if input.dim() == 1:
            input = input.unsqueeze(0) # Transforme (8) en (1, 8)
        x=torch.relu(self.fc1(input))
        x=torch.relu(self.fc2(x))
        v=torch.relu(self.fc3v(x))
        v=self.fcV(v)
        x=torch.relu(self.fc3a(x))
        a=self.fcA(x)
        #a représente a quel point l'état est intéressant et v donne a quel point une action est bonne
        #on lui soustrait sa moyenne (dim=1 car il est de taille batchs_sizexnb_actions et qu'on
        #veut le moyenne sur les actions. Ceci nous renvoie un vecteur de taille (64,0) or on veut un 
        #truc de la taille de v, donc on met keepdim a True
        return v+(a-a.mean(dim=1,keepdim=True))
class Agent:
    def __init__(self,state_size,action_size,buffer_capacity,batch_size,alpha,beta,eps,gamma,lr):
        self.net = Dueling_network(state_size,action_size,lr)
        self.goal_net = Dueling_network(state_size,action_size,lr)
        #on copie les données de net dans goal_net
        self.goal_net.load_state_dict(self.net.state_dict())
        self.buffer = Buffer(buffer_capacity,alpha,batch_size,beta)
        self.eps=eps 
        self.gamma=gamma 
        self.lr=lr 
        self.nb_train=0
    def train(self):
        """entraine le réseau sur un minibatch extrait du buffer"""
        transitions,indices,compensations=self.buffer.tirage()
        """Plan : 
            1-forward les actions sur etats
            2-caculer la target avec le goalnet
            3-backprop
            4-update les values
        """
        # 1-forward les actions sur etats
        etats = torch.tensor(np.array([transitions[i].etat for i in range(self.buffer.batch_size)]))
        actions= torch.tensor(np.array([transitions[i].action for i in range(self.buffer.batch_size)]))
        forws = self.net.forward(etats)
        q_values = forws[torch.arange(self.buffer.batch_size),actions]
        # 2-caculer la target avec le goalnet
        #pour calculer les targets, on ne veut pas que les gradients soient modifiés
        with torch.no_grad():
            #on calcule les actions suivantes selon le net actuel mais on calcule leur q_values avec le goalnet pour plus de stabilité
            etats_suivant= torch.tensor(np.array([transitions[i].prochain_etat for i in range(self.buffer.batch_size)]))
            forws_suivant = self.net.forward(etats_suivant)
            #forws_suivant de dim batch_sizexactions donc on veut argmax selon les actions
            next_actions = forws_suivant.argmax(dim=1)
            #on calcule les q values sur le goal net selon les actions choisies par le net normal
            forws_suivant_goal = self.goal_net.forward(etats_suivant)
            q_values_max_goal = forws_suivant_goal[torch.arange(self.buffer.batch_size),next_actions]
            #on vectorise tout pour calculer les target plus vite
            rewards=torch.tensor(np.array([transitions[i].reward for i in range(self.buffer.batch_size)]))
            dones=torch.tensor(np.array([transitions[i].done for i in range(self.buffer.batch_size)]),dtype=torch.float32)
            targets = rewards+self.gamma*(1-dones)*q_values_max_goal
        td_errors = targets-q_values 
        # 3-backprop
        loss = (compensations * F.smooth_l1_loss(q_values, targets, reduction='none')).mean()
        self.net.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(),10)
        self.net.optimizer.step()
        # 4-update les values
        self.buffer.update(indices,td_errors)
        if(self.nb_train%1000==0):
            self.goal_net.load_state_dict(self.net.state_dict())
        self.nb_train+=1

if __name__=="__main__":
    env=gym.make("LunarLander-v3",continuous=False)
    agent = Agent(8,4,65536,128,0.6,0.4,0.999,0.99,1e-4)
    total_reward=0
    liste_reward=[]
    fin=False
    for iter in range(1000):
        etat,_=env.reset()
        done=False 
        if(iter%50==0):
            print(f"iter : {iter}, reward : {total_reward/50}, epsilon : {agent.eps}")
            liste_reward.append(total_reward/50)
            if total_reward/50>220:
                fin=True
            total_reward=0 
        while not done:
            #on tire une action 
            if np.random.random()>agent.eps : 
                #on forward l'état
                etat_forw=torch.tensor(etat,dtype=torch.float32)
                forw=agent.net.forward(etat_forw)
                #on renvoie l'action avec la plus grosse Q_value
                action=torch.argmax(forw).item()
            else:
                action=env.action_space.sample() 
            #on execute l'action
            etat_suivant,reward,terminated,truncated,_=env.step(action)
            #la partie est finie si on a gagné ou si on est sorti
            done = terminated or truncated
            transi = Transi(etat,action,reward,etat_suivant,done)
            agent.buffer.add(transi)
            total_reward+=reward 
            agent.train()
            etat=etat_suivant 
        agent.eps=max(0.1,agent.eps*0.995)
        if fin : 
            break
    plt.plot(liste_reward)
    plt.show()
    env=gym.make("LunarLander-v3",continuous=False,render_mode="human")
    for iter in range(5):
        time.sleep(3)
        etat,_=env.reset()
        done=False 
        total_reward=0
        while not done:
            #on forward l'état
            etat_forw=torch.tensor(etat,dtype=torch.float32)
            forw=agent.net.forward(etat_forw)
            #on renvoie l'action avec la plus grosse Q_value
            action=torch.argmax(forw).item()
            #on execute l'action
            etat_suivant,reward,terminated,truncated,_=env.step(action)
            #la partie est finie si on a gagné ou si on est sorti
            done = terminated or truncated
            total_reward+=reward 
            etat=etat_suivant 
        print(f"la reward totale de cette game est de {total_reward}")
