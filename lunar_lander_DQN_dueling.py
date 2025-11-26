import gymnasium as gym 
import numpy as np 
import torch 
import matplotlib.pyplot as plt
from collections import deque
import random
class Buffer():
    def __init__(self,batch_size,max_len):
        self.buffer=deque(maxlen=max_len)
        self.batch_size=batch_size 
    def add(self, transi):
        self.buffer.append(transi)
    def get_echantillon(self):
        return random.sample(population=self.buffer,k=self.batch_size)
class Transition():
    def __init__(self,etat,prochain_etat,action,done,reward):
        self.etat=etat
        self.prochain_etat=prochain_etat 
        self.action=action 
        self.done=done 
        self.reward=reward 
class Net(torch.nn.Module):
    def __init__(self,gamma,epsilon,lr,batch_size_buffer,max_len_buffer,valeur_update_goalnet,dueling):
        super(Net,self).__init__()
        self.gamma=gamma 
        self.epsilon = epsilon 
        self.buffer = Buffer(batch_size_buffer,max_len_buffer)
        self.dueling = dueling
        nb_actions = 4 
        taille_etat = 8
        self.fc1 = torch.nn.Linear(taille_etat,256)
        self.fc2 = torch.nn.Linear(256,256)
        if dueling : 
            self.value_fc = torch.nn.Linear(256,256)
            self.value = torch.nn.Linear(256,1)

            self.advantage_fc = torch.nn.Linear(256,256)
            self.advantage = torch.nn.Linear(256,nb_actions)
        else :       
            self.fc3= torch.nn.Linear(256,nb_actions)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        self.compteur_appel=0
        self.valeur_update_goalnet = valeur_update_goalnet
        self.nb_update=0
    def forward(self,input):
        #on attend qqchose en deux dimensions pour les minis batchs qui permet d'appeler a.mean 
        #quand on forward juste pour trouver l'etat suivant a.mean ne marche plus, pour simplifier
        #on crée juste une deuxieme dimension qui sert à rien donc ca bloque plus a.mean
        if input.dim()==1 : 
            input = input.unsqueeze(0)
        if self.dueling : 
            x=self.fc1(input)
            x=torch.relu(x)
            x=self.fc2(x)
            x=torch.relu(x)
            v=self.value_fc(x)
            v=torch.relu(v)
            v = self.value(v)
            a = self.advantage_fc(x) 
            a = torch.relu(a)
            a = self.advantage(a)
            #on renvoie la valeur de l'endroit où on se trouve ie V plus la différence entre l'action
            #choisie et la moyenne des actions
            q_value = v + (a-a.mean(dim=1,keepdim=True))
            return q_value
        else :
            x=self.fc1(input)
            x=torch.relu(x)
            x=self.fc2(x)
            x=torch.relu(x)
            return self.fc3(x)
    def backprop(self,target,output):
        error_function =torch.nn.MSELoss()
        loss = error_function(output,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
    def train(self,net_goal):
        if len(self.buffer.buffer)<self.buffer.batch_size : 
            return 
        echantillon = self.buffer.get_echantillon() 
        etats=np.array([t.etat for t in echantillon])
        actions=torch.tensor([t.action for t in echantillon],dtype=torch.long)
        q_values=self.forward(torch.tensor(etats,dtype=torch.float32))
        # gather sélectionne les éléments le long d'une dimension donnée, ici on veut les Q-values des actions choisies.
        # Comme gather attend un tensor 2D pour dim=1, on transforme la liste d'actions en tensor colonne avec unsqueeze(1),
        # puis on retire cette dimension inutile avec squeeze(1) pour obtenir un tensor 1D.
        Q_value_etat=q_values.gather(1,actions.unsqueeze(1)).squeeze(1)
        """
        Ici, le problème c'est que je calcule la reward sur la Q_value max du prochain etat
        qui est peut être sur évaluée, alors on utilise net pour choisir l'action mais
        on l'évalue avec net_goal qui est plus stable
        """
        with torch.no_grad():
            prochain_etats=np.array([t.prochain_etat for t in echantillon])
            #net choisit l'action, la dim 1 c'est les actions 
            next_actions=self.forward(torch.tensor(prochain_etats,dtype=torch.float32)).argmax(dim=1)
            #on selectionne les Q_value pour chaque action apres avoir forward sur net_goal
            next_actions_goal_net = net_goal.forward(torch.tensor(prochain_etats,dtype=torch.float32))
            Q_value_max_prochain_etat = next_actions_goal_net.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            rewards = torch.tensor([t.reward for t in echantillon], dtype=torch.float32)
            dones = torch.tensor([t.done for t in echantillon], dtype=torch.float32)
            target = rewards + self.gamma * (1 - dones) * Q_value_max_prochain_etat

        self.backprop(target,Q_value_etat)
        self.compteur_appel+=len(echantillon)
        self.nb_update+=len(echantillon)
        if self.compteur_appel>=self.valeur_update_goalnet :
            net_goal.load_state_dict(self.state_dict())
            self.compteur_appel=0
if __name__ == "__main__":
    env=gym.make("LunarLander-v3",continuous=False)
    net=Net(0.99,1,1e-3,64,10000,5000,True)
    goal_net=Net(0.99,1,1e-3,64,10000,5000,True)
    goal_net.load_state_dict(net.state_dict())
    liste_reward=[]
    total_reward=0
    for iter in range(600):
        etat,_=env.reset()
        done=False 
        if(iter%50==0 and iter!=0):
            print(f"iter : {iter}, reward : {total_reward/50}, epsilon : {net.epsilon}")
            liste_reward.append(total_reward/50)
            total_reward=0  
        while not done :
            if np.random.random() > net.epsilon :
                forw = net.forward(torch.tensor(etat,dtype=torch.float32))
                action = torch.argmax(forw).item() 
            else :
                action = env.action_space.sample()
            etat_suivant,reward,terminated,truncated,_ = env.step(action)
            done = terminated | truncated 
            transi  = Transition(etat,etat_suivant,action,done,reward)
            net.buffer.add(transi)
            total_reward+=reward
            net.train(goal_net)
            etat=etat_suivant
        net.epsilon = max(0.1,net.epsilon*0.995)

    env=gym.make("LunarLander-v3",render_mode="human")
    for iter in range(3):
        etat,_=env.reset()
        done=False 
        total_reward=0
        while not done :
            forw = goal_net.forward(torch.tensor(etat,dtype=torch.float32))
            action = torch.argmax(forw).item()
            etat_suivant,reward,terminated,truncated,_ = env.step(action)
            done = terminated | truncated 
            total_reward+=reward
            etat=etat_suivant
        print(f"recompense : {total_reward}")
        total_reward=0
    print(f"nombre d'update totale du réseau : {net.nb_update}")
