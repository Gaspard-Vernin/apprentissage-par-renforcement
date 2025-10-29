import numpy as np # used for arrays
import gymnasium as gym # pull the environment
import matplotlib.pyplot as plt
import math
epsilon=1
alpha = 0.1   # taux d'apprentissage
gamma = 0.9  # facteur de discount
episodes=3000
env=gym.make("MountainCar-v0", render_mode=None)
bornes=list(zip(env.observation_space.low,env.observation_space.high))
discret_range=[40,50]
def discretise(etat):
    proportion=[(etat[i]-bornes[i][0])/(bornes[i][1]-bornes[i][0]) for i in range(len(etat))]
    position=[int(round((discret_range[i]-1)*proportion[i])) for i in range(len(etat))]
    for i in range(len(position)):
        position[i] = min(position[i],discret_range[i]-1)
        position[i] = max(position[i],0)
    return tuple(position)
Q=np.random.uniform(low=-1,high=1,size=(discret_range[0],discret_range[1],env.action_space.n))
recompense_totale=0
recompense_storage=[]
for iteration in range(episodes):
    etat,_=env.reset()#créer un nouvel environnemnt
    etat=discretise(etat)
    done=False 
    while not done:
        if np.random.random()>epsilon:#exploitation
            action=np.argmax(Q[etat])
        else:
            action=env.action_space.sample()
        etat_suivant, recompense, done, truncated, info = env.step(action)
        etat_suivant_discret = discretise(etat_suivant)
        recompense_totale+=recompense 
        Q[etat][action] += alpha*(recompense+gamma*np.max(Q[etat_suivant_discret])-Q[etat][action])
        etat=etat_suivant_discret
    if iteration%10==0:
        recompense_storage.append(recompense_totale)
        print(f"tour : {iteration}, score : {recompense_totale/100}")
        recompense_totale=0
    epsilon=max(0.1, epsilon-1/episodes)
plt.figure(figsize=(6,12))
plt.plot(recompense_storage)
plt.show()
for _ in range(3):
    etat,_=env.reset()#créer un nouvel environnemnt
    etat=discretise(etat)
    done=False 
    while not done:
        action=np.argmax(Q[etat])
        etat_suivant, recompense, done, truncated, info = env.step(action)
        etat_suivant_discret = discretise(etat_suivant)
        Q[etat][action] += alpha*(recompense+gamma*np.max(Q[etat_suivant_discret])-Q[etat][action])
        etat=etat_suivant_discret
