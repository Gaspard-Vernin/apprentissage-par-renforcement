import numpy as np # used for arrays
import gymnasium as gym # pull the environment
import matplotlib.pyplot as plt
alpha = 0.1   # taux d'apprentissage
gamma = 0.9  # facteur de discount
epsilon=0.99 # facteur exploration/exploitation
episodes=8000 #nombre d'episodes
env = gym.make("CartPole-v1")
#[position_du_chariot, vitesse_du_chariot, angle_du_bâton, vitesse_angulaire_du_bâton]
discretisation = [10,10,15,15]
#si param1 = [a1,b1,c1] param2= [a2,b2,c2] alors list(zip(param1,param2))=[(a1,a2),(b1,b2),(c1,c2)]
bornes_valeurs=list(zip(env.observation_space.low,env.observation_space.high))
"""
problème: la vitesse du chariot et la vitesse angulaire du baton son non bornées
donc potentiellement on envoie des calculs avec des infini dans la Q table ce qui est 
pas ok donc on préfère les remplacer par des bornes 
"""
#on borne la vitesse du chariot entre -3 et 3
bornes_valeurs[1]=[-3,3]
#on borne la vitesse angulaire entre -5 et 5
bornes_valeurs[3]=[-5,5]
def tracer_courbe(tab_recompenses):
    plt.figure(figsize=(12,6))
    plt.plot(tab_recompenses, marker='o', linestyle='', markersize=4, color='blue')
    plt.xlabel("Épisode")
    plt.ylabel("Récompense totale")
    plt.title("Courbe d'apprentissage du Q-learning")
    plt.grid(True)
    plt.show()
def discretise_etats(etats):
    #pour chaque etat, on trouve a quel point il est grand, 1 si au max, 0 si au min, linéaire entre les deux
    proportions = [(etats[i]-bornes_valeurs[i][0])/(bornes_valeurs[i][1]-bornes_valeurs[i][0]) for i in range(len(etats))]
    #discretisation-1 pour commencer & compter. * proportion nous place sur la bonne case
    nouvel_etat = [int(round((discretisation[i]-1)*proportions[i])) for i in range(len(etats))]
    for i in range(len(nouvel_etat)):
        if nouvel_etat[i] >discretisation[i]-1 : 
            nouvel_etat[i]=discretisation[i]-1
        elif nouvel_etat[i]<0:
            nouvel_etat[i]=0
    #on veut un tuple pour pouvoir l'utiliser comme indice d'un tableau
    return tuple(nouvel_etat)
#on crée la table avec la dimension des cases et les actions
Q_table = np.random.uniform(low=-1,high=1,size=(discretisation[0],discretisation[1],discretisation[2],
                                                discretisation[3],env.action_space.n))
#reset de l'environnement
stockage_reward=[]
#quand on perd, done devient true
scores_100 = []  # ajout d'une liste pour stocker les scores des 100 derniers épisodes
for iteration in range(episodes):
    etat,_=env.reset()
    etat=discretise_etats(etat)
    done = False 
    recompense_totale=0  # on remet à zéro à chaque épisode
    while not done:
        if np.random.random() > epsilon:
            #on prend le mouvement conseillé par la qtable
            action = np.argmax(Q_table[etat])
        else: 
            #on explore
            action = env.action_space.sample()
        #etat suivant : [pos chariot, vitesse chariot, angle, vitessse angulaire] après le mouvement
        #recompense : 1 si le baton est encore debout, 0 sinon
        #done : true si perdu, false sinon
        #truncated : true si on a un arrêt dû à une victoire, pas à un baton tombé
        #info : dictionnaire d'infos de debug/ utilitaires
        etat_suivant, recompense, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        etat_suivant_discret = discretise_etats(etat_suivant)
        Q_table[etat][action]+=alpha*(recompense+gamma*np.max(Q_table[etat_suivant_discret])-Q_table[etat][action])
        etat=etat_suivant_discret
        recompense_totale+=recompense
    scores_100.append(recompense_totale)
    if(iteration%100==0 and iteration>0):
        moyenne = np.mean(scores_100)
        stockage_reward.append(moyenne)
        print(f"Épisode {iteration}, score = {moyenne}")
        scores_100=[]
    epsilon*=0.999 #on explore de moins en moins    
tracer_courbe(stockage_reward)

#visualisation du résultat
env=gym.make("CartPole-v1",render_mode="human")
etat,_=env.reset()
etat=discretise_etats(etat)
done=False 
recompense_totale=0 
while not done:
    action=np.argmax(Q_table[etat])
    etat_suivant,recompense,done,truncated,info=env.step(action)
    etat_suivant_discret=discretise_etats(etat_suivant)
    Q_table[etat][action]+=alpha*(recompense+gamma*np.max(Q_table[etat_suivant_discret])-Q_table[etat][action])
    etat=etat_suivant_discret 
    recompense_totale+=recompense
