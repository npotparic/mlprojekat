import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6 , alfa = 0.5 , gama = 0.6):
        self.nA = nA
        self.alfa = alfa
        self.gama = gama
        self.Q =defaultdict(lambda: np.zeros(self.nA))
#biranje akcije epsilon greedy politikom
    def izaberi(self,env,eps,stanje):
        if(random.random() > eps): 
            return (np.argmax(self.Q[stanje]))
        else:
            return (random.choice(np.arange(env.action_space.n))) 
        
    def update(self , alfa ,gama ,  Q , stanje , akcija , nagrada , sledeces = None):
            trenutno = self.Q[stanje][akcija]
            if sledeces== None:
                naredno=0
            else:
                naredno = np.max(Q[sledeces])
            return trenutno + (alfa * (nagrada + (gama * naredno)- trenutno))

    def step(self, alfa  , gama , stanje, akcija, nagrada, sledece, gotovo):
        self.Q[stanje][akcija] += self.update(self , alfa , gama , self.Q  ,stanje , akcija , nagrada , sledece)