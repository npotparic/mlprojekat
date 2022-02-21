from agent import Agent
from monitor import igraj
import gym
import numpy as np
import  matplotlib.pyplot as plt

env = gym.make('Taxi-v3') # Ucitavanje okruzenja
agent = Agent()  # Kreiranje instance agent
prnagrade,najbolja = igraj(env, agent) # Treniranje agenta

#Prikazivanje igre 
def render():
    stanje = env.reset()
    score = 0
    while True:
        env.render()
        akcija = agent.izaberi(env,0,stanje)
        novostanje,nagrada,gotovo,info = env.step(akcija)
        score+=nagrada
        if gotovo==True:
            break
        stanje = novostanje
    print('Score {}'.format(score))

render()