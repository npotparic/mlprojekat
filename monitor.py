
import sys
import math
import numpy as np
import agent as a


def igraj(env, agent, br=100, brepizoda=30000,alfa = 0.6):
    # inicijalizovati prosecne nagrade, najnovije nagrade i najbolju nagradu
    prosecne=list()
    najnovije = list()
    najbolja = 0
    for epizoda in range(1, brepizoda+1):
        score = 0
        stanje = env.reset()
        eps = 0.6/epizoda
        while True:
            # agent bira akciju
            akcija = agent.izaberi(env,eps,stanje)
            sledece ,nagrada,gotovo, info = env.step(akcija)
            score += nagrada
            agent.Q[stanje][akcija] = agent.update(0.5 ,0.6 , agent.Q ,stanje,akcija,nagrada,sledece)
            stanje = sledece
            if gotovo==True:
                najnovije.append(score)
                break
        if (epizoda>= 100):
            # dobijamo prosecnu nagradu za poslednjih 100 epizoda
            prosecna= np.mean(najnovije)
            prosecne.append(prosecna)
            # azuriramo najbolju prosecnu nagradu
            if prosecna> najbolja:
                najbolja = prosecna
        # pratiti napredak
        print("\rEpizoda {}/{} || Najbolja prosečna nagrada {}".format(epizoda, brepizoda, najbolja), end="")
        sys.stdout.flush()
        # proveriti da li je zadatak resen (prema OpenAI Gym)
        if najbolja >= 9.4:
            print('\nOkruženje rešeno u {} epizode.'.format(epizoda), end="")
            break
        if epizoda == brepizoda: print('\n')
    return prosecne, najbolja