import gym
import numpy as np
env = gym.make('Taxi-v3')

stanje = env.reset()
i=0
while i < 100:
    akcija = env.action_space.sample() #Biramo nasumicno akciju iz action_space
    stanje, nagrada, gotovo,info = env.step(akcija) # izvrsavanje akcije
    env.render() # Stampanje okruzenja
    i+=1
    if gotovo == True:
        print('Score: ', i+1)
        break
        
env.close()