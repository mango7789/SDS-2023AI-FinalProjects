import importlib
import numpy as np
import random

class Conf:
    banditCount = None
    def __init__(self, count):
        self.banditCount = count

class Obs:
    step = None
    reward = None
    lastActions = None
    agentIndex = None
    def __init__(self):
        self.step = 0
        self.reward = 0
        self.lastActions = [0,0]


def simulate(ag1, param1, ag2, param2):
    agent2 = importlib.import_module(ag2)
    for key, value in param2.items():
        setattr(agent2, key, value)
    agent1 = importlib.import_module(ag1)
    for key, value in param1.items():
        setattr(agent1, key, value)
    bandits_probab = np.random.rand(100)
    configuration1 = Conf(100)
    configuration2 = Conf(100)
    observation1 = Obs()
    observation1.agentIndex =0
    observation2 = Obs()
    observation2.agentIndex =1
    for step in range(2000):
        b1 = agent1.agent(observation1, configuration1)
        b2 = agent2.agent(observation2, configuration2)
        w1 = int(np.random.rand() < bandits_probab[b1])
        w2 = int(np.random.rand() < bandits_probab[b2])
        bandits_probab[b1] *= 0.97
        bandits_probab[b2] *= 0.97
        observation1.reward += w1
        observation2.reward += w2
        observation1.step = step
        observation2.step = step
        observation1.lastActions = [b1, b2]
        observation2.lastActions = [b2, b1]
        # print('result:',observation1.reward,observation2.reward)
    return (observation1.reward, observation2.reward)


agent1 =  ["agent_7",{}]
agents = [
     ["agent_7",{}],
     ]

n_rounds = 20
wins1 = [0] * len(agents)
wins2 = [0] * len(agents)
sum1 = [0] * len(agents)
sum2 = [0] * len(agents)
# for agent in agents:
#     print("\t", agent ,end='')
# print()
for i in range(n_rounds):
    print(i+1, ":\t\t" ,end='')
    for j, agent in enumerate(agents):
        w1 ,w2  = simulate(agent1[0], agent1[1], agent[0], agent[1])
        #print(agent1, ':' , mygame[1999][0].observation.reward, end=" ")
        #print(agent2, ':' , mygame[1999][1].observation.reward, ,end='\t')
        sum1[j] += w1
        sum2[j] += w2
        if w1 > w2:
            wins1[j] +=1
        if w1 < w2:
            wins2[j] +=1
        print(wins1[j],wins2[j], end='\t')
    print()

for j, agent in enumerate(agents):
    print(f'{agent[0]+str(agent[1].get("x1"))+str(agent[1].get("x2")):<22}',wins1[j],wins2[j] ,sum1[j]/ n_rounds ,sum2[j] / n_rounds, sum1[j]/sum2[j], sep="\t")
