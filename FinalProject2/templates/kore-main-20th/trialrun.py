import json
from ayb import agent
from kaggle_environments.envs.kore_fleets.helpers import *
from os.path import exists
import os
from interpreter import Interpreter



from momentum import NeoMomentum
from kaggle_environments.envs.kore_fleets.kore_fleets import balanced_agent
from kaggle_environments.envs.kore_fleets.helpers import Board

file = open("run.json", "r")
with file:
    data = file.read()
run_data = json.loads(data)

import cProfile as profile


step = 10
early_terminate = 1

if exists('kill_0.txt'):
    os.remove('kill_0.txt')
if exists('kill_1.txt'):
    os.remove('kill_1.txt')
pret = Interpreter(run_data['configuration'], run_data['steps'][0][0]['observation'], [agent, balanced_agent], run_data, ff_to=207)

pr = profile.Profile()


# mom = NeoMomentum(pret.board, 40)
#
# pr.enable()
# for i in range(0, 100):
#     mom.make_source_matrices()

while pret.active() and early_terminate > 0:
    print("Step: " + str(step))
    pr.enable()
    pret.step()
    pr.disable()
    step += 1
    early_terminate -= 1

pr.dump_stats('profile_v.pstat')