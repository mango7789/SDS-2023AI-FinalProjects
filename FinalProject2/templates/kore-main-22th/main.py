from kaggle_environments.envs.kore_fleets.helpers import Board
from config.constant import Constant
from tools.persistance import init_persist, update_persist
from tools.actions import attack, ship_in_need, spawn_miner, spawn_shipyard, low_ship_count, launch_fleet_combat, planned_shipyards,\
    mv2frontline, create_ships_new, create_ships_all
from tools.base_funcs.logger import time_info, prt_time
import time

persist = init_persist(Constant())
def agent(obs, config):
    global persist
    agent_t = time.process_time()
    board = Board(obs, config)
    update_persist(obs, config, persist, board)
    prt_time(persist, time.process_time() - agent_t, "update_persist")

    for func in (create_ships_new, ship_in_need, planned_shipyards, attack,  launch_fleet_combat,
                 low_ship_count, spawn_shipyard, mv2frontline, spawn_miner, create_ships_all):
        func(persist, board)
        prt_time(persist, time.process_time() - agent_t, func.__name__)

    time_info(persist, board, time.process_time() - agent_t)
    return board.current_player.next_actions
