from kaggle_environments.envs.kore_fleets.helpers import *
from kaggle_environments.envs.kore_fleets.helpers import Board

class Interpreter():
    def __init__(self, configuration, observation, agents, run_data = None, ff_to=0):
        self.configuration = configuration
        self.observation = observation
        self.agents = agents
        self.board = Board(self.observation, self.configuration)
        self.run_data = run_data
        self.ff_to = ff_to

        if run_data and ff_to > 0:
            for i in range(1, min(ff_to, len(run_data['steps']))):
                for p in range(0, 2):
                    for shipyard in self.board.players[p].shipyard_ids:
                        if run_data['steps'][i][p]['action'] is not None and shipyard in run_data['steps'][i][p]['action'].keys():
                            self.board.shipyards.get(shipyard).next_action = ShipyardAction.from_str(run_data['steps'][i][p]['action'][shipyard])
                self.board = self.board.next()

        #neo = NeoMomentum(self.board, 15)

    def active(self) -> bool:
        ship_count = 0
        if self.board.step == self.board.configuration.episode_steps:
            return False
        for player in self.board.players.values():
            ship_count = 0
            for yard in player.shipyards:
                ship_count += yard.ship_count
            if ship_count == 0 and len(player.fleets) == 0 and (len(player.shipyards) == 0 or player.kore < self.board.configuration.spawn_cost):
                return False
        return True

    def winner(self) -> int:
        ship_count = 0
        for player in self.board.players.values():
            for yard in player.shipyards:
                ship_count += yard.ship_count
            if ship_count == 0 and len(player.fleets) == 0 and (len(player.shipyards) == 0 or player.kore < self.board.configuration.spawn_cost):
                return abs(player.id - 1)
        if self.board.players[0].kore > self.board.players[1].kore:
            return 0
        else:
            return 1

    def step(self) -> Board:
        if self.active():
            self.observation = self.board.observation
            for idx, agent in zip(range(0, len(self.agents)), self.agents):
                self.observation['player'] = idx
                actions = agent(self.observation, self.configuration)
                if actions:
                    for yard in self.board.players[idx].shipyards:
                        if yard.id in actions.keys():
                            yard.next_action = ShipyardAction.from_str(
                                actions[yard.id])
                else:
                    pass
            self.board = self.board.next()
            #print(self.board)
        return self.board