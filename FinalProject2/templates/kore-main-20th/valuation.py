from momentum import NeoMomentum
import math
import numpy as np
from utils import PAR, DistanceManager
from kaggle_environments.envs.kore_fleets.helpers import *

ramp = np.ones((801), dtype=np.intc)
ramp[2:7] = 2
ramp[7:17] = 3
ramp[17:34] = 4
ramp[34:60] = 5
ramp[60:97] = 6
ramp[97:147] = 7
ramp[147:212] = 8
ramp[212:294] = 9
ramp[294:] = 10

discount_table = np.ones((401), dtype=np.single)
discount = 1
for i in range(1, 401):
    discount *= PAR['discount_rate']
    discount_table[i] = discount

class Valuation:
    def __init__(self, mom: NeoMomentum, player: int, step: int):
        self.mom = mom
        self.step = step
        self.me = player
        self.enemy = abs(player-1)
        self.turns_remaining = 400 - mom.board.step
        self._avg_fleet_size = None
        self._marginal_mining_rate = None
        self._total_ships = None
        self._total_fleets = None
        self._kore_total = None
        self._avg_kore_per_square = None
        self._expected_income = None
        self._ships = None
        self._ship_values = None
        self._ship_npv = None
        self._fleets = None
        self._npv_new_base = None
        self._network_capacity = None
        self._starvation_point = None
        self.discount_table = discount_table


    @property
    def network_capacity(self):
        if self._network_capacity is None:
            self._network_capacity = []
            for player in range(0, 2):
                # Capacity of network
                capacity = np.zeros((self.turns_remaining), dtype=np.single)
                time_max = min(self.mom.lookahead + 1, self.turns_remaining)
                for t in range(0, time_max):
                    for yard in self.mom.yards[player][t]:
                        if not yard.captured_start:
                            capacity[t] += ramp[yard.turns_controlled+t]
                if time_max < self.turns_remaining:
                    for yard in self.mom.yards[player][self.mom.lookahead]:
                        capacity[time_max:self.turns_remaining] += ramp[yard.turns_controlled:yard.turns_controlled + (
                                    self.turns_remaining - time_max)]
                self._network_capacity.append(capacity)
        return self._network_capacity

    def calc_capacity_delta_npv(self, player, fleet_size, additional=True, free=False, delay=True, first_turn=0, turns_controlled=0, bias=1.0, captured=False, capture_turn=0):
        capacity = self.network_capacity[player].copy()

        # Capacity of network after increasing network by one yard in t=10
        if additional:
            new_capacity = capacity.copy()
            if delay and self.turns_remaining > PAR['build_delay']:
                new_capacity[max(first_turn, PAR['build_delay']):] = new_capacity[max(first_turn, PAR['build_delay']):] + ramp[:self.turns_remaining - max(first_turn, PAR['build_delay']):]
            elif captured:
                new_capacity[min(self.mom.lookahead, capture_turn + 1):] = capacity[min(self.mom.lookahead, capture_turn + 1):] - \
                            ramp[turns_controlled + capture_turn:][:len(capacity[min(self.mom.lookahead, capture_turn + 1):])]
            else:
                new_capacity[first_turn:] = new_capacity[first_turn:] + ramp[:len(new_capacity[first_turn:])]
        else: # See what happens if we lose the base instead
            new_capacity = capacity.copy()
            new_capacity[first_turn:] = new_capacity[first_turn:] - ramp[turns_controlled:len(new_capacity)+turns_controlled-first_turn]

        # Penalize capacity for fleet launches
        fleet_count = self.fleets[player]

        if captured:
            yard_count = 0
            for yard in self.mom.yards[player][min(self.mom.lookahead, capture_turn+1)]:
                if not yard.captured_start:
                    yard_count += 1
        else:
            yard_count = len(self.mom.yards[player][self.mom.lookahead])

        lost_time = (1 / PAR['avg_flight_time']) * fleet_count / max(1, yard_count)
        capacity *= (1 - lost_time)

        if additional and not captured:
            new_lost_time = (1 / PAR['avg_flight_time']) * (fleet_count - 1 ) / (yard_count + 1)
            if delay:
                new_capacity[PAR['build_delay']:] = new_capacity[PAR['build_delay']:] * (1 - new_lost_time)
            else:
                new_capacity = new_capacity * (1 - new_lost_time)
        elif additional and captured:
            new_lost_time = (1 / PAR['avg_flight_time']) * (fleet_count - 1 ) / max(1, (yard_count - 1))
            if delay:
                new_capacity[PAR['build_delay']:] = new_capacity[PAR['build_delay']:] * (1 - new_lost_time)
            else:
                new_capacity = new_capacity * (1 - new_lost_time)
        else:
            new_lost_time = (1 / PAR['avg_flight_time']) * (fleet_count + 0) / max(1, (yard_count - 1))
            new_capacity = new_capacity * (1 - new_lost_time)

        kore_statusquo = self.mom.kore_bank[player][min(10, self.mom.lookahead)]
        kore_new = self.mom.kore_bank[player][min(10, self.mom.lookahead)]
        kore_starvation = self.mom.kore_bank[player][min(10, self.mom.lookahead)]
        ships_built_statusquo = np.zeros((self.turns_remaining), np.intc)
        ships_built_new = np.zeros((self.turns_remaining), np.intc)
        ships_built_starvation = np.zeros((self.turns_remaining), np.intc)

        # Find builds which are already planned
        planned_builds = 0
        for fleet in self.mom.fleets[player][0]:
            if 'C' in fleet.flight_plan_str and fleet.ship_count >= 50:
                planned_builds += 1

        # print(self.my_expected_income)
        # if additional:
        #     new_expected_income = np.maximum(0, self.expected_income[player] * (self.ships[player] - 50) / max(1, self.ships[player]))
        # else:
        #     new_expected_income = self.expected_income[player]
        # starvation_expected_income = self.expected_income[player] * 0.0
        if additional and not free:
            new_ships = max(1, self.ships[player] + fleet_size - (planned_builds + 1) * 50)
        else:
            new_ships = self.ships[player] + fleet_size - planned_builds * 50

        for kore, ships_built, capacity in zip([kore_statusquo, kore_new, kore_starvation], [ships_built_statusquo, ships_built_new, ships_built_starvation], [capacity, new_capacity, capacity]):
            for t in range(0, self.turns_remaining):
                if t > 0:
                    kore += self.expected_income[player][t] * new_ships
                ships_built[t] = min(math.floor(max(kore, 100000) / 10), capacity[t])
                new_ships += ships_built[t]
                kore -= ships_built[t] * 10

        starvation = np.where(ships_built_starvation == 0)
        if len(starvation) >= 1 and len(starvation[0]) >= 1:
            self._starvation_point = int(starvation[0][0])

        worth_building = np.ones((self.turns_remaining), dtype=np.intc)
        worth_building[self.ship_npv <= 0.0] = 0

        for ships_built in [ships_built_statusquo, ships_built_new]:
            ships_built *= worth_building

        npv_statusquo = np.sum(ships_built_statusquo * self.ship_npv * discount_table[:len(self.ship_npv)])
        npv_new = np.sum(ships_built_new * self.ship_npv * discount_table[:len(self.ship_npv)])

        if not free:
            cost = 50 * self.ship_values[0]
            return bias * (npv_new - npv_statusquo) - cost
        else:
            return bias * max(0, npv_new - npv_statusquo, npv_statusquo - npv_new)

    @property
    def starvation_point(self):
        return self._starvation_point

    @property
    def kore_total(self):
        if not self._kore_total:
            self._kore_total = np.sum(self.mom.kore[0,:,:])
            self._avg_kore_per_square = self._kore_total / (21*21)
        return self._kore_total

    @property
    def avg_kore_per_square(self):
        if not self._avg_kore_per_square:
            x = self.kore_total
        return self._avg_kore_per_square

    @property
    def total_ships(self):
        if not self._total_ships:
            cum_total = 0
            for player in range(0, 2):
                for fleet in self.mom.fleets[player][0]:
                    cum_total += fleet.ship_count
            self._total_ships = cum_total
        return self._total_ships

    @property
    def ships(self):
        if self._ships is None:
            self._ships = []
            cum_total = 0
            for fleet in self.mom.fleets[self.me][0]:
                cum_total += fleet.ship_count
            if self.me == 0:
                self._ships.append(cum_total)
                self._ships.append(np.sum(self.mom.enemy_phantom[0,:,:]) + np.sum(self.mom.enemy_fleet[0,:,:]))
            else:
                self._ships.append(np.sum(self.mom.enemy_phantom[0, :, :]) + np.sum(self.mom.enemy_fleet[0, :, :]))
                self._ships.append(cum_total)
        return self._ships

    @property
    def avg_fleet_size(self):
        if self._avg_fleet_size is None:
            self._fleets = []
            cum_total = 0
            self._total_fleets = 0
            my_fleets = 0
            for fleet in self.mom.fleets[self.me][0]:
                if not fleet.finished_end:
                    cum_total += fleet.ship_count
                    self._total_fleets += 1
                    my_fleets += 1
            cum_total += np.sum(self.mom.enemy_fleet[0,:,:]) + np.sum(self.mom.enemy_phantom[0,:,:])
            enemy_fleets = np.count_nonzero(self.mom.enemy_fleet[0,:,:]) + np.count_nonzero(self.mom.enemy_phantom[0,:,:])
            self._avg_fleet_size = max(2, round(cum_total / max(1, (enemy_fleets + my_fleets))))
            if self.me == 0:
                self._fleets.append(my_fleets)
                self._fleets.append(enemy_fleets)
            else:
                self._fleets.append(enemy_fleets)
                self._fleets.append(my_fleets)
            self._total_fleets = sum(self._fleets)
        return self._avg_fleet_size

    @property
    def fleets(self):
        if self._fleets is None:
            x = self.avg_fleet_size
        return self._fleets

    @property
    def total_fleets(self):
        if self._total_fleets is None:
            x = self.avg_fleet_size
        return self._total_fleets

    @property
    def marginal_mining_rate(self):
        if self._marginal_mining_rate is None:
            self._marginal_mining_rate = (.99, math.log(self.avg_fleet_size + 1) / 20) - (.99, max(1.0, math.log(self.avg_fleet_size)) / 20)
        return self._marginal_mining_rate

    @property
    def expected_income(self):
        if self._expected_income is None:
            self._expected_income = [np.zeros((self.turns_remaining), dtype=np.single), np.zeros((self.turns_remaining), dtype=np.single)]
            mining_rate = min(.99, math.log(self.avg_fleet_size + 1) / 20)
            avg_kore_ps = self.avg_kore_per_square
            for t in range(0, self.turns_remaining-3):
                my_mined = mining_rate * self.total_fleets * (self.ships[0] / max(1, self.total_ships)) * avg_kore_ps
                enemy_mined = mining_rate * self.total_fleets * (self.ships[1] / max(1, self.total_ships)) * avg_kore_ps
                avg_kore_ps -= (my_mined + enemy_mined) / (21*21)
                avg_kore_ps = min(500, avg_kore_ps*1.01)
                self._expected_income[0][t] = my_mined
                self._expected_income[1][t] = enemy_mined
        return self._expected_income

    @property
    def ship_values(self):
        if self._ship_values is None:
            self._ship_values = np.zeros((self.turns_remaining), dtype=np.single)
            self._ship_npv = np.zeros((self.turns_remaining), dtype=np.single)
            income_per_ship = np.add(self.expected_income[0], self.expected_income[1]) / max(4, self.total_ships)
            for t in range(0, self.turns_remaining):
                self._ship_values[t] = np.sum(np.divide(income_per_ship[t:self.turns_remaining], discount_table[0:self.turns_remaining-t]))
                self._ship_npv[t] = np.sum(
                    np.divide(income_per_ship[t:self.turns_remaining], discount_table[0:self.turns_remaining - t])) - 10/(discount_table[t])
            maxidx = np.argmax(self._ship_npv)
            self._ship_npv[:maxidx] = self._ship_npv[maxidx]
            self._ship_values[:maxidx] = self._ship_values[maxidx]
        return self._ship_values

    def income_maximizing_size(self, player, best_paths, max_spawn, maximum_size):
        current_fleet_count = len(self.mom.fleets[player][0])
        best_val = -999999
        best_path = None

        if maximum_size is None or maximum_size >= len(best_paths):
            largest_path = best_paths[len(best_paths) - 1]
        else:
            largest_path = best_paths[maximum_size]

        for path in best_paths:
            if path[2] <= -999999:
                continue
            size = path[0]
            largest_size = largest_path[0]
            t = path[1]
            largest_t = largest_path[1]
            total_val = path[2]
            largest_total_val = largest_path[2]
            strat_val = path[3]
            largest_strat_val = largest_path[3]
            other_val = total_val - strat_val

            if strat_val < largest_strat_val:
                continue

            adjusted_value = other_val / size / max(1, t)

            if adjusted_value >= best_val and adjusted_value > -999999:
                best_val = adjusted_value
                best_path = path
        return best_path

    def income_maximizing_size_archive(self, player, best_paths, max_spawn, maximum_size):
        current_fleet_count = len(self.mom.fleets[player][0])
        best_val = -999999
        best_path = None

        if maximum_size is None or maximum_size >= len(best_paths):
            largest_path = best_paths[len(best_paths) - 1]
        else:
            largest_path = best_paths[maximum_size]

        for path in best_paths:
            if path[2] <= -999999:
                continue
            size = path[0]
            largest_size = largest_path[0]
            t = path[1]
            largest_t = largest_path[1]
            total_val = path[2]
            largest_total_val = largest_path[2]
            strat_val = path[3]
            largest_strat_val = largest_path[3]
            other_val = total_val - strat_val
            largest_other_val = largest_total_val - largest_strat_val
            ratio_to_largest = largest_size // size

            if self.ship_npv[0] > 0:
                build_loss = self.ship_npv[0] * max_spawn * (PAR['discount_rate'] - 1) # Cost of delaying ship build by one turn
            else:
                build_loss = 0.0

            strat_val_loss = largest_strat_val - strat_val
            mining_loss = largest_other_val * (largest_size - size) / largest_size * (PAR['discount_rate'] - 1)

            adjusted_value = ratio_to_largest*other_val + strat_val - build_loss - strat_val_loss - mining_loss

            if adjusted_value >= best_val and adjusted_value > -999999:
                best_val = adjusted_value
                best_path = path
        return best_path

    @property
    def ship_npv(self):
        if self._ship_npv is None:
            x = self.ship_values
        return self._ship_npv

