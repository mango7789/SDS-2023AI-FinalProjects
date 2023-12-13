from cy_graph3 import build_graph
from collections import deque
from kaggle_environments.envs.kore_fleets.helpers import *
from momentum import *
from utils import *


class Graph():
    def __init__(self, player_id: int, mom: NeoMomentum, origin: Point, fleet_starting_size: int, base_create=False,
                 kill_switch=False, time_constraint=None):
        self.mom = mom
        self.player_id = player_id
        self.enemy_id = abs(1 - player_id)
        self.fleet_starting_size = fleet_starting_size
        # Maximum circular path length
        self.max_circular_path = self.mom.board.configuration.size
        self.max_path_length = math.floor(
            2 * math.log(self.fleet_starting_size)) + 1
        self.board_size = self.mom.board.configuration.size
        self.origin = origin
        self.base_create = base_create
        if time_constraint is None:
            self.time_constraint = self.mom.lookahead
        else:
            self.time_constraint = time_constraint
        if kill_switch:
            self.time_constraint = 20

        if base_create:
            P_DEPTH = self.max_path_length - 1
        else:
            P_DEPTH = self.max_path_length + 2

        # Node tracking matrices
        self.n_kore_gained = np.ones((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                  dtype=np.single) * BAD_VAL  # Best value at node
        self.n_kore_lost = np.zeros((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                  dtype=np.single)  # Best value at node
        self.n_turn_penalty = np.zeros((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                  dtype=np.single)   # Best value at node
        self.n_cargo_destroyed = np.ones((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                      dtype=np.single) * BAD_VAL  # Best value at node
        self.n_ships_destroyed = np.zeros((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                      dtype=np.intc)  # Best value at node
        self.n_ships_lost = np.zeros((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                      dtype=np.intc)  # Best value at node
        self.n_fleet_strat = np.ones((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                      dtype=np.single) * BAD_VAL  # Best value at node
        self.n_base = np.ones((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                    dtype=np.single) * BAD_VAL  # Best strategic value at node
        self.n_base_disc = np.ones((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                    dtype=np.single) * BAD_VAL  # Best strategic value at node for time-discounted strategies

        self.n_exists = np.zeros((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES), dtype=np.intc)  # Does the node exist?
        self.n_minfleet = np.ones((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                  dtype=np.intc)  # Min fleet required to survive to this point (arriving from best parent, modified by this step)
        self.n_bp_idx = np.zeros((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                 dtype=np.intc)  # Integer index of best parent node
        self.n_dest_terminal = np.zeros((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES), dtype=np.intc)
        self.n_best_direction = np.zeros((P_DEPTH, LOOK + 1, SIZE, SIZE, 2, 2, FACES),
                                         dtype=np.intc)  # Tracking best face to origin nodes

        self.src_terminal = self.mom.terminal.copy()
        self.src_own_fleet = self.mom.own_fleet.copy()
        self.src_enemy_fleet = self.mom.enemy_fleet.copy()
        self.src_enemy_fleet_indices = self.mom.enemy_fleet_indices.copy()
        self.src_enemy_phantom = self.mom.enemy_phantom.copy()
        self.src_kore_val = self.mom.kore_val.copy()
        self.src_base_val = self.mom.base_val.copy()
        self.src_base_disc_val = self.mom.base_disc_val.copy()
        self.src_cargo_val = self.mom.cargo_val.copy()
        #self.src_cargo_destroyed_val = self.mom.cargo_destroyed_val.copy()
        self.src_fleet_strat_val = self.mom.fleet_strat_val.copy()
        self.src_base_zombie_val = self.mom.base_zombie_val.copy()
        self.src_avoid = self.mom.avoid.copy()
        self.src_enemy_bases = self.mom.enemy_bases.copy()

        self.best_paths = []
        for i in range(0, fleet_starting_size + 1):
            self.best_paths.append([i, 0, BAD_VAL, BAD_VAL, 'DOCK', 0, 0, 0])

        # self.init_lookup_matrices()

        # print enemy phantom at their base
        # print('Enemy phantom at base: {}'.format(self.src_enemy_phantom[LOOK, 5, 5]))


    def build(self, fleet_size, my_ship_value, enemy_ship_value, my_kore_val, enemy_kore_val, min_time_constraint, turn_cost, time_level, zombie_start):
        if self.base_create:
            reserve = 3
        else:
            reserve = 0

        max_length = self.max_path_length - reserve
        if time_level >= 2:
            max_length = min(max_length, 7)
        build_graph(self.mom.lookahead, self.time_constraint, min_time_constraint, turn_cost, zombie_start,
                    self.origin.x, self.origin.y, fleet_size, max_length, my_ship_value, enemy_ship_value, my_kore_val,
                    enemy_kore_val, self.src_terminal, self.src_kore_val, self.src_base_val, self.src_base_disc_val,
                    self.src_base_zombie_val,
                    self.src_cargo_val, self.src_fleet_strat_val,
                    self.src_own_fleet, self.src_enemy_fleet,
                    self.src_enemy_fleet_indices,
                    self.src_enemy_phantom, self.src_avoid, self.src_enemy_bases, self.n_kore_gained, self.n_kore_lost,
                    self.n_turn_penalty, self.n_cargo_destroyed, self.n_ships_destroyed, self.n_ships_lost,
                    self.n_fleet_strat, self.n_base, self.n_base_disc,
                    self.n_exists, self.n_minfleet,
                    self.n_bp_idx, self.n_dest_terminal, self.n_best_direction)



    def get_best_path_and_value(self, m_board, dm, player, ships_at_yard, fleet_size, my_ship_val, enemy_ship_val,
                                my_kore_val, enemy_kore_val, time_constraint=500, base_build=False):
        enemy = abs(player-1)
        if base_build:
            convert_ships = 50
        else:
            convert_ships = 0

        if fleet_size >= 2:

            t_max = min(time_constraint, self.n_dest_terminal.shape[1])
            discount_matrix = np.ones((t_max))
            for q in range(0, t_max):
                discount_matrix[q] = PAR['discount_rate'] ** q

            max_p = math.floor(2 * math.log(fleet_size)) + 1
            if base_build:
                max_p -= 3
            valid = self.n_dest_terminal[0:max_p + 1, :t_max, :, :, :, :, 0] == 1
            valid = np.logical_and(valid, self.n_minfleet[0:max_p + 1, :t_max, :, :, :, :, 0] - 1 + convert_ships <= fleet_size)
            #disc_kore = np.divide(self.n_kore_gained[0:max_p + 1, :t_max, :, :, 0], discount_matrix[None, :, None, None])
            kore = self.n_kore_gained[0:max_p + 1, :t_max, :, :, :, :, 0] * my_kore_val
            enemy_kore = (self.n_cargo_destroyed[0:max_p + 1, :t_max, :, :, :, :, 0] - self.n_kore_lost[0:max_p + 1, :t_max, :, :, :, :, 0]) * enemy_kore_val
            base = self.n_base[0:max_p + 1, :t_max, :, :, :, :, 0] + self.n_base_disc[0:max_p + 1, :t_max, :, :, :, :, 0]
            fleet = self.n_fleet_strat[0:max_p + 1, :t_max, :, :, :, :, 0]
            turn_penalty = -1 * self.n_turn_penalty[0:max_p + 1, :t_max, :, :, :, :, 0] * my_kore_val
            ships_lost = -1 * self.n_ships_lost[0:max_p + 1, :t_max, :, :, :, :, 0] * my_ship_val
            ships_destroyed = self.n_ships_destroyed[0:max_p + 1, :t_max, :, :, :, :, 0] * enemy_ship_val

            time_disc = np.arange(0, t_max, dtype=np.single).reshape((1, -1, 1, 1, 1, 1))
            if time_disc.shape[1] > 0:
                time_disc[:, 0, :, :, :, :] = 1.0

            total_val = np.divide(kore + enemy_kore + ships_lost + ships_destroyed + turn_penalty, time_disc) + base + fleet
            total_val_no_disc = kore + enemy_kore + ships_lost + ships_destroyed + turn_penalty + base + fleet

            indices = np.nonzero(valid)
            valid_options = total_val[valid]
            if len(valid_options) > 0:
                best = np.argmax(valid_options)

                # def __init__(self, model: Booster, norms: dict, mom: NeoMomentum, dm: DistanceManager, player: int,
                #              step: int, ship_count, protected_base: Point = None,
                #              added_base: Point = None, lost_base: Point = None, piracy_base: Point = None,
                #              capture_turn: int = -1, combat_turn: int = -1, income_turn=-1, my_added_kore=0,
                #              enemy_added_kore=0,
                #              my_lost_ships=0, enemy_lost_ships=0, added_ships=0, launched_ships=0, path_time=-1):

                # Debug variables
                kore_b = kore[indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best], indices[5][best]]
                enemy_kore_b = enemy_kore[indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best], indices[5][best]]
                base_b = base[indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best], indices[5][best]]
                fleet_b = fleet[indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best], indices[5][best]]
                turn_penalty_b = turn_penalty[indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best], indices[5][best]]
                ships_lost_b = ships_lost[indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best], indices[5][best]]
                ships_destroyed_b = ships_destroyed[indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best], indices[5][best]]

                p = indices[0][best]
                t = indices[1][best]
                x = indices[2][best]
                y = indices[3][best]
                z = indices[4][best]
                k = indices[5][best]

                if m_board.enemy_bases[t, x, y] == 1 and enemy_kore_b == 0.0 and fleet_b == 0.0:
                    same_base = np.zeros((max_p + 1, t_max, 21, 21, 2, 2), dtype=np.bool)
                    same_base[:, :, x, y, :, :] = True
                    valid = np.logical_and(valid, same_base)
                    quickest = np.logical_or.reduce(valid, axis=5)
                    quickest = np.logical_or.reduce(quickest, axis=4)
                    quickest = np.logical_or.reduce(quickest, axis=3)
                    quickest = np.logical_or.reduce(quickest, axis=2)
                    quickest = np.logical_or.reduce(quickest, axis=0)
                    quickest = np.argmax(quickest)
                    quickest_mask = np.zeros((max_p + 1, t_max, 21, 21, 2, 2), dtype=np.bool)
                    quickest_mask[:, quickest, x, y, :, :] = True
                    valid = np.logical_and(valid, quickest_mask)
                    valid_options = total_val[valid]
                    indices = np.nonzero(valid)
                    best = np.argmax(valid_options)

                    # Debug variables
                    kore_b = kore[
                        indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best],
                        indices[5][best]]
                    enemy_kore_b = enemy_kore[
                        indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best],
                        indices[5][best]]
                    base_b = base[
                        indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best],
                        indices[5][best]]
                    fleet_b = fleet[
                        indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best],
                        indices[5][best]]
                    turn_penalty_b = turn_penalty[
                        indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best],
                        indices[5][best]]
                    ships_lost_b = ships_lost[
                        indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best],
                        indices[5][best]]
                    ships_destroyed_b = ships_destroyed[
                        indices[0][best], indices[1][best], indices[2][best], indices[3][best], indices[4][best],
                        indices[5][best]]

                    p = indices[0][best]
                    t = indices[1][best]
                    x = indices[2][best]
                    y = indices[3][best]
                    z = indices[4][best]
                    k = indices[5][best]

                my_added_kore = self.n_kore_gained[p,t,x,y,z,k,0]
                enemy_added_kore = self.n_kore_lost[p,t,x,y,z,k,0] - self.n_cargo_destroyed[p,t,x,y,z,k,0]
                my_lost_ships = self.n_ships_lost[p,t,x,y,z,k,0]
                enemy_lost_ships = self.n_ships_destroyed[p,t,x,y,z,k,0]

                total_val_of_best = total_val_no_disc[indices[0][best], indices[1][best],
                                              indices[2][best], indices[3][best], indices[4][best], indices[5][best]]

                strat_val_of_best = base_b + fleet_b
                path = self.build_path_string(self.max_path_length, indices[0][best], indices[1][best],
                                              indices[2][best], indices[3][best], indices[4][best], indices[5][best], total_val_of_best,
                                              my_ship_val, enemy_ship_val, my_kore_val, enemy_kore_val, base_build, True)
                self.best_paths[fleet_size] = [fleet_size, indices[1][best], total_val_of_best, strat_val_of_best, path, t, x, y]
        return self.best_paths

    def node_value(self, p, t, x, y, z, k, f, my_kore_val, enemy_kore_val, my_ship_val, enemy_ship_val):
        kore = self.n_kore_gained[p, t, x, y, z, k, f] * my_kore_val
        enemy_kore = (self.n_cargo_destroyed[p, t, x, y, z, k, f] - self.n_kore_lost[p, t, x, y, z, k, f]) * enemy_kore_val
        base = self.n_base[p, t, x, y, z, k, f] + self.n_base_disc[p, t, x, y, z, k, f]
        fleet = self.n_fleet_strat[p, t, x, y, z, k, f]
        turn_penalty = -1 * self.n_turn_penalty[p, t, x, y, z, k, f]
        ships_lost = -1 * self.n_ships_lost[p, t, x, y, z, k, f] * my_ship_val
        ships_destroyed = self.n_ships_destroyed[p, t, x, y, z, k, f] * enemy_ship_val
        total_val = kore + enemy_kore + base + fleet + ships_destroyed + turn_penalty + ships_lost
        return total_val

    def build_path_string(self, max_depth, p, t, x, y, z, k, total_val, my_ship_val, enemy_ship_val, my_kore_val,
                          enemy_kore_val, base_build=False, debug_print=False):
        out = ''
        f = 0

        if p == 0 and t <= 1: # Path says stay home
            return 'DOCK'

        idx = np.zeros(1, dtype=np.longlong)

        debug = deque([])
        debug.appendleft('{}-{}-{}-{}-{}-{}-{}: {} [{:.2f}]'.format(p, t, x, y, z, k, 0, 0, total_val))
        while p > 0:
            temp = ''
            idx[0] = self.n_bp_idx[p, t, x, y, z, k, f]

            if idx[0] == -1: # Stay at home option
                return "DOCK"

            p_org = idx[0] // 2500000
            rem = idx[0] % 2500000
            t = rem // 44100
            rem %= 44100
            x_org = rem // 2100
            rem %= 2100
            y_org = rem // 100
            rem %= 100
            z = rem // 50
            rem %= 50
            k = rem // 25
            rem %= 25
            f = rem // 5
            actual_direction = rem % 5

            node_val = self.node_value(p_org, t, x_org, y_org, z, k, f, my_kore_val, enemy_kore_val, my_ship_val, enemy_ship_val)
            debug.appendleft('{}-{}-{}-{}-{}-{}-{}: {} [{:.2f}]'.format(p_org, t, x_org, y_org, z, k, f, actual_direction, node_val))

            if p_org != max_depth:
                if actual_direction == 1:
                    temp += 'N'
                    while y <= y_org:
                        y += SIZE
                elif actual_direction == 2:
                    temp += 'S'
                    while y >= y_org:
                        y -= SIZE
                elif actual_direction == 3:
                    temp += 'E'
                    while x <= x_org:
                        x += SIZE
                elif actual_direction == 4:
                    temp += 'W'
                    while x >= x_org:
                        x -= SIZE
                diff = max(abs(x - x_org), abs(y - y_org)) - 1
                if diff > 0 and (p != max_depth or base_build):
                    temp += str(diff)
                out = temp + out
            p = p_org
            x = x_org
            y = y_org
        if base_build and out != '' and out != 'DOCK':
            out = out + 'C'
        if debug_print:
            print(debug)
        return out