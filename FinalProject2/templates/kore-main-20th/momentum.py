from kaggle_environments.envs.kore_fleets.helpers import *
import numpy as np
import math
from utils import *

NONE = 0
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4
CONSTRUCTION = 10

upgrade_times = [pow(i, 2) + 1 for i in range(1, 10)]
SPAWN_VALUES = []
current = 0
for t in upgrade_times:
    current += t
    SPAWN_VALUES.append(current)

def fix_dim(dim):
    if dim <= -1:
        return 21 + dim
    elif dim >= 21:
        return dim - 21
    else:
        return dim

class NeoMomentum:
    pass



class NeoFleet:
    def __init__(self, ship_count: int, flight_plan: str, x: int, y: int, face: int, kore: int, steps: int):
        self.ship_count = ship_count
        self.flight_plan_str = flight_plan
        self.x = x
        self.y = y
        self.face = face
        self.kore = kore
        self.finished_end = False
        self.finished_start = False
        self.docked_end = False
        self.docked_start = False
        self.flight_plan = [] * steps
        self.steps = steps

    def copy(self):
        copy = NeoFleet(self.ship_count, self.flight_plan_str,
                        self.x, self.y, self.face, self.kore, self.steps)
        copy.finished_start = self.finished_end
        copy.finished_end = copy.finished_start
        copy.docked_start = self.docked_end
        copy.docked_end = copy.docked_start
        for entry in self.flight_plan:
            copy.flight_plan.append(entry)
        return copy


class NeoYard:
    def __init__(self, x: int, y: int, turns_controlled: int):
        self.x = x
        self.y = y
        self.turns_controlled = turns_controlled
        self.captured_start = False
        self.captured_end = False

    def copy(self):
        copy = NeoYard(self.x, self.y, self.turns_controlled)
        copy.captured_start = self.captured_end
        copy.captured_end = copy.captured_start
        return copy

    @property
    def max_spawn(self):
        for idx, target in enumerate(SPAWN_VALUES):
            if self.turns_controlled < target:
                return idx + 1
        return len(SPAWN_VALUES) + 1


class NeoMomentum:
    def __init__(self, board: Board, origin_yard: Shipyard, lookahead: int, dm: DistanceManager):
        self.board = board
        self.lookahead = lookahead
        self.me = board.current_player_id
        self.enemy = abs(self.me-1)
        self.origin_yard = origin_yard
        self.dm = dm
        self.ship_count = origin_yard.ship_count

        self.fleets: List[List[List[NeoFleet]]] = []
        self.yards: List[List[List[NeoYard]]] = []

        self.shipyard_map: List[List[Dict]] = []

        self.kore_bank = [[], []]

        for player in range(0, 2):
            self.fleets.append([])
            self.yards.append([])
            self.shipyard_map.append([])
            for i in range(0, lookahead+1):
                self.fleets[player].append([])
                self.yards[player].append([])
                self.shipyard_map[player].append({})
            self.kore_bank[player].append(self.board.players[player].kore)

        self.spawn_cost = self.board.configuration.spawn_cost
        self.size = self.board.configuration.size

        self.kore = np.zeros(
            (lookahead+1, self.size, self.size), dtype=np.single)

        self.init_kore()
        # for t in range(1, lookahead+1):
        #     self.update_kore(t)
        #     self.regen_kore(t)

        self.get_fleets()
        self.get_shipyards()
        self.plan_trajectories()
        for t in range(1, lookahead+1):
            self.update_loop(t)

    # def capture_time(self, player: int, check_yard: NeoYard):
    #     for idx, yard in zip(range(0, len(self.yards[player][self.lookahead])), self.yards[player][self.lookahead]):
    #         if yard.captured_end and yard.turns_controlled >= self.lookahead - check_yard.turns_controlled and yard.x == check_yard.x and yard.y == check_yard.y:
    #             for t in range(self.lookahead, -1, -1):
    #                 if not self.yards[player][t][idx].captured_start:
    #                     return t
    #     return self.lookahead + 1
    #
    # def built_time(self, player: int, check_yard: NeoYard):
    #     for idx, yard in zip(range(0, len(self.yards[player][self.lookahead])), self.yards[player][self.lookahead]):
    #         if yard.turns_controlled >= self.lookahead - check_yard.turns_controlled and yard.x == check_yard.x and yard.y == check_yard.y:
    #             for t in range(0, self.lookahead+1):
    #                 if idx < len(self.yards[player][t]):
    #                     return t
    #     assert(False) # Shouldn't happen

    def update_loop(self, t):
        self.time_step(t)
        self.increase_shipyard_turns_controlled(t)
        self.produce_ships_if_threatened(t)
        self.make_shipyards(t)
        self.move_fleets(t)
        self.coalesce_fleets(t)
        self.resolve_fleet_collisions(t)
        self.resolve_shipyard_collisions(t)
        self.deposit_kore(t)
        self.orthogonal_damage(t)
        self.mine(t)
        self.regen_kore(t)

    def produce_ships_if_threatened(self, t: int):
        for myself, opponent in zip([self.me, self.enemy], [self.enemy, self.me]):
            threatened = []
            for yard in self.yards[myself][t]:
                found = False
                for fleet in self.fleets[opponent][t]:
                    if not fleet.docked_end and not fleet.finished_end:
                        for plan_step in fleet.flight_plan[t-1:]:
                            if fleet.ship_count >= 50 and plan_step[2] >= CONSTRUCTION:
                                break
                            elif plan_step[0] == yard.x and plan_step[1] == yard.y:
                                    threatened.append(yard)
                                    found = True
                                    break
                            elif Point(plan_step[0], plan_step[1]) in self.shipyard_map[myself][t] or Point(plan_step[0], plan_step[1]) in self.shipyard_map[opponent][t]:
                                break # It will dock in this time step
                    if found:
                        break
            for yard in threatened:
                if t == 1 and yard.x == self.origin_yard.position.x and yard.y == self.origin_yard.position.y:
                    continue # Don't produce in momentum case on turn one
                to_spawn = int(min(yard.max_spawn, self.kore_bank[myself][t] // 10))
                if to_spawn > 0:
                    self.kore_bank[myself][t] -= to_spawn * 10
                    # Is there a fleet already docked here?
                    for docked_fleet in self.fleets[myself][t]:
                        if docked_fleet.docked_end and docked_fleet.x == yard.x and docked_fleet.y == yard.y:
                            docked_fleet.ship_count += to_spawn
                            break
                    else: # Spawn a new fleet
                        fleet = NeoFleet(to_spawn, 'DOCK', yard.x, yard.y, NONE, 0, self.lookahead)
                        fleet.docked_end = True
                        for i in range(0, self.lookahead+1):
                            fleet.flight_plan.append([yard.x, yard.y, NONE])
                        self.fleets[myself][t].append(fleet)

    def incremental_update(self, origin: Point, next_origin: Point, flight_plan: str, ship_count: int, yard_ships: int):
        # Find existing fleet at new origin location and delete it.
        for idx, flt in zip(range(0, len(self.fleets[self.me][0])), self.fleets[self.me][0]):
            if flt.y == next_origin.y and flt.x == next_origin.x:
                break
        if len(self.fleets[self.me][0]) > 0:
            self.delete_fleet(self.me, 0, flt)
            for t in range(0, self.lookahead+1):
                del self.fleets[self.me][t][idx]

        # Launch the new fleet
        fleet = NeoFleet(ship_count, flight_plan, origin.x, origin.y, NONE, 0, self.lookahead)
        self.fleets[self.me][0].append(fleet)

        if yard_ships > ship_count: # Split the fleet if needed
            self.fleets[self.me][0].append(NeoFleet(yard_ships - ship_count, 'DOCK', origin.x, origin.y, NONE, 0, self.lookahead))

        self.plan_trajectory(fleet)

        full_reset = False
        for t in range(1, len(fleet.flight_plan)-1):
            # Make shipyard
            self.fleets[self.me][t].append(fleet.copy())
            fleet = self.fleets[self.me][t][-1]
            to_delete, yard = self.make_shipyard(t, fleet, self.me, [])

            if yard is not None:
                for t_inc in range(t+1, self.lookahead+1):
                    self.yards[self.me][t_inc].append(yard.copy())
                    self.shipyard_map[self.me][t_inc][Point(yard.x, yard.y)] = yard

            for delfleet in to_delete:
                self.delete_fleet(self.me, t, delfleet)
                break
            if fleet.docked_end:
                break

            # Move fleet
            self.move_fleet(fleet, t)

            # See if we have any crashes
            for myflt in self.fleets[self.me][t]:
                if not myflt.finished_end and not myflt.docked_end and myflt.y == fleet.y and myflt.x == fleet.x:
                    full_reset = True
                    break
            for idx, enflt in zip(range(0, len(self.fleets[self.enemy][t])), self.fleets[self.enemy][t]):
                if not enflt.finished_end and not enflt.docked_end and enflt.y == fleet.y and enflt.x == fleet.x:
                    full_reset = True
                    break
                if enflt.y == next_origin.y and enflt.x == next_origin.x and not self.fleets[self.enemy][max(0, t-1)][idx].finished_end:
                    full_reset = True
                    self.fleets[self.enemy][t][idx].finished_end = False
                    self.fleets[self.enemy][t][idx].docked_end = False
                    break
                for direction in Direction.list_directions():
                    curr_pos = Point(fleet.x, fleet.y).translate(
                        direction.to_point(), self.size)
                    if curr_pos.x == enflt.x and curr_pos.y == enflt.y:
                        if not enflt.finished_end and not enflt.docked_end:
                            full_reset = True
                            break

            if full_reset:
                from_t = t
                break
            else:
                result, new_yard = self.resolve_shipyard_collision(fleet, self.me, t)
                if result:
                    for t_inc in range(t+1, self.lookahead+1):
                        self.yards[self.me][t_inc].append(new_yard)
                        self.shipyard_map[self.me][t_inc][Point(fleet.x, fleet.y)] = new_yard
                        # Redundant
                        #enemy_yard = self.shipyard_map[self.enemy][t_inc][Point(fleet.x, fleet.y)]
                        #enemy_yard.captured_end = True
                        #self.delete_shipyard(self.enemy, t_inc, enemy_yard)
                    break
                if self.deposit_kore_single(fleet, self.me, t):
                    break
                self.mine_single(fleet, t)
                if t < self.lookahead:
                    self.regen_kore(t+1)

        if full_reset:
            for player in range(0, 2):
                del self.yards[player][from_t:]
                del self.fleets[player][from_t:]
                del self.kore_bank[player][from_t:]
                del self.shipyard_map[player][from_t:]
                for i in range(from_t, self.lookahead + 1):
                    self.fleets[player].append([])
                    self.yards[player].append([])
                    self.shipyard_map[player].append({})

            self.plan_trajectories()
            for t in range(from_t, self.lookahead+1):
                self.update_loop(t)

        self.reset_source_matrices()

    def reset_source_matrices(self):
        self.phantom = np.zeros(
            (2, self.lookahead+1, self.fleet_count + self.source_count, self.size, self.size), dtype=np.intc)
        self.terminal = np.zeros((self.lookahead+1, self.size, self.size), dtype=np.intc)
        self.fleet_matrix = np.zeros(
            (2, self.lookahead+1, self.fleet_count, self.size, self.size), dtype=np.intc)
        self.base_val = np.zeros((self.lookahead+1, self.size, self.size), dtype=np.single)
        self.base_disc_val = np.zeros((self.lookahead + 1, self.size, self.size), dtype=np.single)
        self.cargo_val = np.zeros((self.lookahead + 1, self.size, self.size), dtype=np.single)
        self.fleet_strat_val = np.zeros((self.lookahead + 1, self.size, self.size), dtype=np.single)
        self.base_zombie_val = np.zeros((self.lookahead + 1, self.size, self.size), dtype=np.single)
        self.enemy_fleet_indices = np.zeros((self.lookahead+1, self.size, self.size), dtype=np.intc)
        self.own_fleet_indices = np.zeros((self.lookahead + 1, self.size, self.size), dtype=np.intc)
        self.avoid = np.zeros((self.lookahead + 1, self.size, self.size), dtype=np.intc)
        self.yard_locations = np.zeros(((self.lookahead + 1), self.size, self.size), dtype=np.intc)
        self.my_bases = np.zeros(((self.lookahead + 1), self.size, self.size), dtype=np.intc)
        self.enemy_bases = np.zeros(((self.lookahead + 1), self.size, self.size), dtype=np.intc)

        self.own_phantom = np.ascontiguousarray(
            np.sum(self.phantom[self.me, :, :, :, :], axis=1), dtype=np.intc)
        self.enemy_phantom = np.ascontiguousarray(
            np.sum(self.phantom[self.enemy, :, :, :, :], axis=1), dtype=np.intc)
        self.own_fleet = np.ascontiguousarray(
            np.sum(self.fleet_matrix[self.me, :, :, :, :], axis=1), dtype=np.intc)
        self.enemy_fleet = np.ascontiguousarray(
            np.sum(self.fleet_matrix[self.enemy, :, :, :, :], axis=1), dtype=np.intc)
        self.bop = np.subtract(np.add(self.own_fleet, self.own_phantom), np.add(self.enemy_fleet, self.enemy_phantom))
        self.true_bop = np.zeros(self.bop.shape, dtype=np.intc)
        self.true_bop_less_ships = np.zeros(self.bop.shape, dtype=np.intc)

    def make_control_matrices(self):
        self.base_control_values = test_control_matrices(self, None)

    def make_source_matrices(self, fleet_size: int):
        if fleet_size > 0:
            mining_rate = min(.99, math.log(fleet_size) / 20)
        else:
            mining_rate = 0
        source_count = max(len(self.yards[0][self.lookahead]), len(self.yards[1][self.lookahead]))
        fleet_count = max(len(self.fleets[0][self.lookahead]), len(self.fleets[1][self.lookahead]))
        self.source_count = source_count
        self.fleet_count = fleet_count
        self.enemy_fleet_count = len(self.fleets[self.enemy][self.lookahead])
        # source_cum_kore_spent = [[[0] * source_count]*(self.lookahead+1)]*2
        # source_cum_production = [[[0] * source_count]*(self.lookahead+1)]*2


        source_cum_kore_spent = [[], []]
        source_cum_production = [[], []]
        for player in range(0, 2):
            for t in range (0, self.lookahead+1):
                source_cum_production[player].append([])
                source_cum_kore_spent[player].append([])
                for s in range(0, source_count):
                    source_cum_production[player][t].append(0)
                    source_cum_kore_spent[player][t].append(0.0)

        self.kore_val = self.kore[:, :, :]

        self.reset_source_matrices()
        self.make_control_matrices()

        for player in range(0, 2):

            phantom_income = 0.0
            avg_kore_per_square = np.sum(self.kore[:, :, :], axis=(1, 2)) / (21 * 21)
            mining_rate_per_ship = avg_kore_per_square * .15 / 21  # Rough assessment of mining rate

            # Start at time t=1 and increment
            for t in range(1, self.lookahead+1):
                parked_ships = 0
                for fleet, f in zip(self.fleets[player][t], range(0, len(self.fleets[player][t]))):
                    if not fleet.finished_end and fleet.ship_count > 0:
                        if fleet.docked_end:
                            parked_ships += fleet.ship_count

                if player == self.enemy and PAR['phantom_income']:
                    phantom_income += parked_ships * mining_rate_per_ship[t]

                # Produce new ships at the source, if possible
                for s in range(0, len(self.yards[player][t])):
                    yard = self.yards[player][t][s]
                    # Mark as a terminal location
                    self.terminal[t, yard.x, yard.y] = 1
                    # Mark the location of the yard
                    self.yard_locations[t, yard.x, yard.y] = 1
                    if player == self.me and not yard.captured_start:
                        self.my_bases[t, yard.x, yard.y] = 1
                        if t == 1:
                            self.my_bases[0, yard.x, yard.y] = 1
                    if player == self.enemy and yard.turns_controlled > 0 and not yard.captured_start:
                        self.enemy_bases[t, yard.x, yard.y] = 1
                        if t == 1:
                            self.enemy_bases[0, yard.x, yard.y] = 1

                    if t == 1 and yard.x == self.origin_yard.position.x and yard.y == self.origin_yard.position.y:
                        dont_produce = True # Don't produce in same yard, since we'll probably launch
                    else:
                        dont_produce = False

                    if not yard.captured_start and yard.turns_controlled > 0:
                        # Build as much as possible.
                        new_production = min(yard.max_spawn, max(
                            0, (self.kore_bank[player][t] + phantom_income - source_cum_kore_spent[player][t - 1][
                                s])) // self.spawn_cost)
                        if dont_produce:
                            new_production = 0
                        source_cum_production[player][t][s] = source_cum_production[player][t-1][s] + new_production
                        source_cum_kore_spent[player][t][s] = source_cum_kore_spent[player][t -
                                                                                            1][s] + new_production * self.spawn_cost
                    else:
                        source_cum_production[player][t][s] = source_cum_production[player][t - 1][s]
                        source_cum_kore_spent[player][t][s] = source_cum_kore_spent[player][t - 1][s]

        # Map surviving fleets to destination sources
        fleet_destination_map = [[], []]
        for player in range(0, 2):
            for fidx, fleet in zip(range(0, len(self.fleets[player][self.lookahead])), self.fleets[player][self.lookahead]):
                fleet_destination_map[player].append(-1)
                if fidx >= len(self.fleets[player][0]): # Avoid double-counting pseudo-fleets at bases
                    continue
                for t_it, plan_step in zip(range(1, min(self.lookahead, len(fleet.flight_plan))), fleet.flight_plan[:min(self.lookahead, len(fleet.flight_plan))]):
                    if fidx < len(self.fleets[player][t_it]):
                        future_fleet = self.fleets[player][t_it][fidx]
                        if not future_fleet.finished_start:
                            for yidx, yard in zip(range(0, len(self.yards[player][t_it])), self.yards[player][t_it]):
                                if not yard.captured_end and yard.x == plan_step[0] and yard.y == plan_step[1]:
                                    fleet_destination_map[player][len(fleet_destination_map[player])-1] = yidx

        # Set up phantom and fleet matrices in t=0
        for player in range(0, 2):
            new = self.phantom[player, 0, :, :, :]
            for f, fleet in zip(range(0, len(self.fleets[player][0])), self.fleets[player][0]):
                if fleet.docked_start and f < len(self.fleets[player][0]):
                    new[source_count + f, fleet.x, fleet.y] = fleet.ship_count
                else:
                    self.fleet_matrix[player][0][f][fleet.x][fleet.y] = fleet.ship_count
                if player == self.enemy:
                    self.enemy_fleet_indices[0, fleet.x, fleet.y] = f + 1
                else:
                    self.own_fleet_indices[0, fleet.x, fleet.y] = f + 1

        for player in range(0, 2):
            for t in range(1, self.lookahead+1):

                # Copy prior time step
                prior = self.phantom[player, t-1, :, :, :]  # Slices, not copies
                new = self.phantom[player, t, :, :, :]

                # Max of new and prior at same spot
                new[:, :, :] = np.maximum(new[:, :, :], prior[:, :, :])

                new[:, 1:, :] = np.maximum(new[:, 1:, :], prior[:, :-1, :])  # from left side
                new[:, 0, :] = np.maximum(new[:, 0, :], prior[:, -1, :])

                # From right side
                new[:, :-1, :] = np.maximum(new[:, :-1, :], prior[:, 1:, :])
                new[:, -1, :] = np.maximum(new[:, -1, :], prior[:, 0, :])

                new[:, :, 1:] = np.maximum(new[:, :, 1:], prior[:, :, :-1])  # from bottom up
                new[:, :, 0] = np.maximum(new[:, :, 0], prior[:, :, -1])

                # From top down
                new[:, :, :-1] = np.maximum(new[:, :, :-1], prior[:, :, 1:])
                new[:, :, -1] = np.maximum(new[:, :, -1], prior[:, :, 0])

                # Add cumulative production to the origin point of the yard
                if player == self.enemy:
                    for yard, idx in zip(self.yards[player][t], range(0, len(self.yards[player][t]))):
                        new[idx, yard.x, yard.y] = source_cum_production[player][t][idx]
                else: # Treat identically, but no phantom production
                    for yard, idx in zip(self.yards[player][t], range(0, len(self.yards[player][t]))):
                        new[idx, yard.x, yard.y] = source_cum_production[player][t][idx]

                # Indicate present status of the fleets
                for fleet, f in zip(self.fleets[player][t], range(0, len(self.fleets[player][t]))):
                    if f >= len(self.fleets[player][0]):
                        continue
                    if fleet.docked_end and not fleet.docked_start:
                        # new[source_count+f, fleet.x, fleet.y] = fleet.ship_count # Docked ships convert to phantom
                        # # Find how many ships were in this fleet last turn
                        if fleet.flight_plan_str == 'DOCK':
                            prior_ships = fleet.ship_count
                        else:
                            prior_ships = self.fleets[player][t-1][f].ship_count
                            # If we just captured a yard, then our ship count is has already netted out defenders
                            # We should use the prior turn's ship count instead, so that they cancel out
                        new[source_count + f, fleet.x, fleet.y] = prior_ships
                    elif not fleet.docked_end:
                        self.fleet_matrix[player][t][f][fleet.x][fleet.y] = fleet.ship_count # Non-docked are marked as flyers
                        if player == self.enemy:
                            self.enemy_fleet_indices[t, fleet.x, fleet.y] = f+1
                            self.cargo_val[t, fleet.x, fleet.y] += fleet.kore
                            # if self.yard_locations[t, fleet.x, fleet.y] != 1: # Don't look at adjacencies on arrival turn
                            #     for offset in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                            #         x = fleet.x + offset[0]
                            #         y = fleet.y + offset[1]
                            #         x = fix_dim(x)
                            #         y = fix_dim(y)
                                    # if self.yard_locations[t, x, y] != 1: # Don't get credit at bases!
                                    #     #self.strat_val[t, x, y] += fleet.ship_count * 10
                                    #     self.cargo_val[t, x, y] += fleet.kore * 0.4
                                    #     self.cargo_destroyed_val[t, x, y] += fleet.kore * 0.4
                        elif player == self.me:
                            self.own_fleet_indices[t, fleet.x, fleet.y] = f + 1
                            if self.fleets[player][t][f].flight_plan_str != 'DOCK' and \
                                'C' in self.fleets[player][t][f].flight_plan_str: # Avoid merging with base builders, creates zombies
                                self.avoid[t, fleet.x, fleet.y] = 1

        # Aggregate into enemy and own phantom matrices
        self.own_phantom_v1 = np.ascontiguousarray(
            np.sum(self.phantom[self.me, :, :, :, :], axis=1), dtype=np.intc)
        self.enemy_phantom_v1 = np.ascontiguousarray(
            np.sum(self.phantom[self.enemy, :, :, :, :], axis=1), dtype=np.intc)

        # Create single source phantom matrices
        self.ss_phantom = np.zeros((2, self.lookahead+1, source_count, self.size, self.size), dtype=np.intc)
        for player in [self.me, self.enemy]:
            self.ss_phantom[player, :, :, :, :] = self.phantom[player, :, :source_count, :, :]
            for fleet, fidx in zip(self.fleets[player][self.lookahead], range(0, len(self.fleets[player][self.lookahead]))):
                source = fleet_destination_map[player][fidx]
                if source >= 0:
                    self.ss_phantom[player, :, source, :, :] += self.phantom[player, :, source_count + fidx, :, :]

        if len(self.fleets[self.me][self.lookahead]) > 1 and len(self.fleets[self.enemy][self.lookahead]) > 1:
            # Use heuristics to assume some coordination loss in phantom fleets
            # Phantom for most squares considered to be highest single source only
            en_source_raw = np.swapaxes(self.ss_phantom[self.enemy, :, :, :, :], 0, 1)

            # Make error correction to phantom for time shift
            # en_source = np.zeros(en_source_raw.shape, dtype=np.intc)
            # en_source[:, :-1, :, :] = en_source_raw[:, 1:, :, :]
            # en_source[:, -1, :, :] = en_source_raw[:, -1, :, :]
            en_source = en_source_raw

            own_source = np.swapaxes(self.ss_phantom[self.me, :, :, :, :], 0, 1)
            self.enemy_phantom = np.ascontiguousarray(np.sort(np.partition(en_source, kth=-1, axis=0)[-1, :, :, :], axis=0).astype(dtype=np.intc))
            self.own_phantom = np.ascontiguousarray(
                np.sort(np.partition(own_source, kth=-1, axis=0)[-1, :, :, :], axis=0).astype(dtype=np.intc))

            # Enemy bases expected to have coordinated defense, so may have up to 2 sources on standby
            if en_source.shape[0] > 1:
                part = 2
            else:
                part = 1
            two = np.sum(np.sort(np.partition(en_source, kth=-part, axis=0)[-part:,:,:,:], axis=0).astype(dtype=np.single), axis=0).astype(dtype=np.intc)
            for t in range(0, self.lookahead+1):
                for yard in self.yards[self.enemy][t]:
                    if not yard.captured_start:
                        self.enemy_phantom[t, yard.x, yard.y] = two[t, yard.x, yard.y] # Copy five source into base defense

            # I can count on 2 source defense of my own bases, I think. Enemy likely to have double source attack.
            if own_source.shape[0] > 1:
                part = 2
            else:
                part = 1
            two_me = np.sum(np.sort(np.partition(own_source, kth=-part, axis=0)[-part:,:,:,:], axis=0).astype(dtype=np.single), axis=0).astype(dtype=np.intc)
            #two = np.sum(np.sort(np.partition(en_source, kth=-2, axis=0)[-5:,:,:,:], axis=0).astype(dtype=np.single), axis=0).astype(dtype=np.intc)
            for t in range(0, self.lookahead+1):
                for yard in self.yards[self.me][t]:
                    if not yard.captured_start:
                        self.own_phantom[t, yard.x, yard.y] = two_me[t, yard.x, yard.y] # Copy two source into base defense
                        self.enemy_phantom[t, yard.x, yard.y] = two[t, yard.x, yard.y]  # Assume attacks are less coordinated


        else:
            self.own_phantom = self.own_phantom_v1
            self.enemy_phantom = self.enemy_phantom_v1


        self.own_fleet = np.ascontiguousarray(
            np.sum(self.fleet_matrix[self.me, :, :, :, :], axis=1), dtype=np.intc)
        self.enemy_fleet = np.ascontiguousarray(
            np.sum(self.fleet_matrix[self.enemy, :, :, :, :], axis=1), dtype=np.intc)
        self.bop = np.subtract(np.add(self.own_fleet, self.own_phantom), np.add(self.enemy_fleet, self.enemy_phantom)) + self.ship_count
        self.true_bop = np.subtract(np.add(self.own_fleet, self.own_phantom_v1), np.add(self.enemy_fleet, self.enemy_phantom_v1)) + self.ship_count
        self.true_bop_less_ships = np.subtract(np.add(self.own_fleet, self.own_phantom_v1), np.add(self.enemy_fleet, self.enemy_phantom_v1))
        self.quasi_bop_less_ships = np.subtract(np.add(self.own_fleet, self.own_phantom), np.add(self.enemy_fleet, self.enemy_phantom))


        #self.make_distance_matrix()

    def make_distance_matrix(self):
        self.distance = {}
        pointset = []
        for player in range(0, 2):
            for yard in self.yards[player]:
                pointset.append(Point(yard.x, yard.y))
        for i in range(0, len(pointset)):
            for j in range(i, len(pointset)):
                distance = self.dm.calc_distance(pointset[i].x, pointset[i].y, pointset[j].x, pointset[j].y)
                self.distance[str(pointset[i].x)+'-'+str(pointset[i].y)+'-'+str(pointset[j].x)+'-'+str(pointset[j].y)] = distance


    def capture_turn(self, position: Point):
        for yard, idx in zip(self.yards[self.me][self.lookahead], range(0, len(self.yards[self.me][self.lookahead]))):
            if yard.x == position.x and yard.y == position.y:
                break
        else:
            assert(False)
        for t in range(0, self.lookahead+1):
            if self.yards[self.me][t][idx].captured_end:
                return t
        return self.lookahead+1


    def time_step(self, t):
        for player in range(0, 2):
            for fleet in self.fleets[player][t-1]:
                self.fleets[player][t].append(fleet.copy())
            for yard in self.yards[player][t-1]:
                yard_copy = yard.copy()
                self.yards[player][t].append(yard_copy)
                if not yard.captured_end:
                    self.shipyard_map[player][t][Point(yard.x, yard.y)] = yard_copy
            self.kore_bank[player].append(self.kore_bank[player][t-1])

    def mine_single(self, fleet, t):
        if fleet.ship_count > 0 and not fleet.finished_end:
            mining_rate = min(.99, math.log(fleet.ship_count) / 20)
            delta_kore = round(
                self.kore[t-1, fleet.x, fleet.y] * mining_rate, 3)
            if delta_kore > 0:
                fleet.kore += delta_kore
                self.kore[t-1, fleet.x, fleet.y] -= delta_kore

    def mine(self, t):
        for player in range(0, 2):
            for fleet in self.fleets[player][t]:
                self.mine_single(fleet, t)

    def orthogonal_damage(self, t):
        incoming_fleet_dmg = [DefaultDict(lambda: DefaultDict(
            int)), DefaultDict(lambda: DefaultDict(int))]
        for player in range(0, 2):
            for fleet in self.fleets[player][t]:
                if not fleet.finished_end and not fleet.docked_end:
                    for direction in Direction.list_directions():
                        curr_pos = Point(fleet.x, fleet.y).translate(
                            direction.to_point(), self.size)
                        for enemy_fleet in self.fleets[abs(player-1)][t]:
                            if curr_pos.x == enemy_fleet.x and curr_pos.y == enemy_fleet.y:
                                if not enemy_fleet.finished_end and not enemy_fleet.docked_end:
                                    incoming_fleet_dmg[abs(
                                        player-1)][Point(enemy_fleet.x, enemy_fleet.y)][Point(fleet.x, fleet.y)] = fleet.ship_count

        to_distribute = [DefaultDict(lambda: DefaultDict(
            int)), DefaultDict(lambda: DefaultDict(int))]
        for player in range(0, 2):
            for fleet_loc, fleet_dmg_dict in incoming_fleet_dmg[player].items():
                for fleet in self.fleets[player][t]:
                    if fleet_loc.x == fleet.x and fleet_loc.y == fleet.y:
                        break
                damage = sum(fleet_dmg_dict.values())
                if damage >= fleet.ship_count:
                    self.kore[t-1, fleet.x, fleet.y] += fleet.kore / 2
                    to_split = fleet.kore / 2
                    for f_loc, dmg in fleet_dmg_dict.items():
                        to_distribute[abs(player-1)][f_loc][f_loc.to_index(  # Put in other player's to_distribute pile
                            self.size)] = to_split * dmg/damage
                    self.delete_fleet(player, t, fleet)
                else:
                    fleet.ship_count -= damage

        # give kore claimed above to surviving fleets, otherwise add it to the kore of the tile where the fleet died
        for player in range(0, 2):
            # to_distribute.items()
            for fleet_loc, loc_kore_dict in to_distribute[player].items():
                for fleet in self.fleets[player][t]:
                    if fleet_loc.x == fleet.x and fleet_loc.y == fleet.y and not fleet.finished_end:
                        break
                else:  # Is dead
                    for loc_idx, kore in loc_kore_dict.items():
                        location = Point.from_index(loc_idx, self.size)
                        self.kore[t, location.x, location.y] += kore
                    return
                # If still alive
                fleet.kore += sum(loc_kore_dict.values())


    def update_yard_maps(self, t):
        for player in range(0, 2):
            for yard in self.yards[player][t]:
                if not yard.captured_end:
                    self.yard_map[player][t][Point(yard.x, yard.y)] = yard

    def delete_fleet(self, player, t, fleet):
        fleet.kore = 0
        fleet.ship_count = 0
        fleet.finished_end = True

    def dock_fleet(self, fleet):
        fleet.kore = 0
        fleet.docked_end = True

    def delete_shipyard(self, player, t, yard):
        for i in range(len(self.yards[player][t])-1, -1, -1):
            yard_org = self.yards[player][t][i]
            if yard.x == yard_org.x and yard.y == yard_org.y:
                yard_org.captured_end = True
                break
        self.shipyard_map[player][t].pop(Point(yard.x, yard.y))

    def deposit_kore_single(self, fleet, player, t):
        if Point(fleet.x, fleet.y) in self.shipyard_map[player][t] and not fleet.docked_end:
            self.kore_bank[player][t] += fleet.kore
            self.dock_fleet(fleet)
            return True

    def deposit_kore(self, t):
        for player in range(0, 2):
            for fleet in self.fleets[player][t]:
                self.deposit_kore_single(fleet, player, t)


    def resolve_shipyard_collision(self, fleet, player, t):
        enemy = abs(player - 1)
        location = Point(fleet.x, fleet.y)
        if not fleet.finished_end and not fleet.docked_end:
            if location in self.shipyard_map[enemy][t]:
                yard = self.shipyard_map[enemy][t][location]
                # Will collide with fleets at base first
                # if fleet.ship_count > yard.ship_count:
                new_yard = NeoYard(
                    yard.x, yard.y, 0)
                self.kore_bank[player][t] += fleet.kore
                self.yards[player][t].append(new_yard)
                self.shipyard_map[player][t][location] = new_yard
                self.dock_fleet(fleet)
                yard.captured_end = True
                self.delete_shipyard(enemy, t, yard)
                return True, new_yard
        return False, None

    def resolve_shipyard_collisions(self, t):
        for player in range(0, 2):
            for fleet in self.fleets[player][t]:
                self.resolve_shipyard_collision(fleet, player, t)

    def resolve_fleet_collisions(self, t):
        for fleet in self.fleets[0][t]:
            if not fleet.finished_end:
                for other_fleet in self.fleets[1][t]:
                    if other_fleet.x == fleet.x and other_fleet.y == fleet.y and not other_fleet.finished_end:
                        if fleet.ship_count == other_fleet.ship_count:  # No winner
                            if fleet.docked_end:
                                self.kore_bank[0][t] += other_fleet.kore
                            else:
                                self.kore[t-1, other_fleet.x, other_fleet.y] += other_fleet.kore
                            if other_fleet.docked_end:
                                self.kore_bank[1][t] += fleet.kore
                            else:
                                self.kore[t, fleet.x, fleet.y] += fleet.kore
                            self.delete_fleet(0, t, fleet)
                            self.delete_fleet(1, t, other_fleet)
                            break
                        elif fleet.ship_count > other_fleet.ship_count:
                            fleet.ship_count -= other_fleet.ship_count
                            if fleet.docked_end:
                                self.kore_bank[0][t] += other_fleet.kore
                            else:
                                fleet.kore += other_fleet.kore
                            self.delete_fleet(1, t, other_fleet)
                        else:
                            other_fleet.ship_count -= fleet.ship_count
                            if other_fleet.docked_end:
                                self.kore_bank[1][t] += fleet.kore
                            else:
                                other_fleet.kore += fleet.kore
                            self.delete_fleet(0, t, fleet)

    def coalesce_fleets(self, t):
        for player in range(0, 2):
            collision_sets = []
            already_collided = []
            for i in range(0, len(self.fleets[player][t])):
                if not self.fleets[player][t][i].finished_end and not self.fleets[player][t][i].docked_end and i not in already_collided: # Dock collisions allowed
                    collision = False
                    collision_set = [i]
                    j = i + 1
                    while j < len(self.fleets[player][t]):
                        if not self.fleets[player][t][j].finished_end and not self.fleets[player][t][j].docked_end:
                            if self.fleets[player][t][j].x == self.fleets[player][t][i].x and self.fleets[player][t][j].y == self.fleets[player][t][i].y:
                                collision = True
                                already_collided.append(j)
                                collision_set.append(j)
                        j += 1
                    if collision:
                        collision_sets.append(collision_set)
            for collision_set in collision_sets:
                winner = self.collide_priority(player, t, collision_set)
                for q in range(0, len(collision_set)):
                    if collision_set[q] != winner:
                        self.fleets[player][t][winner].ship_count += self.fleets[player][t][collision_set[q]].ship_count
                        self.fleets[player][t][winner].kore += self.fleets[player][t][collision_set[q]].kore
                        self.delete_fleet(
                            player, t, self.fleets[player][t][collision_set[q]])

    def collide_priority(self, player, t, collision_set):
        winner = collision_set[0]
        for i in range(1, len(collision_set)):
            if self.fleets[player][t][winner].ship_count < self.fleets[player][t][collision_set[i]].ship_count or (self.fleets[player][t][winner].ship_count == self.fleets[player][t][collision_set[i]].ship_count and self.fleets[player][t][winner].kore < self.fleets[player][t][collision_set[i]].kore) or (self.fleets[player][t][winner].ship_count == self.fleets[player][t][collision_set[i]].ship_count and self.fleets[player][t][winner].kore == self.fleets[player][t][collision_set[i]].kore and self.fleets[player][t][collision_set[i]].face == NORTH) or (self.fleets[player][t][winner].ship_count == self.fleets[player][t][collision_set[i]].ship_count and self.fleets[player][t][winner].kore == self.fleets[player][t][collision_set[i]].kore and self.fleets[player][t][collision_set[i]].face == EAST and self.fleets[player][t][winner].face in [SOUTH, WEST]) or (self.fleets[player][t][winner].ship_count == self.fleets[player][t][collision_set[i]].ship_count and self.fleets[player][t][winner].kore == self.fleets[player][t][collision_set[i]].kore and self.fleets[player][t][collision_set[i]].face == SOUTH and self.fleets[player][t][winner].face == WEST):
                winner = collision_set[i]
        return winner

    def move_fleet(self, fleet, t):
        if not fleet.docked_end and not fleet.finished_end:
            fleet.x = fleet.flight_plan[t-1][0]
            fleet.y = fleet.flight_plan[t-1][1]

    def move_fleets(self, t):
        for player in range(0, 2):
            for fleet in self.fleets[player][t]:
                self.move_fleet(fleet, t)
                    #fleet.flight_plan = fleet.flight_plan[1:]

    def make_shipyard(self, t, fleet, player, to_delete):
        new_yard = None
        if fleet.flight_plan[t-1][2] > CONSTRUCTION:  # Want to build a shipyard
            if fleet.ship_count > self.board.configuration.convert_cost:  # Have enough ships?
                fleet.ship_count -= self.board.configuration.convert_cost
                if fleet.ship_count == 0:
                    to_delete.append(fleet)
                self.kore_bank[player][t] += fleet.kore
                self.dock_fleet(fleet)
                new_yard = NeoYard(
                    fleet.flight_plan[t-1][0], fleet.flight_plan[t-1][1], 0)
                self.yards[player][t].append(new_yard)
            else:
                # Skip if failed
                # fleet.flight_plan = fleet.flight_plan[1:]
                del fleet.flight_plan[t-1]
        return to_delete, new_yard

    def make_shipyards(self, t):
        to_delete = []
        for player in range(0, 2):
            for fleet in self.fleets[player][t]:
                to_delete, yard = self.make_shipyard(t, fleet, player, to_delete)
            for fleet in to_delete:
                self.delete_fleet(player, t, fleet)

    def increase_shipyard_turns_controlled(self, t):
        for player in range(0, 2):
            for yard in self.yards[player][t]:
                if not yard.captured_end:
                    yard.turns_controlled += 1

    def init_kore(self):
        for cell in self.board.cells.values():
            self.kore[0, cell.position.x, cell.position.y] = cell.kore

    def update_kore(self, t):
        self.kore[t, :, :] = self.kore[t-1, :, :]

    def regen_kore(self, t: int):  # vectorized kore regeneration
        if t < self.lookahead:
            self.kore[t, :, :][self.kore[t-1, :, :] < self.board.configuration.max_cell_kore] = (
                self.kore[t-1, :, :][self.kore[t-1, :, :] < self.board.configuration.max_cell_kore] * (1 + self.board.configuration.regen_rate))
            self.kore[t, :, :][self.kore[t-1, :, :] >= self.board.configuration.max_cell_kore] = self.kore[t-1, :, :][self.kore[t-1, :, :] >= self.board.configuration.max_cell_kore]
        # BUG: should not regen if fleets on top if it.

    def plan_trajectory(self, fleet):
        fleet.flight_plan = []
        if not fleet.finished_end:
            x = fleet.x
            y = fleet.y
            running_plan = fleet.flight_plan_str
            count = self.lookahead
            orig_face = fleet.face
            while count >= 0:
                if running_plan == 'DOCK':
                    next_loc = [x, y, fleet.face]
                elif len(running_plan) == 0 or running_plan[0].isdigit():
                    if fleet.face == NORTH:
                        y += 1
                        y = fix_dim(y)
                    elif fleet.face == SOUTH:
                        y -= 1
                        y = fix_dim(y)
                    elif fleet.face == EAST:
                        x += 1
                        x = fix_dim(x)
                    elif fleet.face == WEST:
                        x -= 1
                        x = fix_dim(x)
                    next_loc = [x, y, fleet.face]
                    step_len = 0
                    if len(running_plan) > 0 and running_plan[0].isdigit():
                        for i in range(0, len(running_plan)):
                            if running_plan[i].isdigit():
                                step_len += 1
                            else:
                                break
                        steps_remaining = int(running_plan[0:step_len]) - 1

                        if steps_remaining > 0:
                            running_plan = str(
                                steps_remaining) + running_plan[step_len:]
                        else:
                            running_plan = running_plan[step_len:]
                elif running_plan[0] == 'C':
                    next_loc = [x, y, fleet.face + CONSTRUCTION]
                    count += 1
                    running_plan = running_plan[1:]
                else:
                    if running_plan[0] == 'N':
                        y += 1
                        y = fix_dim(y)
                        next_loc = [x, y, NORTH]
                        fleet.face = NORTH
                    if running_plan[0] == 'S':
                        y -= 1
                        y = fix_dim(y)
                        next_loc = [x, y, SOUTH]
                        fleet.face = SOUTH
                    if running_plan[0] == 'E':
                        x += 1
                        x = fix_dim(x)
                        next_loc = [x, y, EAST]
                        fleet.face = EAST
                    if running_plan[0] == 'W':
                        x -= 1
                        x = fix_dim(x)
                        next_loc = [x, y, WEST]
                        fleet.face = WEST
                    running_plan = running_plan[1:]
                for axis in [0, 1]:
                    if next_loc[axis] < 0:
                        next_loc[axis] += self.board.configuration.size
                    elif next_loc[axis] == self.board.configuration.size:
                        next_loc[axis] = 0
                fleet.flight_plan.append(next_loc)
                count -= 1
            fleet.face = orig_face # Reset facing to original

    def plan_trajectories(self):
        for player in range(0, 2):
            for fleet in self.fleets[player][0]:
                self.plan_trajectory(fleet)

    def get_fleets(self):
        for fleet in self.board.fleets.values():
            if fleet.player_id == self.me:
                self.initialize_fleet(fleet, self.me)
            else:
                self.initialize_fleet(fleet, self.enemy)

    def initialize_fleet(self, fleet: Fleet, player: int):
        if fleet.direction == Direction.NORTH:
            face = NORTH
        elif fleet.direction == Direction.SOUTH:
            face = SOUTH
        elif fleet.direction == Direction.EAST:
            face = EAST
        elif fleet.direction == Direction.WEST:
            face = WEST
        else:
            face = NONE
        fleet = NeoFleet(fleet.ship_count, fleet.flight_plan,
                                      fleet.position.x, fleet.position.y, face, fleet.kore, self.lookahead)
        self.fleets[player][0].append(fleet)
        return fleet

    def get_shipyards(self):
        for yard in self.board.shipyards.values():
            if yard.player_id == self.me:
                self.initialize_yard(yard, self.me)
            else:
                self.initialize_yard(yard, self.enemy)
            #if yard.ship_count > 0:


    def initialize_yard(self, yard: Shipyard, player: int):
        neo_yard = NeoYard(yard.position.x, yard.position.y,
                           yard._turns_controlled)
        if Point(neo_yard.x, neo_yard.y) != self.origin_yard.position: # Special case for the origin shipyard, so we don't double-count fleets we're about to launch
            next_action = yard.next_action
            if next_action is not None and next_action.action_type == ShipyardActionType.SPAWN:
                #neo_yard.ship_count += next_action.num_ships
                # self.initialize_fleet(
                #     Fleet('x', next_action.num_ships + yard.ship_count, Direction.NORTH, Point(yard.position.x, yard.position.y), 0.0, 'DOCK',
                #           player, self.board), player)
                self.initialize_fleet(
                    Fleet('x', yard.ship_count, Direction.NORTH, Point(yard.position.x, yard.position.y), 0.0, 'DOCK',
                          player, self.board), player)
                # self.kore_bank[player][0] -= next_action.num_ships * \
                #     self.spawn_cost
                # assert(self.kore_bank[player][0] >= 0)
                #pass
            elif next_action is not None and next_action.action_type == ShipyardActionType.LAUNCH:
                #neo_yard.ship_count -= next_action.num_ships
                #assert(neo_yard.ship_count >= 0)
                fleet_at_yard = None
                for fleet in self.fleets[player][0]:
                    if yard.position.x == fleet.x and yard.position.y == fleet.y:
                        fleet_at_yard = fleet
                        break
                if fleet_at_yard:
                    self.fleets[player][0].append(NeoFleet(next_action.num_ships, next_action.flight_plan, yard.position.x, yard.position.y, NONE, 0, self.lookahead+1))
                    if next_action.num_ships < fleet_at_yard.ship_count:
                        docked_fleet = NeoFleet(fleet_at_yard.ship_count - next_action.num_ships, 'DOCK', yard.position.x, yard.position.y, NONE, 0,
                                     self.lookahead + 1)
                        docked_fleet.docked_start = True
                        docked_fleet.docked_end = True
                        self.fleets[player][0].append(docked_fleet)
            else:
                docked_fleet = NeoFleet(yard.ship_count, 'DOCK', yard.position.x,
                                        yard.position.y, NONE, 0,
                                        self.lookahead + 1)
                docked_fleet.docked_start = True
                docked_fleet.docked_end = True
                self.fleets[player][0].append(docked_fleet)

        self.yards[player][0].append(NeoYard(
            yard.position.x, yard.position.y, yard._turns_controlled))

def test_control_matrices(mom: NeoMomentum, new_pt = None):
    base_control_values = {}
    for player in range(0, 2):
        for yard in mom.yards[player][mom.lookahead]:
            base_control_values[Point(yard.x, yard.y)] = 0.0
    if new_pt is not None:
        base_control_values[new_pt] = 0.0
    for x in range(0, 21):
        for y in range(0, 21):
            if mom.kore[0, x, y] > 0.0 and Point(x, y) not in base_control_values:
                weights = []
                for base in base_control_values:
                    dist = mom.dm.calc_distance(base.x, base.y, x, y)
                    weights.append(1/dist)
                weights = list(map(lambda wt: wt / sum(weights), weights))
                for base, idx in zip(base_control_values, range(0, len(base_control_values))):
                    base_control_values[base] += weights[idx] * mom.kore[0, x, y]
    return base_control_values
