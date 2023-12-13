from kaggle_environments.envs.kore_fleets.helpers import *
from kaggle_environments.envs.kore_fleets.kore_fleets import balanced_agent
from momentum import NeoMomentum, NeoYard, fix_dim, test_control_matrices
from valuation import Valuation, ramp
import time
import numpy as np
from utils import *
from graph import Graph
from os.path import exists
from labeler import cy_label_strategy


def enable_kill_switch(me):
    with open('kill_' + str(me) + '.txt', 'w') as f:
        pass

def locate_yard(me: Player, yard: NeoYard):
    for shipyard in me.shipyards:
        if shipyard.position.x == yard.x and shipyard.position.y == yard.y:
            return shipyard

def produce(me, yard, kore_left):
    orig_kore = kore_left
    ship_count = min(ramp[yard.turns_controlled], int(kore_left / 10))
    kore_left -= 10 * ship_count
    action = ShipyardAction.spawn_ships(ship_count)
    shipyard = locate_yard(me, yard)
    if shipyard is None: # Hasn't been built yet
        return orig_kore
    if shipyard.next_action is not None and shipyard.next_action.action_type == ShipyardActionType.LAUNCH: # Don't overwrite launches
        return orig_kore
    else:
        shipyard.next_action = action
        return kore_left


def check_kill_switch(mom: NeoMomentum, valuation: Valuation, ship_count: int):
    if exists('kill_' + str(mom.me) + '.txt') and mom.board.observation['remainingOverageTime'] > 30:
        print("KILL")
        return True
    # if valuation.network_capacity[mom.enemy][mom.lookahead] * 20 + valuation.ships[mom.enemy] < (valuation.total_ships + ship_count) * 0.3333 and mom.board.observation['remainingOverageTime'] > 30:
    #     enable_kill_switch(mom.me)
    #     print("KILL")
    #     return True
    else:
        return False


def compute_base_distances(mom: NeoMomentum):
    pointset = []
    dm = mom.dm
    for player in range(0, 2):
        for yard in mom.yards[player][mom.lookahead]:
            pointset.append(Point(yard.x, yard.y))
    distances = {}
    for point in pointset:
        distances[point] = {}
        for inner in pointset:
            distances[point][inner] = dm.calc_distance(point.x, point.y, inner.x, inner.y)
    return distances

def in_bounds(x, y, boundary):
    return x >= boundary[0][0] and x <= boundary[1][0] and y >= boundary[0][1] and y <= boundary[1][0]

def quadrant_strategy(mom: NeoMomentum, valuation: Valuation, ship_count: int):
    me = mom.me
    enemy = mom.enemy
    quadrant_boundaries = []
    quadrant_boundaries.append([[[5, 5], [15, 15]]]) # King (121 squares)
    if mom.me == 0:
        quadrant_boundaries.append([[[0, 15], [5, 20]], [[0, 0], [5, 5]], [[15, 15], [20, 20]], [[17, 0], [20, 3]]]) # Queen (124 squares)
    else:
        quadrant_boundaries.append([[[0, 17], [3, 20]], [[0, 0], [5, 5]], [[15, 15], [20, 20]], [[15, 0], [15, 5]]])  # Queen (124 squares)
    if mom.me == 0:
        quadrant_boundaries.append([[[0, 5], [5, 15]], [[16, 5], [20, 15]]]) # Jack (121 squares)
    else:
        quadrant_boundaries.append([[[0, 5], [4, 15]], [[15, 5], [20, 15]]])  # Jack (121 squares)
    if mom.me == 0:
        quadrant_boundaries.append([[[5, 15], [15, 20]], [[5, 0], [15, 4]]])  # Joker (121 squares)
    else:
        quadrant_boundaries.append([[[5, 16], [15, 20]], [[5, 0], [15, 5]]])  # Joker (121 squares)

    no_strategy = [[[0, 0], [20, 20]]]

    quadrant_sizes = [121, 124, 121, 121]
    quadrant_kore = []
    quadrant_seeds = []
    my_base_count = []
    enemy_base_count = []
    seed_density = []
    avg_kore = []
    score = []

    # Don't need a quadrant strategy if we are winning handily
    if valuation.ships[mom.me] + ship_count - 200 > valuation.total_ships * 0.5:
        print("Default strategy")
        return NONE, no_strategy

    for idx, quadrant, size in zip(range(0, len(quadrant_boundaries)), quadrant_boundaries, quadrant_sizes):
        kore = 0
        seeds = 0
        base_count_me = 0
        base_count_other = 0
        for boundary in quadrant:
            for yard in mom.yards[me][mom.lookahead]:
                if not yard.captured_end:
                    if in_bounds(yard.x, yard.y, boundary):
                        base_count_me += 1
            for yard in mom.yards[enemy][mom.lookahead]:
                if not yard.captured_end:
                    if in_bounds(yard.x, yard.y, boundary):
                        base_count_other += 1
            for x in range(boundary[0][0], boundary[1][0]+1):
                for y in range(boundary[0][1], boundary[1][1]+1):
                    if mom.kore[0, x, y] > 0:
                        seeds += 1
                        kore += mom.kore[0, x, y]
        my_base_count.append(base_count_me)
        enemy_base_count.append(base_count_other)
        quadrant_kore.append(kore)
        quadrant_seeds.append(seeds)
        seed_density.append(seeds / size)
        avg_kore.append(kore / size)
        score.append(avg_kore[idx] * seed_density[idx])

    best_score = 0
    best_idx = 0
    for idx in range(0, len(score)):
        if score[idx] > best_score:
            best_score = score[idx]
            best_idx = idx

    if my_base_count[best_idx] > enemy_base_count[best_idx] + 1: # We already dominate the area, revert to base strategy
        return NONE, no_strategy
    else:
        print(list(["King", "Queen", "Jack", "Joker"])[best_idx])
        return list([KING, QUEEN, JACK, JOKER])[best_idx], quadrant_boundaries[best_idx]


def optimal_new_base_v6(mom: NeoMomentum, valuation: Valuation, ship_count, dm: DistanceManager):
    optimal_point = None
    optimal_value = -1000.0
    bop_look = min(mom.lookahead, 20)

    strategy, boundaries = quadrant_strategy(mom, valuation, ship_count)

    # lower_range = 6
    # if strategy == KING:
    lower_range = 4

    # Seek out particular distance
    for target_distance in range(6, lower_range-1, -1):
        if optimal_point is not None:
            break
        for boundary in boundaries:
            for y in range(boundary[0][1], boundary[1][1]+1):
                for x in range(boundary[0][0], boundary[1][0]+1):
                    if Point(x, y) not in mom.shipyard_map[0][mom.lookahead] and Point(x, y) not in mom.shipyard_map[1][mom.lookahead] and mom.kore[0, x, y] <= 50.0:
                        skip = True
                        for yard in mom.yards[mom.me][0]: # Check if we're close enough to other yards
                            if not yard.captured_end:
                                distance = dm.calc_distance(x, y, yard.x, yard.y)
                                if distance < target_distance:
                                    skip = True
                                    break
                                if distance > target_distance:
                                    continue
                                if distance == target_distance:
                                    skip = False
                        if skip:
                            continue # Skip this loop if not at target distance

                        # for yard in mom.yards[mom.me][mom.lookahead]: # ... and not too close to other yards in the future
                        #     if not yard.captured_end:
                        #         #distance = int(dm.calc_distance(x, y, yard.x, yard.y))
                        #         if dm.calc_distance(x, y, yard.x, yard.y) < target_distance:
                        #             skip = True
                        #             break
                        if skip:
                            continue # Skip this loop because we're already building a yard in this vicinity

                        # Don't build too close to enemy (including future enemy locations, or locations which will be captured)
                        for yard in mom.yards[mom.enemy][mom.lookahead]:
                            distance = dm.calc_distance(x, y, yard.x, yard.y)
                            if not yard.captured_end and distance <= target_distance * 1.5:
                                skip = True
                                break
                        if skip:
                            continue

                        bop = mom.own_phantom[bop_look, x, y] + ship_count - 50 - np.max(mom.enemy_phantom[:bop_look, x, y] + mom.enemy_fleet[:bop_look, x, y])
                        if bop < 0:
                            continue

                        kore = 0.0
                        for offset_distance in range(1, 20):
                            offset_entries = []
                            for x_off in range(0, offset_distance+1):
                                offset_entries.append([x_off, offset_distance-x_off])
                                offset_entries.append([x_off, x_off-offset_distance])
                                offset_entries.append([-x_off, offset_distance-x_off])
                                offset_entries.append([x_off, x_off - offset_distance])
                            for offset in offset_entries:
                                x_off = fix_dim(x + offset[0])
                                y_off = fix_dim(y + offset[1])
                                closest_base_dist = 20
                                for yard in mom.yards[mom.me][mom.lookahead]:
                                    distance = dm.calc_distance(x_off, y_off, yard.x, yard.y)
                                    if distance < closest_base_dist:
                                        closest_base_dist = distance

                                if distance > offset_distance: # Only value kore marginally closer now
                                    #weight = 1 / (math.log(1 + offset_distance)**0.5)
                                    weight = min(1.0, 1 / offset_distance)
                                else:
                                    weight = 0.0
                                #weight = min(1.0, 1 / offset_distance)
                                if closest_base_dist < offset_distance:
                                    weight *= (closest_base_dist / offset_distance)

                                if y_off == yard.x or y_off == yard.y:
                                    weight *= 2

                                if mom.kore[0, x_off, y_off] == 0.0:
                                    kore -= 25 * weight
                                else:
                                    kore += mom.kore[0, x_off, y_off] * weight


                        if kore >= optimal_value:
                            optimal_value = kore
                            optimal_point = Point(x, y)
    return optimal_point

def label_enemy_base_values(me: int, m_board: NeoMomentum, valuation: Valuation, fleet_size: int, origin_point: Point, dm: DistanceManager):
    enemy = abs(1-me)
    free_base_value = max(PAR['min_base_value'], valuation.calc_capacity_delta_npv(
        me, fleet_size, free=True, delay=False)) #, bias=PAR['base_build_bias']
    values = []
    for idx, yard in zip(range(0, len(m_board.yards[enemy][m_board.lookahead])),
                         m_board.yards[enemy][m_board.lookahead]):

        distance = dm.calc_distance(origin_point.x, origin_point.y, yard.x, yard.y)

        for t in range(0, m_board.lookahead+1):
            if idx < len(m_board.yards[enemy][t]):
                first_t = t
                break

        turns_controlled = m_board.yards[enemy][first_t][idx].turns_controlled
        value = free_base_value + max(PAR['min_base_value'], valuation.calc_capacity_delta_npv(enemy, 0, additional=True, free=True,
                                                                    delay=False, first_turn=first_t, turns_controlled=turns_controlled))
                                                                    #bias=PAR['base_build_bias'] * valuation.ships[me] / max(1, valuation.total_ships)))

        bcv = m_board.base_control_values[Point(yard.x, yard.y)]
        value += bcv

        values.append(value)
        captured = False
        first_t = -1
        for t in range(0, m_board.lookahead+1):  # Look forward in time
            #if idx >= len(m_board.yards[abs(1 - me)][t]):  # Yard not built yet
                # continue # skip
            #    break  # ONLY CAPTURE YARDS THAT ALREADY EXIST - DON'T RECAPTURE SINCE IT CAUSES ME TO TRY RECAPTURING
            if first_t < 0 and idx < len(m_board.yards[abs(1 - me)][t]):
                first_t = t  # Label first time when the board is present
                for my_yd in m_board.yards[me][t]: # Was this a capture?
                    if my_yd.x == yard.x and my_yd.y == yard.y and not my_yd.captured_start:
                        first_t = min(m_board.lookahead, t+1)
            if idx < len(m_board.yards[abs(1 - me)][t]) and m_board.yards[abs(1 - me)][t][idx].captured_end:
                captured = True  # Board already captured
                break
        if not captured:  # No strategic value if the board will already be captured
            m_board.base_disc_val[first_t:, yard.x, yard.y] = m_board.base_disc_val[first_t:, yard.x, yard.y] + value
            #m_board.strat_val[first_t:, yard.x, yard.y] = m_board.strat_val[first_t:, yard.x, yard.y] + value * PAR['capture_bias']
    return values


def label_home_base_values(me: int, m_board: NeoMomentum, valuation: Valuation, ship_count: int):
    enemy = abs(1-me)
    free_base_value = max(PAR['min_base_value'], valuation.calc_capacity_delta_npv(enemy, 0, additional=True, free=True, delay=False)) #, bias=PAR['base_build_bias']))  # Value of my bases to the enemy

    values = []
    for idx, yard in zip(range(0, len(m_board.yards[me][m_board.lookahead])),
                         m_board.yards[me][m_board.lookahead]):
                                                                    #bias=PAR['base_build_bias'] * valuation.ships[me] / max(1, valuation.total_ships)))
        value = 0
        captured_t = m_board.lookahead+1
        first_t = -1
        blame = []
        inbound_ships = []
        crash_value = 0
        for t in range(0, m_board.lookahead):  # Look forward in time
            if idx >= len(m_board.yards[me][t]):  # Yard not built yet
                continue  # so skip
            elif first_t < 0:
                first_t = t  # Label first time when the board is present
            if m_board.yards[me][t][idx].captured_end:
                captured_t = m_board.lookahead+1
                for t_check in range(m_board.lookahead, -1, -1):
                    if m_board.yards[me][t_check][idx].captured_end and not m_board.yards[me][t_check][idx].captured_start:
                        captured_t = t_check
                        break
                turns_controlled_at_capture = m_board.yards[me][captured_t][idx].turns_controlled

                value = free_base_value + valuation.calc_capacity_delta_npv(me, ship_count, free=True, delay=False, first_turn=first_t, turns_controlled=turns_controlled_at_capture,
                                                          captured=True, capture_turn=captured_t)

                # Find the son of a gun who did this!
                for fidx, fleet in zip(range(0, len(m_board.fleets[enemy][t])), m_board.fleets[enemy][t]):
                    if fleet.x == yard.x and fleet.y == yard.y:
                        if captured_t==0:
                            blame.append(fidx)
                            inbound_ships.append(fleet.ship_count)
                        if not fleet.docked_start and t == captured_t: # Was flying on prior turn
                            blame.append(fidx)
                            inbound_ships.append(m_board.fleets[enemy][captured_t-1][fidx].ship_count)
                            crash_value += m_board.fleets[enemy][captured_t-1][fidx].kore # Bonus kore for causing enemy to crash into defended base
                        elif not fleet.docked_start and t < captured_t: # Was flying on prior turn
                            blame.append(fidx)
                            inbound_ships.append(m_board.fleets[enemy][captured_t-1][fidx].ship_count)
                            # No incremental crash value since we already win this fight
                        elif not fleet.docked_start and t > captured_t:
                            blame.append(fidx)
                            inbound_ships.append(fleet.ship_count)

                total_invaders = sum(inbound_ships)


        blame = list(set(blame)) # Get unique values for blame

        if len(blame) == 1:
            fidx = blame[0]
            resp_fleet = m_board.fleets[enemy][0][fidx]
            for path_step, t in zip(resp_fleet.flight_plan, range(1, len(resp_fleet.flight_plan)+1)):
                m_board.fleet_strat_val[t, path_step[0], path_step[1]] += value
                if t == captured_t:
                    m_board.fleet_strat_val[t-1, path_step[0], path_step[1]] += value # t-1 at base ok, due to cadence
                    break
                # for offset in [[0, 1], [1, 0], [0, -1], [-1, 0]]: # Duplicated in cygraph
                #     x_off = fix_dim(path_step[0] + offset[0])
                #     y_off = fix_dim(path_step[1] + offset[1])
                #     if not m_board.yard_locations[t, x_off, y_off]:
                #         m_board.fleet_strat_val[t, x_off, y_off] += value
        if captured_t < m_board.lookahead+1:
            if np.count_nonzero(m_board.true_bop[first_t:captured_t + 1, yard.x, yard.y] < 0) == 0: # Can I make a difference?
                m_board.base_val[first_t:captured_t + 1, yard.x, yard.y] += value
                m_board.enemy_phantom[first_t:captured_t + 1, yard.x, yard.y] = 0 # No phantoms at base if I can help
        values.append(value)
    return values


def create_base(mom: NeoMomentum, location: Point, npv_base: float):

    # Make it so we never want to go through a terminal node to build a base
    mom.base_val[mom.terminal > 0] = -9999999999.0
    # Zero out original terminals so that we can specify our target destination
    mom.terminal = np.zeros(mom.terminal.shape, dtype=np.intc)

    # for i in range(1, 30):
    #     np.savetxt("enemy_phantom" + str(i) + ".csv", np.flipud(np.fliplr(np.transpose(np.flipud(mom.enemy_phantom[i,:,:])))), delimiter=",")
    # mom.enemy_phantom = np.zeros(mom.enemy_phantom.shape)
    for t in range(0, mom.terminal.shape[0]):
        mom.terminal[t, location.x, location.y] = 1
        mom.base_val[t, location.x, location.y] += npv_base * PAR['base_build_bias'] / PAR['discount_rate']**t

# def dump_strategy(mom: NeoMomentum):
#     data = np.zeros((len(mom.yards[0][mom.lookahead])+len(mom.yards[1][mom.lookahead]), mom.lookahead+3), dtype=np.single)
#     bop = np.zeros((len(mom.yards[0][mom.lookahead]) + len(mom.yards[1][mom.lookahead]), mom.lookahead + 3),
#                     dtype=np.single)
#     idx = 0
#     for yard in mom.yards[mom.me][mom.lookahead]:
#         data[idx, 0] = yard.x
#         data[idx, 1] = yard.y
#         data[idx, 2:] = mom.strat_val[:, yard.x, yard.y] + mom.strat_disc_val[:, yard.x, yard.y]
#         bop[idx, 0] = yard.x
#         bop[idx, 1] = yard.y
#         bop[idx, 2:] = mom.bop[:, yard.x, yard.y]
#         idx += 1
#     for yard in mom.yards[mom.enemy][mom.lookahead]:
#         data[idx, 0] = yard.x
#         data[idx, 1] = yard.y
#         data[idx, 2:] = mom.strat_val[:, yard.x, yard.y] + mom.strat_disc_val[:, yard.x, yard.y]
#         bop[idx, 0] = yard.x
#         bop[idx, 1] = yard.y
#         bop[idx, 2:] = mom.bop[:, yard.x, yard.y]
#         idx += 1
#     np.savetxt("strategy.csv", data, delimiter=",")
#     np.savetxt("bop.csv", bop, delimiter=",")

def agent(obs, config):
    board = Board(obs, config)
    m_board = None

    me = board.current_player
    enemy = abs(1-me.id)
    spawn_cost = board.configuration.spawn_cost
    kore_left = me.kore
    valuation = None
    halt_production = False

    print('\n')
    t_graph = 0

    # Should order shipyards by ship count, then production
    queue = PriorityQueue()
    for shipyard in me.shipyards:
        queue.insert(shipyard, shipyard.ship_count*100 + shipyard.max_spawn)

    # if board.step <= 150:
    preferred_size = 21
    minimum_size = 8
    # else:
    #     preferred_size = 50
    #     minimum_size = 21

    if board.step <= 20:
        maximum_size = 21
    else:
        maximum_size = None

    # Check how much time we have remaining, and throttle back activity as required
    TIME_LEVEL = 0
    TIME_ALLOTMENT = 90000000.0

    IGNORE_THROTTLE = False


    if not IGNORE_THROTTLE:
        overage_remaining = board.observation['remainingOverageTime']
        print("Over {:.2f}".format(overage_remaining))
        if overage_remaining <= 40:
            TIME_LEVEL = 1
            PAR['max_lookahead'] = 25
            TIME_ALLOTMENT = 6.0
            print("**TIME REMINDER**")
        else:
            TIME_ALLOTMENT = 12.0
        if overage_remaining <= 30:
            TIME_LEVEL = 2
            PAR['max_lookahead'] = 20
            TIME_ALLOTMENT = 2.8
            print("**TIME WARNING**")
        if overage_remaining <= 15:
            TIME_LEVEL = 3
            PAR['max_lookahead'] = 15
            TIME_ALLOTMENT = 2.0
            print("**TIME CRITICAL**")
        if overage_remaining <= 2: # Hard abort - might still win it
            print("**TIME ABORT**")
            return me.next_actions


    dm = DistanceManager()

    turn_time_start = time.time()

    # Spawn control ##########################

    t_momentum = 0
    spawn_queue = PriorityQueue()
    if len(board.current_player.shipyards) > 0:
        if m_board is None:  # Create a momentum if we don't have one yet
            m_board = NeoMomentum(board, board.current_player.shipyards[0], min(LOOK, 400 - board.step - 1), dm)
            m_board.make_source_matrices(5)
            t_momentum = time.time() - turn_time_start
        if valuation is None and m_board is not None:  # Ditto for valuation
            valuation = Valuation(m_board, me.id, board.step)

        # Determine new base npv
        npv_base = valuation.calc_capacity_delta_npv(me.id, 0)

        # Critical level priority
        captured = False
        for idx, yard in zip(range(0, len(m_board.yards[me.id][m_board.lookahead])),
                             m_board.yards[me.id][m_board.lookahead]):
            if idx >= len(m_board.yards[me.id][0]):
                continue

            if yard.captured_end:  # Ignore non-captured yards
                captured = True

            # Check if we are auto-producing in momentum, which means we're under threat
            for fidx, fleet in zip(range(0, len(m_board.fleets[me.id][1])),
                             m_board.fleets[me.id][1]):
                if fleet.x == yard.x and fleet.y == yard.y and \
                        (idx >= len(m_board.fleets[me.id][0])
                        or m_board.fleets[me.id][0][fidx].ship_count < m_board.fleets[me.id][1][fidx].ship_count):
                    captured = True

            if not captured:
                continue

            # Find capture time
            # Small bug here if momentum is auto-producing
            t_captured = m_board.lookahead
            t_first = 0
            for t in range(m_board.lookahead, -1, -1):
                if idx < len(m_board.yards[me.id][t]) and m_board.yards[me.id][t][idx].captured_end:
                    t_captured = t
                # if idx >= len(m_board.yards[me.id][t]):
                #     t_first = t + 1
                #     break

            # Determine total volume of inbound ships
            need = 0
            last_threat = 0
            for fidx, fleet in zip(range(0, len(m_board.fleets[enemy][0])), m_board.fleets[enemy][0]):
                for step, details in zip(range(0, len(fleet.flight_plan)), fleet.flight_plan):
                    if step < len(m_board.fleets[enemy]):
                        if fidx < len(m_board.fleets[enemy][step]):
                            if details[0] == yard.x and details[1] == yard.y and not \
                                    m_board.fleets[enemy][step][fidx].finished_start and not \
                                    m_board.fleets[enemy][step][fidx].docked_start:
                                last_threat = max(last_threat, step + 1)
                                need += fleet.ship_count

            if need == 0:
                continue

            turns_needed = 0
            turns_controlled = m_board.yards[me.id][0][idx].turns_controlled
            for turn in range(1, len(ramp[yard.turns_controlled:])):
                if np.sum(ramp[turns_controlled:turns_controlled + turn]) >= need:
                    turns_needed = turn
                    break
            priority = 100000
            if t_first + turns_needed <= t_captured:
                priority += 1 / need
            spawn_queue.insert(m_board.yards[me.id][t_first][idx], priority, min(last_threat, turns_needed))

            # Assign high priority to any neighbors who might be able to help
            for nidx, neighbor in zip(range(0, len(m_board.yards[me.id][t_captured])),
                                      m_board.yards[me.id][t_captured]):
                captured = False
                if m_board.yards[me.id][m_board.lookahead][nidx].captured_end:  # This is under threat too, will get critical
                    continue

                # Check if we are auto-producing in momentum, which means we're under threat
                for fidx, fleet in zip(range(0, len(m_board.fleets[me.id][1])),
                                      m_board.fleets[me.id][1]):
                    if fleet.x == m_board.yards[me.id][m_board.lookahead][nidx].x and \
                            fleet.y == m_board.yards[me.id][m_board.lookahead][nidx].y and \
                            (fidx >= len(m_board.fleets[me.id][0])
                             or m_board.fleets[me.id][0][fidx].ship_count < m_board.fleets[me.id][1][fidx].ship_count):
                        captured = True

                if captured:
                    continue

                if neighbor in spawn_queue.queue:  # Don't duplicate
                    continue
                dist = dm.calc_distance(yard.x, yard.y, neighbor.x, neighbor.y)
                if dist < last_threat - 1:  # Could support
                    turns_needed = 0
                    t_first = 0
                    for t in range(0, m_board.lookahead + 1):
                        if nidx < len(m_board.yards[me.id][t]):
                            t_first = t
                            break
                    turns_controlled = m_board.yards[me.id][t_first][nidx].turns_controlled
                    for turn in range(1, len(ramp[turns_controlled:])):
                        if np.sum(ramp[turns_controlled:(turns_controlled + turn)]) >= need:
                            turns_needed = turn
                            break
                    priority = 10000
                    if t_first + turns_needed < last_threat - dist:
                        priority += 1 / need
                    spawn_queue.insert(m_board.yards[me.id][t_first][nidx], priority,
                                       min(turns_needed, last_threat - dist - t_first - 1))

        # Don't build bases when we have high priority spawning going on. Implies we are under attack.
        if npv_base > 0 and len(spawn_queue.queue) > 0:
            npv_base = 0.0
            print("Don't build base while under attack")

        # Don't build bases when we have more shipyards and fewer ships
        my_yards = 0
        my_plans = 0
        en_yards = 0
        en_plans = 0
        for idx, yard in zip(range(0, len(m_board.yards[me.id][m_board.lookahead])), m_board.yards[me.id][m_board.lookahead]):
            if not yard.captured_end:
                my_yards += 1
            if idx >= len(m_board.yards[me.id][0]):
                my_plans += 1
        for idx, yard in zip(range(0, len(m_board.yards[enemy][m_board.lookahead])), m_board.yards[enemy][m_board.lookahead]):
            if not yard.captured_end:
                en_yards += 1
            if idx >= len(m_board.yards[enemy][0]):
                en_plans += 1
        if npv_base > 0 and my_yards > en_yards and \
            valuation.ships[me.id] - 50 * my_plans < valuation.ships[enemy] - 50 * en_plans:
            npv_base = 0.0
            print("Dangerous to build base")


        # Normal priority to all others, based on short-term BOP
        for yard in m_board.yards[me.id][0]:
            if yard not in spawn_queue.queue:
                spawn_queue.insert(yard, -m_board.bop[min(15, m_board.lookahead), yard.x, yard.y],
                                   1)  # Normal priority

        cum_prod = 0
        kore_left = me.kore
        while not spawn_queue.isEmpty():
            yard, value, turns = spawn_queue.pop()
            turns = min(turns, len(m_board.kore_bank[me.id]))
            if value >= 10000:  # Critical / high priority. Serve these to the max
                if kore_left > spawn_cost:
                    kore_left = produce(me, yard, kore_left)
                    cum_prod += np.sum(ramp[yard.turns_controlled:yard.turns_controlled + turns])
                    if cum_prod * spawn_cost > m_board.kore_bank[me.id][min(turns, m_board.lookahead)]:  # If we will exhaust our income on this yard, skip all others
                        halt_production = True
                        break
                else:
                    spawn_queue.insert(yard, value, 0)  # Add back and check launches
                    break
            else:
                # kore_left = produce(me, yard, kore_left)
                spawn_queue.insert(yard, value, 0)  # Add back and check launches
                break

    # Double check to make sure we aren't spawning at very end
    if valuation is not None and valuation.ship_npv[0] <= 0.0 and board.step > 350 and valuation.ships[
        me.id] > valuation.total_ships * 0.5 or board.step > 395:
        for shipyard in me.shipyards:
            if shipyard.next_action is not None and shipyard.next_action.action_type == ShipyardActionType.SPAWN:
                shipyard.next_action = None
                print("Cancel spawn")


    # Launch control ##########################

    # if m_board is not None:
    #     start = time.time()
    #     old_pos = shipyard.position
    #     old_count = shipyard.ship_count
    #     shipyard = queue.pop()
    #     m_board.incremental_update(old_pos, shipyard.position, None, None, None)
    #     end = time.time()
    #     t_momentum = end - start

    #for shipyard in me.shipyards:
    kore_left_temp = kore_left
    m_board = None
    largest_launch_yard = 0
    shipyard = None
    if not queue.isEmpty():
        shipyard = queue.pop()

    while shipyard is not None:

        if shipyard.next_action is not None:
            if not queue.isEmpty():
                shipyard = queue.pop()
            else:
                shipyard = None
            continue # We are already spawning here, skip

        best_path = None

        # Check if we have sufficient time to act
        time_now = time.time()
        if time_now - turn_time_start > TIME_ALLOTMENT:
            print("Time elapsed: {:.2f}".format(time_now - turn_time_start))
            break

        action = None
        top_path = None
        top_score = -999999.00

        print("Yard " + str(shipyard.position.x) + "_" + str(shipyard.position.y))

        max_spawn = shipyard.max_spawn
        max_spawn = min(max_spawn, int(kore_left / spawn_cost))

        # Spawn or launch or none
        # if shipyard.ship_count < max_spawn * 7 and kore_left >= spawn_cost * max_spawn:
        if board.step <= 24 and board.players[abs(me.id-1)].shipyards[0].ship_count >= shipyard.ship_count \
                and kore_left_temp > spawn_cost and not halt_production:
            decision = 'spawn'
            kore_left_temp -= shipyard.max_spawn * 10
            kore_left_temp = max(0, kore_left_temp)
            if not queue.isEmpty():
                shipyard = queue.pop()
            else:
                shipyard = None
            continue # Don't launch
            #initial_spawn_queue.insert(shipyard, shipyard.ship_count)
        elif shipyard.ship_count < 21 and kore_left_temp > spawn_cost and (valuation is None or valuation.ship_npv[0] > 0 or valuation.ships[me.id] < valuation.total_ships * .5) and not halt_production:
            decision = 'spawn'
            kore_left_temp -= shipyard.max_spawn * 10
            kore_left_temp = max(0, kore_left_temp)
            if not queue.isEmpty():
                shipyard = queue.pop()
            else:
                shipyard = None
            continue # Don't launch
            #initial_spawn_queue.insert(shipyard, shipyard.ship_count)
        else:
            decision = 'launch'

        if shipyard.ship_count > 0 and decision == 'launch':

            if shipyard.ship_count > largest_launch_yard:
                largest_launch_yard = shipyard.ship_count

            best_paths = None

            if m_board is None:
                start = time.time()
                m_board = NeoMomentum(
                    board, shipyard, min(LOOK, 400 - board.step - 1), dm)
                end = time.time()
                t_momentum = end - start
            start = time.time()
            m_board.make_source_matrices(shipyard.ship_count)
            end = time.time()
            t_source_matrices = end - start
            start = time.time()
            valuation = Valuation(m_board, me.id, board.step)


            # m_board.kore_val[m_board.terminal == 0] = m_board.kore_val[m_board.terminal == 0] - valuation.avg_kore_per_square

            # Risk adjustment for enemy phantom ships - take more risks when power advantaged
            if PAR['risk_adjustment']:
                if valuation.ships[me.id] > valuation.total_ships * 0.5:
                    m_board.enemy_phantom = np.divide(m_board.enemy_phantom, np.maximum(0.001, (valuation.ships[me.id] / np.maximum(1, valuation.ships[enemy])))).astype(dtype=np.intc)

            end = time.time()
            t_valuation = end - start

            kill_switch = False
            if not kill_switch:
                start = time.time()
                # Label strategic value of home bases
                base_vals = label_home_base_values(me.id, m_board, valuation, shipyard.ship_count)

                # Label strategic values of enemy bases
                base_vals.extend(label_enemy_base_values(me.id, m_board, valuation, shipyard.ship_count, shipyard.position, dm))

                # m_board.print_BOP(len(m_board.boards)-1)

                #label_strategy(m_board, shipyard.ship_count, base_vals)
                distances = compute_base_distances(m_board)
                ahead = min(m_board.lookahead, len(m_board.yards[0]))
                yard_ct = len(m_board.yards[me.id][ahead]) + len(m_board.yards[enemy][ahead])
                my_yard_ct = len(m_board.yards[me.id][ahead])
                t_captured = np.zeros((yard_ct), dtype=np.intc)
                t_first = np.zeros((yard_ct), dtype=np.intc)
                x = np.zeros((yard_ct), dtype=np.intc)
                y = np.zeros((yard_ct), dtype=np.intc)
                for player, yards, first, end, offset in [(me.id, m_board.yards[me.id], 0, my_yard_ct, 0),
                                                  (enemy, m_board.yards[enemy], 0, yard_ct-my_yard_ct, my_yard_ct)]:
                    for idx in range(first, end):
                        x[idx+offset] = yards[ahead][idx].x
                        y[idx+offset] = yards[ahead][idx].y
                        if player == me.id and shipyard.position.x == x[idx+offset] and shipyard.position.y == y[idx+offset]:
                            origin_idx = idx+offset
                        for t in range(0, ahead):
                            if idx < len(yards[t]):
                                t_first[idx+offset] = t
                                break
                        t_captured[idx+offset] = ahead + 1
                        for t in range(ahead, t_first[idx+offset]-1, -1):
                            if idx < len(yards[t]) and yards[t][idx].captured_end:
                                t_captured[idx+offset] = t
                cy_label_strategy(m_board.bop, ahead, shipyard.ship_count, np.ascontiguousarray(base_vals, dtype=np.single), distances,
                                  my_yard_ct, t_captured, t_first, x, y, m_board.base_val, m_board.base_disc_val, origin_idx)

                #dump_strategy(m_board)

                end = time.time()
                t_label = end - start
            else:
                t_label = 0

            # Build the graph
            start = time.time()
            choice_set = []
            choice_type = []
            zombie_sizes = None

            ZOMBIE = 1
            NOZOMBIE = 0

            if TIME_LEVEL < 2:
                # if shipyard.ship_count >= minimum_size:
                choice_set.append(minimum_size)
                choice_type.append(NOZOMBIE)
                # if shipyard.ship_count > preferred_size:
                choice_set.append(preferred_size)
                choice_type.append(NOZOMBIE)
                if shipyard.ship_count // 2 > preferred_size:
                    choice_set.append(shipyard.ship_count // 2)
                    choice_type.append(NOZOMBIE)
                if shipyard.ship_count <= 130:
                    choice_set.append(13)
                    choice_type.append(NOZOMBIE)
                if shipyard.ship_count <= 63:
                    choice_set.append(5)
                    choice_type.append(NOZOMBIE)
                    choice_set.append(3)
                    choice_type.append(NOZOMBIE)
                if shipyard.ship_count <= 42:
                    choice_set.append(2)
                    choice_type.append(NOZOMBIE)
            if maximum_size is None or shipyard.ship_count <= maximum_size or TIME_LEVEL >= 2:
                choice_set.append(shipyard.ship_count)
                choice_type.append(NOZOMBIE)

            if TIME_LEVEL < 2:
                if zombie_sizes is not None and len(zombie_sizes) > 0:
                    zombie_sizes = list(set(zombie_sizes))
                    zombie_sizes.sort()
                    if zombie_sizes is not None:
                        for z_size in zombie_sizes:
                            choice_set.append(z_size)
                            choice_type.append(ZOMBIE)
                    if shipyard.ship_count not in choice_set:
                        choice_set.append(shipyard.ship_count)
                        choice_type.append(ZOMBIE)
                else:
                    choice_set.append(shipyard.ship_count)
                    choice_type.append(ZOMBIE)
            else:
                if zombie_sizes is not None and len(zombie_sizes) > 0:
                    zombie_sizes.sort(reverse=True)
                    choice_set.append(zombie_sizes[0])
                    choice_type.append(ZOMBIE)

            prior_best_paths = []
            for i in range(0, shipyard.ship_count + 1):
                prior_best_paths.append([i, 0, BAD_VAL, BAD_VAL, 'DOCK', 0, 0, 0])

            for size, zombie_choice in zip(choice_set, choice_type):

                if zombie_choice == ZOMBIE:
                    continue


                g = Graph(board.current_player_id, m_board,
                          shipyard.position, size, base_create=False, kill_switch=False)  # Build base set of paths without base creation

                # Define minimum efficient path for a given size
                if size == shipyard.ship_count or zombie_choice == 1:
                    min_efficient_time = 0
                else:
                    min_efficient_time = shipyard.ship_count // size

                # Define minimum time to avoid production bottlenecking
                if m_board.kore_bank[me.id][0] < spawn_cost or valuation.ship_npv[0] <= 0.0 or zombie_choice == 1 or size == shipyard.ship_count:
                    min_bottlneck_time = 0
                else:
                    for t in range(1, len(m_board.kore_bank[me.id])):
                        capacity = sum(valuation.network_capacity[me.id][:min(t+1, len(valuation.network_capacity[me.id])-1)])
                        capacity -= shipyard.max_spawn
                        if capacity * 10 >= m_board.kore_bank[me.id][t]: # Time when our kore will be exhausted
                            min_bottlneck_time = t
                            break
                    else:
                        min_bottlneck_time = m_board.lookahead + 1
                    if size == shipyard.ship_count:
                        min_bottlneck_time = 0 # Don't force super-long paths, in case we are truly bottlenecked

                mining_rate = min(.99, math.log(size) / 20)
                if PAR['turn_penalty']:
                    turn_penalty = np.sum(m_board.kore_val[0,:,:]) / 441 * mining_rate * 0.25
                else:
                    turn_penalty = 0.0



                # if size == 50:
                g.build(size, 10.0, 10.0, 1.0, 1.0,
                        max(min_bottlneck_time, min_efficient_time), turn_penalty, TIME_LEVEL, zombie_start=zombie_choice)
                # else:
                #     continue




                # Create a time constraint if we're too small!
                if shipyard.ship_count < minimum_size and zombie_choice == NOZOMBIE:
                    time_constraint = 500
                    for t in range(0, len(m_board.fleets[me.id])):
                        for fleet in m_board.fleets[me.id][t]:
                            if fleet.docked_end and not fleet.docked_start:
                                time_constraint = t
                else:
                    time_constraint = 500


                # Create a time constraint if we are trying to build a base
                if shipyard.ship_count == largest_launch_yard and npv_base > 0.0 and \
                        valuation.ships[me.id] + shipyard.ship_count > 100:

                    if largest_launch_yard < 50:
                        when = np.nonzero(m_board.own_phantom_v1[:, shipyard.position.x, shipyard.position.y] +
                                          largest_launch_yard >= 50)
                        if len(when[0]) > 0:
                            time_constraint = when[0][0]

                # Determine the best available path given constraints
                best_paths = g.get_best_path_and_value(m_board, dm, me.id,
                                                       ships_at_yard=shipyard.ship_count, fleet_size=size,
                                                       my_ship_val=10.0, enemy_ship_val=10.0,
                                                       my_kore_val=1, enemy_kore_val=1, time_constraint=time_constraint,
                                                       base_build=False)

                # Merge best paths
                for idx, prior, new in zip(range(0, len(best_paths)), prior_best_paths, best_paths):
                    if prior[2] < new[2]:
                        prior_best_paths[idx] = new
                best_paths = prior_best_paths

                if prior_best_paths[len(prior_best_paths) - 1][2] >= 10000: # Don't bother checking smaller ones if we have a whale on our hands
                    break
                end = time.time()

            if best_paths is not None:
                best_path = valuation.income_maximizing_size(me.id, best_paths, max_spawn, maximum_size)

                if best_path is not None:
                    while best_path[2] != BAD_VAL:
                        if best_path[0] > shipyard.ship_count:
                            if kore_left > spawn_cost:
                                best_path = None
                                break
                            else:
                                best_paths[best_path[0]] = [best_path[0], 0, BAD_VAL, BAD_VAL, 'DOCK', 0, 0, 0]
                                best_path = valuation.income_maximizing_size(me.id, best_paths, max_spawn, maximum_size)
                                if best_path is None:
                                    break
                        else:
                            break

            end = time.time()
            t_graph = end - start

            #best_path = best_paths[len(best_paths) - 1]

            # Fleet splitting conditions
            # if m_board.kore_bank[me.id][0] / max(1, valuation.network_capacity[me.id][0]) < 4 and best_path[4] < 1000.0:
            #best_path = valuation.income_maximizing_size(me.id, best_paths, max_spawn)

            if best_path is not None:
                print("Norm")
                maximizing_fleet_size = best_path[0]
                top_score = best_path[2]
                top_path = best_path[4]
                top_t = best_path[5]
                top_x = best_path[6]
                top_y = best_path[7]

                # If this is a risky mission, do it as a zombie
                if m_board.true_bop[top_t, top_x, top_y] <= 0 and m_board.enemy_bases[top_t, top_x, top_y] == 1:
                    if board.step <= 100:
                        best_path = None
                        top_score = BAD_VAL
                        top_path = 'DOCK'
                    else:
                        m_board.base_zombie_val[:top_t+1, top_x, top_y] = m_board.base_val[:top_t+1, top_x, top_y] + m_board.base_disc_val[:top_t+1, top_x, top_y]
                        g = Graph(board.current_player_id, m_board,
                                  shipyard.position, shipyard.ship_count, base_create=False,
                                  kill_switch=False)  # Build base set of paths without base creation
                        g.build(shipyard.ship_count, 10.0, 10.0, 1.0, 1.0, 0, 0, TIME_LEVEL, zombie_start=ZOMBIE)
                        best_paths = g.get_best_path_and_value(m_board, dm, me.id,
                                                               ships_at_yard=shipyard.ship_count, fleet_size=shipyard.ship_count,
                                                               my_ship_val=10.0, enemy_ship_val=10.0,
                                                               my_kore_val=1, enemy_kore_val=1,
                                                                base_build=False)
                        print("Risky")
                        best_path = best_paths[shipyard.ship_count]
                        top_score = best_path[2]
                        top_path = best_path[4]

                # See if we can invade enemy bases
                if board.step > 100:
                    my_base_ct = len(m_board.yards[me.id][m_board.lookahead])
                    best_val = top_score
                    best_target = None
                    best_t_first = None
                    best_dist = None
                    best_opportunity = None
                    for idx, yard in zip(range(0, len(m_board.yards[enemy][m_board.lookahead])), m_board.yards[enemy][m_board.lookahead]):
                        if not yard.captured_end:
                            distance = dm.calc_distance(shipyard.position.x, shipyard.position.y, yard.x, yard.y)
                            if distance <= m_board.lookahead:
                                bop = np.max(m_board.true_bop[distance:, yard.x, yard.y])
                                t_first = max(0, m_board.lookahead - yard.turns_controlled)
                                if bop > 0:
                                    value = base_vals[my_base_ct + idx]
                                    first_opportunity = np.nonzero(m_board.true_bop[distance:, yard.x, yard.y] > 0)[0][0] + distance
                                    if value > best_val:
                                        best_val = value
                                        best_dist = distance
                                        best_t_first = t_first
                                        best_opportunity = first_opportunity
                                        best_target = Point(yard.x, yard.y)

                    # Apply the invasion strategy
                    if best_target is not None:
                        for yard in m_board.yards[me.id][m_board.lookahead]:
                            time_to_reach = dm.calc_distance(shipyard.position.x, shipyard.position.y, yard.x, yard.y)
                            onward_distance = dm.calc_distance(yard.x, yard.y, best_target.x, best_target.y)
                            total_time = time_to_reach + onward_distance
                            if total_time > best_opportunity: # Too slow
                                m_board.avoid[:, yard.x, yard.y] = 1
                            else:
                                m_board.base_zombie_val[:time_to_reach+1, yard.x, yard.y] = best_val
                        for yard in m_board.yards[enemy][m_board.lookahead]:
                            if yard.x == best_target.x and yard.y == best_target.y: # Apply the zombie values
                                m_board.base_zombie_val[best_opportunity, yard.x, yard.y] = best_val
                            else:
                                if not yard.captured_end:
                                    m_board.avoid[:, yard.x, yard.y] = 1
                        # Do the zombie march
                        print("Zombie march")
                        g = Graph(board.current_player_id, m_board,
                                  shipyard.position, shipyard.ship_count, base_create=False,
                                  kill_switch=False)  # Build base set of paths without base creation
                        g.build(shipyard.ship_count, 10.0, 10.0, 1.0, 1.0, 0, 0, TIME_LEVEL, zombie_start=ZOMBIE)
                        best_paths = g.get_best_path_and_value(m_board, dm, me.id,
                                                               ships_at_yard=shipyard.ship_count,
                                                               fleet_size=shipyard.ship_count,
                                                               my_ship_val=10.0, enemy_ship_val=10.0,
                                                               my_kore_val=1, enemy_kore_val=1,
                                                               base_build=False)
                        best_path = best_paths[shipyard.ship_count]
                        if best_path[2] > top_score:
                            print("March")
                            top_score = best_path[2]
                            top_path = best_path[4]
                            top_t = best_path[5]
                            top_x = best_path[6]
                            top_y = best_path[7]
                            maximizing_fleet_size = shipyard.ship_count

                            # If we ended up at a home base, zombie is not appropriate
                            if m_board.my_bases[top_t, top_x, top_y] == 1:
                                # Find distance from base to target
                                onward_distance = dm.calc_distance(top_x, top_y, best_target.x, best_target.y)
                                time_constraint = best_opportunity - onward_distance
                                for yard in m_board.yards[me.id][m_board.lookahead]:
                                    if yard.x != top_x or yard.y != top_y:
                                        m_board.avoid[:, yard.x, yard.y] = 1
                                g = Graph(board.current_player_id, m_board,
                                          shipyard.position, shipyard.ship_count, base_create=False,
                                          kill_switch=False)  # Build base set of paths without base creation
                                g.build(shipyard.ship_count, 10.0, 10.0, 1.0, 1.0, 1, 0, TIME_LEVEL, zombie_start=NOZOMBIE)
                                best_paths = g.get_best_path_and_value(m_board, dm, me.id,
                                                                       ships_at_yard=shipyard.ship_count,
                                                                       fleet_size=shipyard.ship_count,
                                                                       my_ship_val=10.0, enemy_ship_val=10.0,
                                                                       my_kore_val=1, enemy_kore_val=1, time_constraint=time_constraint,
                                                                       base_build=False)
                                best_path = best_paths[shipyard.ship_count]
                                print("Inappropriate")
                                top_score = best_path[2]
                                top_path = best_path[4]

                if isinstance(top_path, float):
                    print(best_path)


                if top_path is not None and len(top_path) > 0 and top_path != 'DOCK':  # Otherwise skip turn
                    action = ShipyardAction.launch_fleet_with_flight_plan(
                        maximizing_fleet_size, top_path)

                # If we have extra ships which don't need to remain docked
                if top_path == 'DOCK' and maximizing_fleet_size < shipyard.ship_count - 1: # Don't need to dock ALL of our ships
                    prior_val = top_score
                    m_board.base_val[:, shipyard.position.x, shipyard.position.y] = 0.0
                    launch_size = shipyard.ship_count - maximizing_fleet_size
                    g = Graph(board.current_player_id, m_board,
                              shipyard.position, launch_size, base_create=False, kill_switch=False)
                    g.build(launch_size, 10.0, 10.0, 1.0, 1.0,
                            0, turn_penalty, TIME_LEVEL, zombie_start=False)
                    best_paths = g.get_best_path_and_value(m_board, dm, me.id,
                                                           ships_at_yard=shipyard.ship_count, fleet_size=launch_size,
                                                           my_ship_val=10.0, enemy_ship_val=10.0,
                                                           my_kore_val=1, enemy_kore_val=1,
                                                           base_build=False)
                    if best_paths is not None:
                        best_path = valuation.income_maximizing_size(me.id, best_paths, max_spawn, maximum_size)
                        t_best = 0
                        maximizing_fleet_size = best_path[0]
                        top_score = best_path[2] + prior_val
                        top_path = best_path[4]

            t_base = 0
            if decision == 'launch' and shipyard.ship_count >= 50 and board.step <= 350 and (top_path is None or top_path != 'DOCK') and not kill_switch \
                    and (valuation.expected_income[me.id][0]*4 + m_board.kore_bank[me.id][0]) / valuation.network_capacity[me.id][0] > 4.0 and npv_base > top_score and \
                    (valuation.ships[me.id] + shipyard.ship_count) / len(m_board.yards[me.id][0]) >= 100:
                start = time.time()
                base_loc = optimal_new_base_v6(m_board, valuation, shipyard.ship_count, dm)
                end = time.time()
                t_base = end - start

                #base_loc = Point(6, 14)
                if base_loc:
                    print("Base location (" + str(base_loc.x) +
                          ", " + str(base_loc.y) + ")")
                    create_base(m_board, base_loc, npv_base)

                    #dump_strategy(m_board)

                    g = Graph(board.current_player_id, m_board,
                              shipyard.position, shipyard.ship_count,
                              base_create=True, kill_switch=False)  # Build base set of paths without base creation
                    #g.build(shipyard.ship_count, sq_valuation.ship_values[enemy], 0, 250.0, TIME_LEVEL) # Bigger time penalty when on build path
                    g.build(shipyard.ship_count, 10.0, 10.0, 1.0, 1.0,
                            0, 40.0, TIME_LEVEL, zombie_start=0)
                    best_paths = g.get_best_path_and_value(m_board, dm, me.id,
                                                           ships_at_yard=shipyard.ship_count,
                                                           fleet_size=shipyard.ship_count,
                                                           my_ship_val=10.0,
                                                           enemy_ship_val=10.0,
                                                   my_kore_val=1.0, enemy_kore_val=1.0, base_build=True)
                    # time_constraint=m_board.capture_turn(shipyard.position),
                    # base_build=True)
                    best_path = best_paths[len(best_paths) - 1]

                    if best_path and best_path[2] > 0.0 and best_path[2] > top_score:
                        maximizing_fleet_size = best_path[0]
                        top_score = best_path[2]
                        top_path = best_path[4]
                        print(str(board.step) + ':{} Path '.format(maximizing_fleet_size) + top_path + ' [{:.3f}]'.format(
                            top_score))
                        if len(top_path) > 0 and top_path != "DOCK":  # Otherwise skip turn
                            action = ShipyardAction.launch_fleet_with_flight_plan(
                                maximizing_fleet_size, top_path)
                        end = time.time()
                        t_base = end - start
                    else:
                        if top_path is not None:
                            print(str(board.step) + ':{} Path '.format(maximizing_fleet_size) + top_path + ' [{:.3f}]'.format(
                                top_score))
                            action = ShipyardAction.launch_fleet_with_flight_plan(
                                maximizing_fleet_size, top_path)
                        else:
                            shipyard.next_action = None
                            print(str(board.step) + ": None")

            if top_path is not None:
                print(str(board.step) + ':{} Path '.format(maximizing_fleet_size) + top_path + ' [{:.3f}]'.format(
                    top_score))
                if len(top_path) > 0 and top_path != "DOCK":  # Otherwise skip turn
                    action = ShipyardAction.launch_fleet_with_flight_plan(
                        maximizing_fleet_size, top_path)

            # If this launch serves no strategic value, if we're below preferred size, and there's another inbound fleet, spawn instead
            if valuation is not None and action == ShipyardActionType.LAUNCH:
                if np.sum(m_board.phantom[me.id, 1, :, shipyard.position.x, shipyard.position.y]) - shipyard.max_spawn > np.sum(m_board.phantom[me.id, 0, :, shipyard.position.x, shipyard.position.y]) and \
                        maximizing_fleet_size < preferred_size and kore_left >= valuation.network_capacity[me.id][0] * spawn_cost:
                    decision = 'spawn'
                    action = None

            shipyard.next_action = action
            if action is not None and action.action_type == ShipyardActionType.LAUNCH:
                print("Timing: Mom {:.2f} Src {:.2f} Val {:.2f} Lbl {:.2f} Grph {:.2f} Base {:.2f} TOTAL: {:.2f}".format(
                    t_momentum, t_source_matrices, t_valuation, t_label, t_graph, t_base, t_base + t_momentum + t_source_matrices + t_label + t_valuation + t_graph + t_base))
                if not queue.isEmpty():
                    start = time.time()
                    old_pos = shipyard.position
                    old_count = shipyard.ship_count
                    shipyard = queue.pop()
                    m_board.incremental_update(old_pos, shipyard.position, top_path, maximizing_fleet_size, old_count)
                    end = time.time()
                    t_momentum = end - start
                    continue
                else:
                    shipyard = None
                    break

            if queue.isEmpty():
                break
            else:
                shipyard = queue.pop()
        else:
            if queue.isEmpty():
                break
            else:
                shipyard = queue.pop()

    ##############################
    # Spawn at very end if we aren't launching
    #############################

    if not halt_production:
        while not spawn_queue.isEmpty():
            yard, value, turns = spawn_queue.pop()
            shipyard = locate_yard(me, yard)
            if shipyard is None or shipyard.next_action is not None:
                continue

            # turns = min(turns, len(m_board.kore_bank[me.id]))
            kore_left = produce(me, yard, kore_left)

        # Double check to make sure we aren't spawning at very end
        if valuation is not None and valuation.ship_npv[0] <= 0.0 and board.step > 350 and valuation.ships[
            me.id] > valuation.total_ships * 0.5 or board.step > 395:
            for shipyard in me.shipyards:
                if shipyard.next_action is not None and shipyard.next_action.action_type == ShipyardActionType.SPAWN:
                    shipyard.next_action = None
                    print("Cancel spawn")

    # pr.dump_stats('c:\\profile_label.pstat')
    return me.next_actions

def asymmetric_agent(obs, config):
    board = Board(obs, config)
    if board.step <= 22:
        return balanced_agent(obs, config)
    else:
        return agent(obs, config)
