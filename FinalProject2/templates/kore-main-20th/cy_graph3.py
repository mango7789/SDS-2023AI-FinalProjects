import cython
import math
#from cpython cimport array
#from cython.cimports import numpy as np

#cimport numpy as np
import numpy as np

SIZE = cython.declare(cython.int, 21)
MAX_LOOP = cython.declare(cython.int, 21)
BOARD_SIZE = cython.declare(cython.int, SIZE * SIZE)
BAD_VAL = cython.declare(cython.float, -9999999999.0)
FACES = cython.declare(cython.int, 5)

NONE = cython.declare(cython.int, 0)
NORTH = cython.declare(cython.int, 1)
SOUTH = cython.declare(cython.int, 2)
EAST = cython.declare(cython.int, 3)
WEST = cython.declare(cython.int, 4)

DONT_CONTINUE = cython.declare(cython.int, 0)
CONTINUE = cython.declare(cython.int, 1)
FALSE = cython.declare(cython.int, 0)
TRUE = cython.declare(cython.int, 1)
NOKILL = cython.declare(cython.int, 0)
KILL = cython.declare(cython.int, 1)
NOZOMBIE = cython.declare(cython.int, 0)
ZOMBIE = cython.declare(cython.int, 1)
OUTBREAK = cython.declare(cython.int, 2)
BLOODSPILLED = cython.declare(cython.int, 2)

@cython.cclass
class Source:
    terminal = cython.declare(cython.int[:, :, ::1])
    kore_val = cython.declare(cython.float[:, :, ::1])
    base_val = cython.declare(cython.float[:, :, ::1])
    base_disc_val = cython.declare(cython.float[:, :, ::1])
    base_zombie_val = cython.declare(cython.float[:, :, ::1])
    cargo_val = cython.declare(cython.float[:, :, ::1])
    cargo_destroyed_val = cython.declare(cython.float[:, :, ::1])
    fleet_strat_val = cython.declare(cython.float[:, :, ::1])
    own_fleet = cython.declare(cython.int[:, :, ::1])
    enemy_fleet = cython.declare(cython.int[:, :, ::1])
    enemy_fleet_indices = cython.declare(cython.int[:, :, ::1])
    enemy_phantom = cython.declare(cython.int[:, :, ::1])
    avoid = cython.declare(cython.int[:, :, ::1])
    enemy_bases = cython.declare(cython.int[:, :, ::1])

@cython.cclass
class Nodes:
    kore_gained = cython.declare(cython.float[:, :, :, :, :, :, ::1])
    kore_lost = cython.declare(cython.float[:, :, :, :, :, :, ::1])
    turn_penalty = cython.declare(cython.float[:, :, :, :, :, :, ::1])
    cargo_destroyed = cython.declare(cython.float[:, :, :, :, :, :, ::1])
    ships_destroyed = cython.declare(cython.int[:, :, :, :, :, :, ::1])
    ships_lost = cython.declare(cython.int[:, :, :, :, :, :, ::1])
    fleet_strat = cython.declare(cython.float[:, :, :, :, :, :, ::1])
    base = cython.declare(cython.float[:, :, :, :, :, :, ::1])
    base_disc = cython.declare(cython.float[:, :, :, :, :, :, ::1])
    exists = cython.declare(cython.int[:, :, :, :, :, :, ::1])
    minfleet = cython.declare(cython.int[:, :, :, :, :, :, ::1])
    rollup = cython.declare(cython.int[:, :, :, :, :, :, ::1])
    bp_idx = cython.declare(cython.int[:, :, :, :, :, :, ::1])
    dest_terminal = cython.declare(cython.int[:, :, :, :, :, :, ::1])
    best_direction = cython.declare(cython.int[:, :, :, :, :, :, ::1])

PathStats = cython.struct(valid=cython.int, kore_gained=cython.float, kore_lost=cython.float, turn_penalty=cython.float,
                          cargo_destroyed=cython.float, ships_destroyed=cython.int, ships_lost=cython.int, fleet_strat=cython.float,
                          base=cython.float, base_disc=cython.float, kill=cython.int, zombie=cython.int,
                          minfleet=cython.int, rollup=cython.int)

Instance = cython.struct(p=cython.int, t=cython.int, x=cython.int, y=cython.int, z=cython.int, k=cython.int, f=cython.int, direct=cython.int)

Config = cython.struct(fleet_size=cython.int, my_ship_value=cython.float,
        enemy_ship_value=cython.float, my_kore_value=cython.float, enemy_kore_value=cython.float,
        time_constraint=cython.int, min_time_constraint=cython.int, origin_x=cython.int, origin_y=cython.int,
        turn_cost=cython.float)

@cython.cclass
class Next: # Wraparound manager
    x = cython.declare(cython.int[:, ::1])
    y = cython.declare(cython.int[:, ::1])

    def __init__(self):
        x_org = cython.declare(cython.int)
        y_org = cython.declare(cython.int)

        self.x = np.zeros((SIZE, FACES), dtype=np.intc) # Lookup matrices for wraparound calcs
        self.y = np.zeros((SIZE, FACES), dtype=np.intc) 

        # Init wraparound matrices
        for x_org in range(0, SIZE):
            self.x[x_org, EAST] = x_org + 1
            self.x[x_org, WEST] = x_org - 1
            self.x[x_org, NORTH] = x_org # Unchanged
            self.x[x_org, SOUTH] = x_org
        self.x[SIZE-1, EAST] = 0
        self.x[0, WEST] = SIZE-1

        for y_org in range(0, SIZE):
            self.y[y_org, NORTH] = y_org + 1
            self.y[y_org, SOUTH] = y_org - 1
            self.y[y_org, EAST] = y_org
            self.y[y_org, WEST] = y_org
        self.y[SIZE-1, NORTH] = 0
        self.y[0, SOUTH] = SIZE-1

# Create graph
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.locals(LOOK=cython.int, time_constraint=cython.int, min_time_constraint=cython.int, turn_cost=cython.float,
               zombie_start=cython.int, origin_x=cython.int, origin_y=cython.int,
               fleet_size=cython.int, max_path=cython.int, my_ship_value=cython.float, enemy_ship_value=cython.float,
               my_kore_value=cython.float, enemy_kore_value=cython.float,
               source_terminal=cython.int[:, :, ::1], source_kore_val=cython.float[:, :, ::1],
               source_base_val=cython.float[:, :, ::1], source_base_disc_val=cython.float[:, :, ::1],
               source_base_zombie_val=cython.float[:, :, ::1], source_cargo_val=cython.float[:, :, ::1],
               source_fleet_strat_val=cython.float[:, :, ::1],
               source_own_fleet=cython.int[:,:,::1], source_enemy_fleet=cython.int[:,:,::1],
               source_enemy_fleet_indices=cython.int[:,:,::1], source_enemy_phantom=cython.int[:,:,::1],
               source_avoid=cython.int[:,:,::1], source_enemy_bases=cython.int[:,:,::1],
               n_kore_gained=cython.float[:,:,:,:,:,:,::1],
               n_kore_lost=cython.float[:,:,:,:,:,:,::1], n_turn_penalty=cython.float[:,:,:,:,:,:,::1],
               n_cargo_destroyed=cython.float[:,:,:,:,:,:,::1], n_ships_destroyed=cython.int[:,:,:,:,:,:,::1],
               n_ships_lost=cython.int[:,:,:,:,:,:,::1], n_fleet_strat=cython.float[:,:,:,:,:,:,::1],
               n_base=cython.float[:,:,:,:,:,:,::1], n_base_disc=cython.float[:,:,:,:,:,:,::1],
               n_exists=cython.int[:,:,:,:,:,:,::1], n_minfleet=cython.int[:,:,:,:,:,:,::1],
               n_bp_idx=cython.int[:,:,:,:,:,:,::1], n_dest_terminal=cython.int[:,:,:,:,:,:,::1],
               n_best_direction=cython.int[:,:,:,:,:,:,::1])
def build_graph(LOOK, time_constraint, min_time_constraint, turn_cost, zombie_start, origin_x, origin_y, fleet_size,
                max_path, my_ship_value, enemy_ship_value, my_kore_value, enemy_kore_value, source_terminal, source_kore_val,
                source_base_val, source_base_disc_val, source_base_zombie_val, source_cargo_val,
                source_fleet_strat_val, source_own_fleet, source_enemy_fleet, source_enemy_fleet_indices,
                source_enemy_phantom,  source_avoid, source_enemy_bases, n_kore_gained, n_kore_lost, n_turn_penalty,
                n_cargo_destroyed, n_ships_destroyed, n_ships_lost, n_fleet_strat, n_base, n_base_disc,
                n_exists, n_minfleet,  n_bp_idx,  n_dest_terminal,  n_best_direction):

    if cython.compiled:
        print("CyGraph compiled")
    else:
        print("CyGraph interpreted")

    P_DEPTH = cython.declare(cython.int, max_path + 1)
    FACES = cython.declare(cython.int, 5)

    i = cython.declare(cython.int)
    value = cython.declare(cython.float)
    found = cython.declare(cython.int, 1)
    zombie_outbreak = cython.declare(cython.int, zombie_start + 1)
    blood_spilled = cython.declare(cython.int, 1)

    stats = cython.declare(PathStats)

    org = cython.declare(Instance)
    dest = cython.declare(Instance)

    mining_rates = cython.declare(cython.float[:])


    config = cython.declare(Config)
    config.fleet_size = fleet_size
    config.my_ship_value = my_ship_value
    config.enemy_ship_value = enemy_ship_value
    config.my_kore_value = my_kore_value
    config.enemy_kore_value = enemy_kore_value
    config.time_constraint = time_constraint
    config.min_time_constraint = min_time_constraint
    config.origin_x = origin_x
    config.origin_y = origin_y
    config.turn_cost = turn_cost


    mining_rates = np.zeros((5*fleet_size+1), np.single)
    for i in range(1, 5*fleet_size+1):
        mining_rates[i] = min(.99, math.log(i) / 20)

    src = Source() # Populate the source data struct
    src.terminal = source_terminal
    src.kore_val = source_kore_val
    src.base_val = source_base_val
    src.base_disc_val = source_base_disc_val
    src.base_zombie_val = source_base_zombie_val
    src.cargo_val = source_cargo_val
    src.fleet_strat_val = source_fleet_strat_val
    src.own_fleet = source_own_fleet
    src.enemy_fleet = source_enemy_fleet
    src.enemy_fleet_indices = source_enemy_fleet_indices
    src.enemy_phantom = source_enemy_phantom
    src.avoid = source_avoid
    src.enemy_bases = source_enemy_bases

    # Node tracking matrices
    n = Nodes()
    n.kore_gained = n_kore_gained
    n.kore_lost = n_kore_lost
    n.turn_penalty = n_turn_penalty
    n.cargo_destroyed = n_cargo_destroyed
    n.ships_destroyed = n_ships_destroyed
    n.ships_lost = n_ships_lost
    n.fleet_strat = n_fleet_strat
    n.base = n_base
    n.base_disc = n_base_disc
    n.exists = n_exists
    n.minfleet = n_minfleet
    n.bp_idx = n_bp_idx
    n.dest_terminal = n_dest_terminal
    n.best_direction = n_best_direction
    n.rollup = np.zeros((n_bp_idx.shape[0], n_bp_idx.shape[1], n_bp_idx.shape[2], n_bp_idx.shape[3],
                         n_bp_idx.shape[4], n_bp_idx.shape[5], n_bp_idx.shape[6]), dtype=np.intc)

    p_org = cython.declare(cython.int) # Path step iterator
    t_org = cython.declare(cython.int) # Time step iterator
    x_org = cython.declare(cython.int) # X-axis iterator
    y_org = cython.declare(cython.int) # Y-axis iterator
    k_org = cython.declare(cython.int) # Kill flag iterator
    z_org = cython.declare(cython.int) # Zombie flag iterator

    p_dest_max = cython.declare(cython.int) # Maximum additional path to follow
    t_dest_max = cython.declare(cython.int) # Maximum additional time to follow

    direct = cython.declare(cython.int) # Direction iterator
    l_it = cython.declare(cython.int) # Flight loop iterator
    quit_sweep = cython.declare(cython.int) # Should we quit out of sweep?

    nxt = Next() # Wraparound manager

    # Set the origin values
    n.kore_gained[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
    n.kore_lost[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
    n.turn_penalty[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
    n.cargo_destroyed[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
    n.ships_destroyed[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0
    if zombie_start:
        n.ships_lost[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = fleet_size
    else:
        n.ships_lost[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0
    n.fleet_strat[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
    n.base[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
    n.base_disc[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
    n.exists[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = TRUE
    n.dest_terminal[0, 0, origin_x, origin_y, zombie_start, NOKILL, NONE] = FALSE

    # See if there will be enemies arriving in t=1, in which case create a path to stay home
    if src.enemy_fleet[1, origin_x, origin_y] > 0:
        if fleet_size >= src.enemy_fleet[1, origin_x, origin_y]:
            n.kore_gained[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = src.cargo_val[1, origin_x, origin_y]
            n.kore_lost[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = 0.0
            n.turn_penalty[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = 0.0
            n.cargo_destroyed[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = src.cargo_destroyed_val[1, origin_x, origin_y]
            n.ships_destroyed[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = src.enemy_fleet[1, origin_x, origin_y]
            if zombie_start:
                n.ships_lost[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = fleet_size
            else:
                n.ships_lost[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = src.enemy_fleet[1, origin_x, origin_y]
            n.fleet_strat[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = src.fleet_strat_val[1, origin_x, origin_y]
            n.base[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = src.base_val[1, origin_x, origin_y]
            n.base_disc[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = src.base_disc_val[1, origin_x, origin_y]
            n.exists[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = 1
            n.dest_terminal[0, 1, origin_x, origin_y, zombie_start, KILL, NONE] = TRUE
        else:
            n.kore_gained[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = 0.0
            n.kore_lost[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = 0.0
            n.turn_penalty[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = 0.0
            n.cargo_destroyed[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = 0.0
            n.ships_destroyed[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = fleet_size
            n.ships_lost[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = fleet_size
            n.fleet_strat[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = 0.0
            n.base[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = src.base_zombie_val[1, origin_x, origin_y]
            n.base_disc[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = 0.0
            n.exists[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = 1
            n.dest_terminal[0, 1, origin_x, origin_y, ZOMBIE, KILL, NONE] = TRUE
    else: # Create a terminal in t=1 for the "DOCK" case
        n.kore_gained[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
        n.kore_lost[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
        n.turn_penalty[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
        n.cargo_destroyed[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
        n.ships_destroyed[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0
        if zombie_start:
            n.ships_lost[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = fleet_size
        else:
            n.ships_lost[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0
        n.fleet_strat[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
        if zombie_start:
            n.base[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = src.base_zombie_val[1, origin_x, origin_y]
            n.base_disc[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
        else:
            n.base[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = src.base_val[1, origin_x, origin_y]
            n.base_disc[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 0.0
        n.exists[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = 1
        n.dest_terminal[0, 1, origin_x, origin_y, zombie_start, NOKILL, NONE] = TRUE

    # Set up initial stats variable
    org.p = 0
    org.t = 0
    org.x = origin_x
    org.y = origin_y
    org.z = zombie_start
    org.k = NOKILL
    org.f = NONE
    stats = stats_from_node(org, n)

    if zombie_start == TRUE:
        stats = become_zombie(stats, config, captured=FALSE)
        zombie_outbreak = OUTBREAK

    breakpoints = []
    breakpoints.append([0,0,15,5,0,0,1])
    breakpoints.append([2,9,15,14,0,0,2])
    breakpoints.append([4,16,15,7,0,0,1])
    breakpoints.append([6,18,15,9,0,0,2])

    for p_org in range(0, P_DEPTH):
        t_dest_max = min(LOOK, MAX_LOOP + p_org, config.time_constraint) # Maximum time of destination is lesser of lookahead and (t-org + max loop)
        found = 1
        for t_org in range(p_org, t_dest_max):  # Ok, don't mess with this
            if found == 0: # Early termination if we've exhausted all paths
                break
            else:
                found = 0
            for z_org in range(zombie_start, zombie_outbreak):
                for k_org in range(0, blood_spilled):
                    for x_org in range(0, SIZE):
                        for y_org in range(0, SIZE):
                            for direct in range(1, 5):

                                org.p = p_org
                                org.t = t_org
                                org.x = x_org
                                org.y = y_org
                                org.z = z_org
                                org.k = k_org

                                dest.direct = direct

                                if org.p == P_DEPTH - 1: # Determine face of origin node. For all but very end this should be NONE
                                    org.f = direct
                                    dest.p = org.p       # If we're at max depth, then don't increment further
                                else:
                                    org.f = NONE
                                    dest.p = org.p + 1

                                #************
                                # Run some validity checks before getting going
                                #************

                                if not cython.compiled:
                                    for b in breakpoints:
                                        if b[0] == org.p and b[1] == org.t and b[2] == org.x and b[3] == org.y and \
                                                b[4] == org.z and b[5] == org.k and b[6] == dest.direct:
                                            print("breakpoint")
                                            break

                                # Check: the origin node exists
                                if n.exists[org.p, org.t, org.x, org.y, org.z, org.k, org.f] == FALSE:
                                    continue

                                # Check: the origin node isn't terminal
                                if n.dest_terminal[org.p, org.t, org.x, org.y, org.z, org.k, org.f] == TRUE:
                                    continue

                                # print("P" + str(org.p) + "T" + str(org.t) + "X" + str(org.x) + "Y" + str(
                                #     org.y) + "Z" + str(org.z) + "K" + str(org.k) + "F" + str(org.f) + "D" + str(
                                #     dest.direct))

                                # Check: not past our lookahead (should be prevented by loops already, since we're at first step)
                                dest.t = org.t + 1

                                quit_sweep = FALSE

                                # Check that we are continuing in same direction at final step
                                if dest.p == P_DEPTH - 1 and org.t != 0:
                                    if org.f != dest.direct and org.f != NONE:
                                        continue
                                else: # Check: our best route to origin node wasn't in the same facing, in which case this is liable to be redundant
                                    if org.z == dest.z and n.bp_idx[org.p, org.t, org.x, org.y, org.z, org.k, org.f] % FACES == dest.direct:
                                        continue # Truncate search if best path to this location was in the same direction, since
                                                # this will cause search to repeat covered ground, unless this is a blank path

                                # Set up temp variables to track our progress through the sweep
                                stats = stats_from_node(org, n)

                                #*****************
                                # Path length = 0
                                #*****************

                                # Already coded above

                                #*****************
                                # Path length = 1
                                #*****************
                                dest.x = nxt.x[org.x, dest.direct]
                                dest.y = nxt.y[org.y, dest.direct]
                                dest.z = org.z
                                dest.k = org.k

                                if dest.p == P_DEPTH - 1: # Facing direction at destination only matters if at P_DEPTH-1
                                    dest.f = dest.direct
                                else:
                                    dest.f = NONE

                                stats = relax(dest, org, n, src, stats, nxt, config, mining_rates)
                                if stats.valid == FALSE:
                                    continue
                                else:
                                    found = 1 # Mark that at least one node exists in this time period
                                if stats.zombie == ZOMBIE:
                                    zombie_outbreak = OUTBREAK
                                if stats.kill == KILL:
                                    blood_spilled = BLOODSPILLED

                                #*****************
                                # Path length = 2
                                #*****************
                                if dest.p <= P_DEPTH - 1:
                                    if dest.p == P_DEPTH - 1: # Facing direction at destination only matters if at P_DEPTH-2
                                        dest.f = dest.direct
                                    else:
                                        dest.p = dest.p + 1
                                        dest.f = NONE
                                    for l_it in range(1, 10): # Nine steps
                                        dest.x = nxt.x[dest.x, dest.direct]
                                        dest.y = nxt.y[dest.y, dest.direct]

                                        dest.t = dest.t + 1

                                        if dest.t > LOOK:
                                            quit_sweep = TRUE
                                            break

                                        stats = relax(dest, org, n, src, stats, nxt, config, mining_rates)
                                        if stats.valid == FALSE:
                                            quit_sweep = TRUE
                                            break
                                        if stats.zombie == ZOMBIE:
                                            zombie_outbreak = OUTBREAK
                                        if stats.kill == KILL:
                                            blood_spilled = BLOODSPILLED

                                    if quit_sweep == TRUE:
                                        continue
                                else:
                                    continue

                                #*****************
                                # Path length = 3
                                #*****************
                                if dest.p <= P_DEPTH - 1:
                                    if dest.p == P_DEPTH - 1: # Facing direction at destination only matters if at P_DEPTH-2
                                        dest.f = dest.direct
                                    else:
                                        dest.p = dest.p + 1
                                        dest.f = NONE

                                    for l_it in range(10, MAX_LOOP + 1):
                                        dest.x = nxt.x[dest.x, dest.direct]
                                        dest.y = nxt.y[dest.y, dest.direct]

                                        dest.t = dest.t + 1

                                        if dest.t > LOOK:
                                            quit_sweep = TRUE
                                            break

                                        stats = relax(dest, org, n, src, stats, nxt, config, mining_rates)

                                        if stats.valid == FALSE:
                                            quit_sweep = TRUE
                                            break
                                        if stats.zombie == ZOMBIE:
                                            zombie_outbreak = OUTBREAK
                                        if stats.kill == KILL:
                                            blood_spilled = BLOODSPILLED


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
@cython.exceptval(check=True)
@cython.returns(PathStats)
@cython.locals(dest=Instance, org=Instance, n=Nodes, src=Source, stats=PathStats, config=Config, nxt=Next, mining_rates=cython.float[:])
def relax(dest, org, n, src, stats, nxt, config, mining_rates):
    # _org values refer to the true origin point for the test
    # p,t,x,y,f refer to the test destination
    nxt_x = cython.declare(cython.int)
    nxt_y = cython.declare(cython.int)
    kill_temp = cython.declare(cython.int, 0)
    largest_kill = cython.declare(cython.int, 0)
    direct_kills = cython.declare(cython.int, 0)
    depletion_penalty = cython.declare(cython.float)
    mining_rate = cython.declare(cython.float)
    captured = cython.declare(cython.int, FALSE)
    value = cython.declare(cython.float)
    prior_value = cython.declare(cython.float)
    terminate = cython.declare(cython.int)

    stats.valid = TRUE # By default

    # Don't double back if we're a zombie
    if zombie_doubleback(n, org, dest):
        stats.valid = FALSE
        return stats

    # Don't go anywhere marked as "avoid"
    if src.avoid[dest.t, dest.x, dest.y] > 0:
        stats.valid = FALSE
        return stats

    # If destination is terminal then make sure f = NONE
    if src.terminal[dest.t, dest.x, dest.y] == TRUE: # Checks destination, NOT origin
        dest.f = NONE
        terminate = TRUE
    else:
        terminate = FALSE

    # Calc depletion penalty for repeat visits to same square
    depletion_penalty = calc_depletion_penalty(dest, org, src, n, config, mining_rates)

    # Calc adjusted mining rate
    if src.kore_val[dest.t, dest.x, dest.y] > 0.0:
        mining_rate = mining_rates[max(0, min(len(mining_rates)-1, config.fleet_size + 1 - stats.minfleet))]
    else:
        mining_rate = 0.0

    if stats.zombie == TRUE:
        stats.kore_lost = stats.kore_lost + (src.kore_val[dest.t, dest.x, dest.y] - depletion_penalty) * mining_rate
        if terminate == TRUE and stats.kill == NOKILL: # Don't get the base credit if we kill a fleet first
            stats.base = stats.base + src.base_zombie_val[dest.t, dest.x, dest.y]*(0.98039)**dest.t
    else:
        stats.kore_gained = stats.kore_gained + (src.kore_val[dest.t, dest.x, dest.y] - depletion_penalty) * mining_rate
        if terminate == TRUE:
            stats.base = stats.base + src.base_val[dest.t, dest.x, dest.y]
            if src.base_disc_val[dest.t, dest.x, dest.y] > 0:
                stats.base_disc = stats.base_disc + src.base_disc_val[dest.t, dest.x, dest.y]*(0.98039)**dest.t

    direction = cython.declare(cython.int)

    # Handle fleet mergers
    if src.own_fleet[dest.t, dest.x, dest.y] > 0:
        if stats.zombie == ZOMBIE: # Never merge while in zombie state
            stats.valid = FALSE
            return stats
        stats.turn_penalty = stats.turn_penalty + 50.0 # Apply a strategic merge penalty
        if stats.zombie == NOZOMBIE and stats.rollup == 0:
            if src.own_fleet[dest.t, dest.x, dest.y] < config.fleet_size + 1 - stats.minfleet :
                stats.minfleet -= src.own_fleet[dest.t, dest.x, dest.y] # Record the merger
                stats.rollup = 1

    # Handle direct fleet collisions
    direct_kills = src.enemy_fleet[dest.t, dest.x, dest.y]
    if stats.zombie == ZOMBIE:
        # If we are a zombie, allowed to eat phantoms at enemy bases
        if src.enemy_bases[dest.t, dest.x, dest.y] == 1:
            direct_kills = max(0, min(src.enemy_phantom[dest.t, dest.x, dest.y] - stats.ships_destroyed, stats.ships_lost - stats.ships_destroyed))
            stats.ships_destroyed += direct_kills
            stats.kill = KILL # Mark as kill so we don't double-count later
    if direct_kills > 0:
        # If we survive
        if config.fleet_size > stats.minfleet - 1 + direct_kills:
            if stats.zombie == NOZOMBIE: # Zombies never take damage
                stats.minfleet += direct_kills
                stats.ships_lost += direct_kills
            if stats.kill == NOKILL: # Only gain credit for our first kill
                stats.ships_destroyed += direct_kills
                stats.fleet_strat += src.fleet_strat_val[dest.t, dest.x, dest.y]
            if stats.kill == NOKILL and stats.zombie == NOZOMBIE:
                stats.kore_gained += src.cargo_val[dest.t, dest.x, dest.y]
                stats.cargo_destroyed += src.cargo_val[dest.t, dest.x, dest.y]
        else:
            # We die
            if stats.rollup == TRUE: # Never go into zombie mode after a rollup
                stats.valid = FALSE
                return stats
            if stats.kill == NOKILL:
                stats.ships_destroyed += (config.fleet_size - (stats.minfleet - 1))
            if direct_kills > config.fleet_size - (stats.minfleet - 1):
                captured = TRUE
            stats = become_zombie(stats, config, captured)

    # Handle orthogonal fleet damage
    if config.fleet_size > stats.minfleet and terminate == FALSE and captured == FALSE: # Ok to be zombie, if we haven't tasted blood yet
        for direction in range(1, 5): #[NORTH, SOUTH, EAST, WEST]:
            nxt_x = nxt.x[dest.x, direction]
            nxt_y = nxt.y[dest.y, direction]
            if src.terminal[dest.t, nxt_x, nxt_y] == FALSE: # Fleet not hiding in a base
                kill_temp = min(src.enemy_fleet[dest.t, nxt_x, nxt_y], config.fleet_size + 1 - stats.minfleet)
                if kill_temp > largest_kill: # Keep track of largest fleet killed, so we can know if any survivors
                    largest_kill = kill_temp
                if kill_temp > 0 and stats.kill == FALSE: # Gain credit for the kills if this is the first combat square we've entered
                    stats.ships_destroyed += kill_temp
                    # Record value only if the adjacent fleet is actually destroyed
                    if config.fleet_size + 1 - stats.minfleet >= src.enemy_fleet[dest.t, nxt_x, nxt_y]:
                        if stats.kill == NOKILL: # Only get value if this is our first turn killing
                            if stats.zombie == ZOMBIE: # Zombies can't derive additional strategy value from killing fleets
                                stats.cargo_destroyed += src.cargo_val[dest.t, nxt_x, nxt_y] * 0.5 # 50% of cargo destroyed, rest will be recaptured
                            else:
                                stats.fleet_strat += src.fleet_strat_val[dest.t, nxt_x, nxt_y]
                                stats.kore_gained += src.cargo_val[dest.t, nxt_x, nxt_y] * 0.5 # Cargo partially captured
                                stats.cargo_destroyed += src.cargo_val[dest.t, nxt_x, nxt_y] * 0.5

        if largest_kill > 0:
            if config.fleet_size > stats.minfleet - 1 + largest_kill: # If we survive
                # Increase our minfleet
                if stats.zombie == NOZOMBIE:
                    stats.minfleet += largest_kill
                    if stats.kill == NOKILL:
                        stats.ships_lost += largest_kill
            else: # We die and become a zombie
                if stats.rollup == TRUE:  # Never go into zombie mode after a rollup
                    stats.valid = FALSE
                    return stats
                if largest_kill > config.fleet_size - (stats.minfleet - 1):
                    captured = TRUE
                stats = become_zombie(stats, config, captured)

    # Record blood on our hands
    if direct_kills > 0 or largest_kill > 0:
        stats.kill = TRUE

    # Don't go there if we risk a needless death (zombies don't fear the reaper)
    if stats.zombie == NOZOMBIE:
        if stats.minfleet-1 + src.enemy_phantom[dest.t, dest.x, dest.y] >= config.fleet_size or \
            stats.minfleet-1 + src.enemy_phantom[dest.t, fix_axis(dest.x+1), dest.y] >= config.fleet_size or \
                stats.minfleet-1 + src.enemy_phantom[dest.t, fix_axis(dest.x-1), dest.y] >= config.fleet_size or \
                    stats.minfleet-1 + src.enemy_phantom[dest.t, dest.x, fix_axis(dest.y+1)] >= config.fleet_size or \
                        stats.minfleet-1 + src.enemy_phantom[dest.t, dest.x, fix_axis(dest.y-1)] >= config.fleet_size:
            stats.valid = FALSE
            return stats

    # Prepare our return value
    if terminate == FALSE:
        stats.valid = TRUE
    else:  # At a terminal destination
        stats.valid = FALSE # Don't continue past our termination point
        # Only treat as a valid choice if time to reach above minimum threshold, unless strategy dictates otherwise
        if dest.t < config.min_time_constraint or stats.base + stats.base_disc + stats.fleet_strat < \
                n.base[0, 0, config.origin_x, config.origin_y, NOZOMBIE, NOKILL, NONE] + \
                n.base_disc[0, 0, config.origin_x, config.origin_y, NOZOMBIE, NOKILL, NONE] + \
                n.fleet_strat[0, 0, config.origin_x, config.origin_y, NOZOMBIE, NOKILL, NONE]:
            stats.valid = FALSE
            return stats # Don't save if we don't meet our min_time or strategy criteria!

    # If everything is ok, update the value and connection

    # Update our destination zombie and kill flags
    dest.z = stats.zombie
    dest.k = stats.kill

    # Apply turn penalty
    #if dest.z == NOZOMBIE:
    stats.turn_penalty = stats.turn_penalty + config.turn_cost
    # else:
    #     stats.turn_penalty = stats.turn_penalty + 0.001 # Speed is tie-breaker for zombies

    if n.exists[dest.p, dest.t, dest.x, dest.y, dest.z, dest.k, dest.f] == TRUE:
        value = calc_value(stats, config)
        prior_value = calc_value(stats_from_node(dest, n), config)
        # Did we improve?
        if value > prior_value:
            save_to_node(stats, org, dest, n, terminate)
    else: # First visit
        save_to_node(stats, org, dest, n, terminate)

    return stats



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
@cython.exceptval(check=True)
@cython.returns(int)
@cython.locals(axis=cython.int)
def fix_axis(axis):
    if axis==-1:
        return SIZE-1
    elif axis==SIZE:
        return 0
    else:
        return axis

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
@cython.exceptval(check=True)
@cython.returns(cython.int)
@cython.locals(n=Nodes, org=Instance, dest=Instance)
def zombie_doubleback(n, org, dest):
    skip = cython.declare(cython.int, FALSE)
    p_prev = cython.declare(cython.int)
    t_prev = cython.declare(cython.int)

    if org.z == ZOMBIE:  # Check to make sure zombies don't double-back on themselves
        if org.p > 1 and org.t > 1:
            for p_prev in range(1, org.p):
                if skip == TRUE:
                    break
                for t_prev in range(1, org.t):
                    if n.exists[p_prev, t_prev, dest.x, dest.y, ZOMBIE, dest.k, dest.f]:
                        skip = TRUE
                        break
    return skip

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
@cython.exceptval(check=True)
@cython.returns(Instance)
@cython.locals(this=Instance, n=Nodes)
def prior_instance(this: Instance, n: Nodes):
    bp_prev = cython.declare(cython.int)
    prev = cython.declare(Instance)
    rem = cython.declare(cython.int)
    bp_prev = n.bp_idx[this.p, this.t, this.x, this.y, this.z, this.k, this.f]
    prev.p = bp_prev // 2500000
    rem = bp_prev % 2500000
    prev.t = rem // 44100
    rem = rem % 44100
    prev.x = rem // 2100
    rem = rem % 2100
    prev.y = rem // 100
    rem = rem % 100
    prev.z = rem // 50
    rem = rem % 50
    prev.k = rem // 25
    rem = rem % 25
    prev.f = rem // 5
    prev.direct = rem % 5
    return prev

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
@cython.exceptval(check=True)
@cython.returns(cython.int)
@cython.locals(i=Instance, direct=cython.int)
def encode_index(i: Instance, direct: int):
    return i.p*2500000 + i.t*44100 + i.x*2100 + i.y*100 + i.z*50 + i.k*25 + i.f*5 + direct

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
@cython.exceptval(check=True)
@cython.returns(PathStats)
@cython.locals(stats=PathStats, config=Config)
def become_zombie(stats, config, captured=FALSE):
    stats = stats
    stats.minfleet = 1
    if captured == TRUE:
        stats.kore_lost += stats.kore_gained
    stats.kore_gained = 0.0
    stats.ships_lost = config.fleet_size
    stats.zombie = ZOMBIE
    return stats

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
@cython.exceptval(check=True)
@cython.returns(cython.float)
@cython.locals(dest=Instance, org=Instance, src=Source, n=Nodes, config=Config, mining_rates=cython.float[:])
def calc_depletion_penalty(dest, org, src, n, config, mining_rates):
    this = cython.declare(Instance)
    prev = cython.declare(Instance)
    rem = cython.declare(cython.int)
    t_overlap = cython.declare(cython.int)
    depletion_penalty = cython.declare(cython.float, 0.0)
    # Determine depletion penalty
    if src.kore_val[dest.t, dest.x, dest.y] > 0.0:
        if dest.t > 1:
            # Look beginning at origin
            this = org
            while this.p > 0:
                # Parse prior node
                prev = prior_instance(this, n)

                # Find out if it overlaps our current space
                t_overlap = -1
                if prev.x == dest.x:
                    if prev.direct == NORTH:
                        if this.y > prev.y:
                            if this.y >= dest.y and dest.y > prev.y:
                                t_overlap = prev.t + dest.y - prev.y
                        else:
                            if dest.y <= this.y:
                                t_overlap = prev.t + dest.y + 21 - prev.y
                            elif dest.y > prev.y:
                                t_overlap = prev.t + dest.y - prev.y
                    elif prev.direct == SOUTH:
                        if this.y < prev.y:
                            if this.y <= dest.y and dest.y < prev.y:
                                t_overlap = prev.t + prev.y - dest.y
                        else:
                            if dest.y >= this.y:
                                t_overlap = prev.t + prev.y + 21 - dest.y
                            elif dest.y < prev.y:
                                t_overlap = prev.t + prev.y - dest.y
                if prev.y == dest.y:
                    if prev.direct == EAST:
                        if this.x > prev.x:
                            if this.x >= dest.x and dest.x > prev.x:
                                t_overlap = prev.t + dest.x - prev.x
                        else:
                            if dest.x <= this.x:
                                t_overlap = prev.t + dest.x + 21 - prev.x
                            elif dest.x > prev.x:
                                t_overlap = prev.t + dest.x - prev.x
                    elif prev.direct == WEST:
                        if this.x < prev.x:
                            if this.x <= dest.x and dest.x < prev.x:
                                t_overlap = prev.t + prev.x - dest.x
                        else:
                            if dest.x >= this.x:
                                t_overlap = prev.t + prev.x + 21 - dest.x
                            elif dest.x < prev.x:
                                t_overlap = prev.t + prev.x - dest.x
                if t_overlap >= 0:
                    mining_rate = mining_rates[
                        config.fleet_size + 1 - max(1, n.minfleet[prev.p, prev.t, prev.x, prev.y, prev.z, prev.k, prev.f])]
                    depletion_penalty = depletion_penalty + (src.kore_val[t_overlap, dest.x, dest.y] * mining_rate) * (1.02) ** (
                                dest.t - t_overlap)
                this = prev
    return depletion_penalty

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
@cython.exceptval(check=True)
@cython.returns(cython.float)
@cython.locals(stats=PathStats, config=Config)
def calc_value(stats: PathStats, config: Config):
    value = cython.declare(cython.float, 0.0)
    value += stats.kore_gained * config.my_kore_value
    value -= stats.kore_lost * config.enemy_kore_value
    value -= stats.turn_penalty
    value += stats.cargo_destroyed
    value += stats.ships_destroyed * config.enemy_ship_value
    value -= stats.ships_lost * config.my_ship_value
    value += stats.fleet_strat
    value += stats.base
    value += stats.base_disc
    return value

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
@cython.returns(PathStats)
@cython.locals(i=Instance, n=Nodes)
def stats_from_node(i: Instance, n: Nodes):
    stats = cython.declare(PathStats)
    stats.kore_gained = n.kore_gained[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.kore_lost = n.kore_lost[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.turn_penalty = n.turn_penalty[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.cargo_destroyed = n.cargo_destroyed[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.ships_destroyed = n.ships_destroyed[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.ships_lost = n.ships_lost[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.fleet_strat = n.fleet_strat[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.base = n.base[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.base_disc = n.base_disc[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.rollup = n.rollup[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.minfleet = n.minfleet[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    stats.kill = i.k
    stats.zombie = i.z
    stats.valid = n.exists[i.p, i.t, i.x, i.y, i.z, i.k, i.f]
    return stats

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cfunc
#@cython.exceptval(-1)
#@cython.exceptval(check=True)
@cython.locals(stats=PathStats, i=Instance, config=Config, terminal=cython.int)
def save_to_node(stats: PathStats, org: Instance, i: Instance, n: Nodes, terminal: int):
    n.bp_idx[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = encode_index(org, i.direct)
    n.kore_gained[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.kore_gained
    n.kore_lost[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.kore_lost
    n.turn_penalty[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.turn_penalty
    n.cargo_destroyed[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.cargo_destroyed
    n.ships_destroyed[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.ships_destroyed
    n.ships_lost[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.ships_lost
    n.fleet_strat[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.fleet_strat
    n.base[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.base
    n.base_disc[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.base_disc
    n.rollup[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.rollup
    n.minfleet[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = stats.minfleet
    n.exists[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = TRUE
    n.dest_terminal[i.p, i.t, i.x, i.y, i.z, i.k, i.f] = terminal
    




