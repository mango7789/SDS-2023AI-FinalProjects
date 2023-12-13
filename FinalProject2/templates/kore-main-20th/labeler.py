import cython
from kaggle_environments.envs.kore_fleets.helpers import Point
from utils import PAR
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.exceptval(-1)
@cython.locals(bop=cython.int[:,:,:], lookahead=cython.int, fleet_size=cython.int, base_vals=cython.float[::1], distances=dict, my_yard_ct=cython.int, t_captured=cython.int[:], t_first=cython.int[:], x=cython.int[:], y=cython.int[:], strat_vals=cython.float[:,:,:], strat_disc_vals=cython.float[:,:,:], origin_idx=cython.int)
def cy_label_strategy(bop, lookahead, fleet_size, base_vals, distances, my_yard_ct, t_captured, t_first, x, y, strat_vals, strat_disc_vals, origin_idx):
    yard_count = cython.declare(cython.int, len(base_vals))
    dist = cython.declare(cython.int)
    t_delay = cython.declare(cython.int)
    idx = cython.declare(cython.int)
    i = cython.declare(cython.int)
    t = cython.declare(cython.int)
    x_temp = cython.declare(cython.int)
    y_temp = cython.declare(cython.int)
    found = cython.declare(cython.int)
    target_time = cython.declare(cython.int)
    dest_idx = cython.declare(cython.int)
    BOPnew = cython.declare(cython.int[:,:,:,:])
    BOPsq = cython.declare(cython.int[:,:])
    effect = cython.declare(cython.float[:])
    best_strat_vals = cython.declare(cython.float[:,:])
    new_cumsum = cython.declare(cython.float)
    sq_cumsum = cython.declare(cython.float)
    dist_mtx = cython.declare(cython.int[:,:])
    time_to_reach = cython.declare(cython.int)

    dist_mtx = np.zeros((yard_count, yard_count), dtype=np.intc)
    BOPsq = np.zeros((yard_count, lookahead+1), dtype=np.intc)
    BOPnew = np.zeros((yard_count, yard_count, lookahead+1, lookahead+1), dtype=np.intc)

    best_strat_vals = np.zeros((yard_count, lookahead + 1), dtype=np.single)

    for idx, x_temp, y_temp in zip(range(0, yard_count), x, y):
        BOPsq[idx, :] = bop[:, x_temp, y_temp]
        for dest_idx, x_dest, y_dest in zip(range(0, yard_count), x, y):
            time_to_reach = dist_mtx[origin_idx, dest_idx]
            dist = distances[Point(x_temp, y_temp)][Point(x_dest, y_dest)]
            dist_mtx[dest_idx, idx] = dist
            BOPnew[dest_idx, idx, None, :] = bop[:, x_temp, y_temp]
            for t_delay in range(0, lookahead + 1):
                if t_delay + time_to_reach + dist <= lookahead:
                    for i in range(t_delay+dist+time_to_reach, lookahead+1):
                        BOPnew[dest_idx, idx, t_delay, i] = BOPnew[dest_idx, idx, t_delay, i] + fleet_size
                else:
                    break

    # For each destination
    for dest_idx in range(0, yard_count):
        time_to_reach = dist_mtx[origin_idx, dest_idx]
        # For each home base
        for idx in range(0, my_yard_ct):
            # If it is captured, no additional value assigned (value 100% assigned to the captured base itself)
            if t_captured[idx] != lookahead+1:
                continue

            # Check to see if it is fully protected, in which case no value assigned
            for i in range(0, lookahead+1):
                if BOPsq[idx, i] < 0:
                    break # Continue with the loop if some entries in BOP sq are negative
            else:
                continue # Skip this loop if all positive

            dist = dist_mtx[dest_idx, idx]
            target_time = -1
            if t_captured[idx]+1-dist >= time_to_reach: # Can we even get there in time from the specified destination
                for t_delay in range(max(time_to_reach, t_first[dest_idx]), min(t_captured[dest_idx]+2, t_captured[idx]-dist+2, lookahead+1-dist)): # Amount of time to delay - looking for maximum
                    sq_cumsum = 0
                    new_cumsum = 0
                    #for t in range(t_captured[idx]-1, max(time_to_reach+dist, t_first[idx])-1, -1): # Find target time when all negatives flip to zero+
                    for t in range(min(t_captured[dest_idx], t_captured[idx]-1), t_first[idx]-1, -1): # We care about ALL flips, even prior flips. Showing up too late is useless
                        if BOPnew[dest_idx, idx, t_delay, t] < 0 and t >= t_delay + dist + time_to_reach:
                            new_cumsum = new_cumsum + BOPnew[dest_idx, idx, t_delay, t]
                        if BOPsq[idx, t] < 0:
                            sq_cumsum = sq_cumsum + BOPsq[idx, t]
                    if new_cumsum == 0: # All BOPnew are positive, record this t_delay as target
                        strat_vals[t_delay, x[dest_idx], y[dest_idx]] = strat_vals[t_delay, x[dest_idx], y[dest_idx]] + 0.50 * base_vals[idx]
                    else:
                        strat_vals[t_delay, x[dest_idx], y[dest_idx]] = strat_vals[t_delay, x[dest_idx], y[dest_idx]] + ((sq_cumsum - new_cumsum) / sq_cumsum) * 0.05 * base_vals[idx]

        # For each enemy base
        for idx in range(my_yard_ct, yard_count):
            if idx != dest_idx and t_captured[idx] < lookahead+1:
                continue
            if max(time_to_reach, t_first[dest_idx]) < lookahead+1-dist:
                for t_delay in range(max(time_to_reach, t_first[dest_idx]), lookahead+1-dist):
                    # assert (dest_idx < len(t_first))
                    if t_delay < lookahead + 1:
                        if max(time_to_reach, t_first[dest_idx]) < lookahead+1:
                            for t in range(max(time_to_reach, t_first[dest_idx]), lookahead+1):
                                # assert(dest_idx < yard_count)
                                # assert(idx < yard_count)
                                # assert(t_delay < lookahead+1)
                                # assert(t < lookahead+1)
                                if BOPnew[dest_idx, idx, t_delay, t] > 0:
                                    if BOPsq[idx, t] < 0:
                                        best_strat_vals[dest_idx, t] = max(best_strat_vals[dest_idx, t], base_vals[idx] * 0.50)

    best_strat_vals = np.flip(np.maximum.accumulate(np.flip(best_strat_vals, axis=1), axis=1), axis=1)
    for dest_idx in range(0, yard_count):
        for t in range(0, lookahead + 1):
            strat_vals[t, x[dest_idx], y[dest_idx]] = strat_vals[t, x[dest_idx], y[dest_idx]] + best_strat_vals[dest_idx, t]

    strat_max = cython.declare(cython.float)
    s_plus_d_max = cython.declare(cython.float)
    for dest_idx in range(0, yard_count):
        strat_max = 0.0
        s_plus_d_max = 0.0
        for t in range(strat_vals.shape[0] - 1, -1, -1): # Check backwards
            if t < t_first[dest_idx]:
                break
            strat_max = max(strat_max, strat_vals[t, x[dest_idx], y[dest_idx]])
            s_plus_d_max = max(s_plus_d_max, strat_vals[t, x[dest_idx], y[dest_idx]] + strat_disc_vals[t, x[dest_idx], y[dest_idx]])
            if strat_vals[t, x[dest_idx], y[dest_idx]] < strat_max:
                strat_vals[t, x[dest_idx], y[dest_idx]] = strat_max
            if strat_vals[t, x[dest_idx], y[dest_idx]] + strat_disc_vals[t, x[dest_idx], y[dest_idx]] < s_plus_d_max:
                strat_vals[t, x[dest_idx], y[dest_idx]] = s_plus_d_max - strat_disc_vals[t, x[dest_idx], y[dest_idx]]











