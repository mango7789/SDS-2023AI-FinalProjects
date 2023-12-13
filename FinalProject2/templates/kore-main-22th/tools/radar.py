from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from .base_funcs.funcs import get_dist, two_way_path, cpath_max_kore, get_nearest_pos
from .base_funcs.logger import log_it
from .base_funcs.spawn_ships import mk_ships


def fleet_combat(persist, board, sy):
    avail_ship_count = persist['min_avail_ship_count'][sy.position]
    targets = [t for t in persist['fut_fls'] if t.sid == sy.id]
    if persist['opp_sy_pos']:
        targets = [t for t in targets if get_nearest_pos(t.position, persist['opp_sy_pos'])[0] < get_dist(t.position, sy.position) + 1]
    go_on = True
    for target in targets:
        if board.step == target.launch_round and go_on:
            if board.fleets[target.fid].ship_count < avail_ship_count:
                log_it(persist, 45, f"Fleet combat {get_dist(sy.position, target.position)}: {sy.position} {avail_ship_count} => {target.position}")
                cmd = cpath_max_kore(persist, board, sy.position, two_way_path(sy.position, target.position))
                if 20 < avail_ship_count:
                    if 0 == len(cmd):
                        log_it(persist, 10, f"Radar error cmd length 0: {sy.position} {target.position}")
                    else:
                        sy.next_action = ShipyardAction.launch_fleet_with_flight_plan(max(21, board.fleets[target.fid].ship_count + 1), cmd)
                        persist['todo'].remove(sy)
                        go_on = False
    for target in targets:
        if board.step + 1 == target.launch_round and go_on:
            if board.fleets[target.fid].ship_count < avail_ship_count + sy.max_spawn:
                log_it(persist, 50, f"Fleet combat waiting for: {sy.position} => {target.position}")
                mk_ships(persist, board, sy)
                go_on = False
