from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from .base_funcs.funcs import get_dist_coord, one_way_path, get_dist, cpath_max_kore, mv_path
from .base_funcs.logger import log_it


def spawn_sy(persist, board, sy):
    avail_ship_count = min(persist['avail_ship_count'][sy.position])
    n_coord = (21, 21)
    max_res = 0.0
    if persist['me_sy_cnt'] == 1 and persist['opp_sy_cnt'] == 1:
        nsy_dist = persist['const'].NewShipyard.New_dist_first
        pot_nsy = get_dist_coord(sy.position, nsy_dist)
    else:
        nsy_dist = persist['const'].NewShipyard.New_dist
        pot_nsy = get_dist_coord(sy.position, nsy_dist)
        if persist['my_sy_pos']:
            pot_nsy = [c for c in pot_nsy if persist['const'].NewShipyard.Min_dist_me <= min(get_dist(c, syp) for syp in persist['my_sy_pos'])]
        if persist['opp_sy_pos']:
            pot_nsy = [c for c in pot_nsy if persist['const'].NewShipyard.Min_dist_opp <= min(get_dist(c, syp) for syp in persist['opp_sy_pos'])]
    if pot_nsy:
        for pn in pot_nsy:
            tmp_res = sum(board.observation['kore'][ind] * persist['nsy_multiplier'][i] for i, ind in enumerate(mv_path(pn, persist['nsy_coords'])))
            if max_res < tmp_res:
                n_coord = pn
                max_res = tmp_res
        cpaths = [p + ["C"] for p in one_way_path(sy.position, n_coord)]
        log_it(persist, 45, f"Best place New Shipyard: {sy.position} => {n_coord} {persist['my_sy_pos']}")
        cmd = cpath_max_kore(persist, board, sy.position, cpaths)
        if 0 < len(cmd):
            sy.next_action = ShipyardAction.launch_fleet_with_flight_plan(avail_ship_count, cmd)
            persist['no_shipyard_spawned'] = 0
            return
    log_it(persist, 30, f"No place for new Shipyard found: {sy.position} {persist['my_sy_pos']}")
