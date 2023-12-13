from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction
from .base_funcs.funcs import kore_sum, one_way_path, cpath_max_kore, mv_path, get_dist, ind2coord
from .base_funcs.spawn_ships import mk_ships
from .base_funcs.logger import log_it


def fill_fp7_nxt(persist, board, sy, fp_count: int):
    flp_effect = []
    max_kore_effect = 0.0
    max_path_len = persist['const'].Mine.max_fp7_len[min(len(persist['const'].Mine.max_fp7_len) - 1, persist['me_sy_cnt'])]
    for cmd, i_path in persist['fp7_sy'][sy.id]:
        # if persist['me_sy_cnt'] < 3 or (persist['me_sy_cnt'] < 5 and len(i_path) < 21) or len(i_path) < 19:
        if len(i_path) <= max_path_len:
            multi = 1.0
            if persist['opp_sy_pos']:
                for i, icoord in enumerate(i_path):
                    if any(get_dist(ind2coord(icoord), o_pos) <= i + 1 for o_pos in persist['opp_sy_pos']):
                        multi *= persist['const'].Mine.Risk_factor
            tmp_kore_effect = (kore_sum(board.observation['kore'], i_path) / (len(i_path) + 1)) * multi
            if max_kore_effect < tmp_kore_effect:
                max_kore_effect = tmp_kore_effect
            if 0.8 * max_kore_effect < tmp_kore_effect:
                flp_effect.append((tmp_kore_effect, cmd, i_path))
    flp_effect.sort(reverse=True)
    flp_effect = flp_effect[:fp_count]
    flp_effect = [fe for fe in flp_effect if fe[0] > 0.8 * max_kore_effect]
    flp_effect.reverse()
    persist['fp7_nxt'][sy.id] = flp_effect


def chk_collision(persist, fp7_nxt, my_coll_chk) -> int:
    collision = set()
    coll_size = 0
    for i, ind in enumerate(fp7_nxt[2]):
        i_coord = ind2coord(ind)
        if i + 1 < len(persist['gen_future']):
            for opp_pos in (of_pos for of_pos in persist['opp_fleets'][i + 1] if get_dist(i_coord, of_pos) < 2):
                of_id, of_amount = persist['gen_future'][i + 1].opp_fleets[opp_pos]
                if of_id not in collision:
                    collision.add(of_id)
                    coll_size += of_amount
            # no collisions with own ships
            # if my_coll_chk and i_coord in persist['gen_future'][i + 1].my_fleets.keys():
            if my_coll_chk and i_coord in persist['my_fleets'][i + 1]:
                return coll_size + 10000
                # coll_size += 10000
    return coll_size


def mk_fleets(persist):
    assert persist['my_fleets'] == []
    assert persist['opp_fleets'] == []
    for meta in persist['gen_future']:
        persist['my_fleets'].append(set([m_pos for m_pos in meta.my_fleets.keys()]))
        persist['opp_fleets'].append(set([m_pos for m_pos in meta.opp_fleets.keys()]))


def spawn_fp7(persist, board, sy, fl_size: int, fp_count: int, my_coll_chk: bool):
    assert 20 < fl_size
    if not persist['my_fleets']:
        mk_fleets(persist)
    coll_amount = 10000
    cnt = 0
    while fl_size <= coll_amount and cnt < fp_count:
        if sy.id not in persist['fp7_nxt'].keys() or not persist['fp7_nxt'][sy.id]:
            fill_fp7_nxt(persist, board, sy, fp_count)
        fp7_nxt = persist['fp7_nxt'][sy.id].pop()
        coll_amount = chk_collision(persist, fp7_nxt, my_coll_chk)
        cnt += 1
    if 0 < coll_amount < 10000:
        log_it(persist, 50, f"Spawn_fp7 {sy.position}: CollisionAmount {coll_amount}")
    if coll_amount < fl_size:
        def_fl_size = persist['me_ship_cnt'] // (persist['me_sy_cnt'] * persist['const'].Mine.Ships_per_shipyard)
        sy.next_action = ShipyardAction.launch_fleet_with_flight_plan(min(max(21, coll_amount + 1, def_fl_size), fl_size), fp7_nxt[1])
        persist['todo'].remove(sy)
    elif my_coll_chk:
        spawn_fp7(persist, board, sy, fl_size, fp_count, False)


def spawn_fp(persist, board, sy, flight_plan: str):
    assert flight_plan in ('fp5', 'fp6')
    if not persist['my_fleets']:
        mk_fleets(persist)
    if not persist['avail_ship_count'][sy.position]:
        return
    if max(persist['avail_ship_count'][sy.position]) < 21:
        t_cnt = 100
    else:
        t_cnt = min(i for i, ava in enumerate(persist['avail_ship_count'][sy.position]) if 20 < ava)
    if flight_plan == 'fp5':
        amount = 8
    else:
        amount = 13
    max_kore = 0.0
    m_cmd = ""
    for cmd, pat in persist[flight_plan].items():
        if len(pat) < t_cnt:
            k_sum = kore_sum(board.observation['kore'], mv_path(sy.position, pat))
            if max_kore < k_sum and chk_collision(persist, pat, True) < amount:
                max_kore = k_sum
                m_cmd = cmd
    if m_cmd:
        log_it(persist, 55, f"SpawnFp {flight_plan}: {t_cnt} {m_cmd}")
        sy.next_action = ShipyardAction.launch_fleet_with_flight_plan(amount, m_cmd)
        persist['todo'].remove(sy)
        if persist['builder'] and sy.id in persist['builder']:
            persist['Miner_pause'][sy.id] = 0
    else:
        mk_ships(persist, board, sy)


def spawn_fp3(persist, sy):
    sy.next_action = ShipyardAction.launch_fleet_with_flight_plan(3, persist['fp3'][persist['fp3_cnt']])
    persist['fp3_cnt'] = (persist['fp3_cnt'] + 1) % 2
    persist['todo'].remove(sy)


def ship_transfer(persist, board, source, target_pos, amount, save=False, maximize=True):
    if amount < 21:
        return
    cpaths = [p for p in one_way_path(source.position, target_pos)]
    cmd = cpath_max_kore(persist, board, source.position, cpaths, True, maximize)
    if cmd == "":
        if save:
            return
        cmd = cpath_max_kore(persist, board, source.position, cpaths, False)
    log_it(persist, 50, f"Transfer {amount} ships: {source.position} => {target_pos}")
    source.next_action = ShipyardAction.launch_fleet_with_flight_plan(amount, cmd)
    persist['todo'].remove(source)
