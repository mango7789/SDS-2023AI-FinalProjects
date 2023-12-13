from kaggle_environments.envs.kore_fleets.helpers import Board
from .base_funcs.funcs import get_dist, mv_path, cpath2coords, split_path
from .base_funcs.logger import log_it


# opponent fleet, that may be attacked
class OppFleet:

    def __init__(self, distance: int, sid: int, fid: str, step: int, position: tuple, ship_count: int, kore: float):
        self.distance = distance
        self.sid = sid
        self.fid = fid
        self.step = step
        self.position = position
        self.ship_count = ship_count
        self.kore = kore

    def __str__(self):
        return f"(distance: {self.distance},sid: {self.sid}, fid: {self.fid}, step: {self.step}," \
               f"coord: {self.position}, ship_count: {self.ship_count}, kore: {self.kore})"


# opponent fleet, that is identified as target
# will be stored in a dict with key sy.id
class OppTargetFleet:

    def __init__(self, launch_round: int, position: tuple, fid: int, sid: int):
        self.launch_round = launch_round  # round to launch the fleet from shipyard
        self.position = position          # position of the planned collision
        self.fid = fid                    # fleet.id of the target (opp.fleet)
        self.sid = sid                    # shipyard.id of the source (my shipyard)

    def __str__(self):
        return f"(launch_round: {self.launch_round}, position: {self.position}, fid: {self.fid})"


# Metainfo Future
class Metainfo:

    def __init__(self, rnd: int, my_kore: int, my_spawn: int, my_fleets: {}, opp_kore: int, opp_spawn: int, opp_fleets: {}):
        self.rnd = rnd
        self.my_kore = my_kore
        self.my_spawn = my_spawn
        self.my_fleets = my_fleets
        self.opp_kore = opp_kore
        self.opp_spawn = opp_spawn
        self.opp_fleets = opp_fleets

    def __str__(self):
        return f"(rnd: {self.rnd}, my_kore: {self.my_kore}, my_spawn: {self.my_spawn}, " \
               f"opp_kore: {self.opp_kore}, opp_spawn: {self.opp_spawn}, opp_fleets: {list(self.opp_fleets.keys())})"


# Shipyard Future
# will be stored in a dict with key = sy.position
class Sypos:

    def __init__(self, sid: str, rnd: int, amount: int, spawn: int):
        self.sid = sid          # sy.id at time self.rnd
        self.rnd = rnd          # round
        self.amount = amount    # sy.ship_count (negative + 1 for opponent )
        self.spawn = spawn      # sy.max_spawn

    def __str__(self):
        return f"(sid: {self.sid}, rnd: {self.rnd}, amount: {self.amount}, spawn: {self.spawn})"


# frontline
class Frontline:

    def __init__(self, me_pos: tuple, opp_pos: tuple, distance: int):
        self.me_pos = me_pos
        self.opp_pos = opp_pos
        self.distance = distance

    def __str__(self):
        return f"(me_pos: {self.me_pos}, opp_pos: {self.opp_pos}, distance: {self.distance})"


def init_persist(const) -> dict:
    persist = {'const': const,
               'fp3': ["N", "E"],
               'fp3_cnt': 0,
               'fp5': mk_flight_plan(5, const.Mine.fp5_max_ship_dist),
               'fp6': mk_flight_plan(6, const.Mine.fp6_max_ship_dist),
               'fp7': mk_flight_plan(7, const.Mine.fp7_max_ship_dist),
               'fp7_sy': {},
               'fp7_nxt': {},
               'sum_time': 0.0,
               'max_rnd_time': (0.0, 0),
               'impact': {},
               'fut_fls': [],
               'log_lvl': const.Logger.level,
               'log_f_name': const.Logger.f_name,
               'frontline': [],
               'Miner_pause': {},
               'under_attack': {},
               'no_shipyard_spawned': 0,
               'mk_shipyard': []
               }
    new_sy_area(persist)
    return persist


def update_persist(obs, config, persist, board):
    me = board.current_player.id
    opp = (me + 1) % 2
    persist['me']: int = me
    persist['opp']: int = opp
    persist['round'] = board.step
    persist['todo'] = [sy for sy in board.shipyards.values() if sy.player_id == me]
    persist['todo'].sort(key=lambda sy: sy.max_spawn)
    persist['kore'] = int(board.players[me].kore)
    persist['no_shipyard_spawned'] += 1
    prefix = str(board.step) + "-"
    p_len = len(prefix)
    persist['new_shipyard_ids'] = [sid for sid in board.shipyards.keys() if sid[:p_len] == prefix]
    set_sy_future(obs, config, me, persist['const'].PreCog.attack_detection, persist)
    persist['fp7_opp'] = []
    persist['my_fleets'] = []
    persist['opp_fleets'] = []
    persist['my_sy_pos'] = [sy.position for sy in board.players[me].shipyards]
    persist['opp_sy_pos'] = [sy.position for sy in board.players[opp].shipyards]
    persist['nxt_shipyard'] = get_nxt_shipyard(persist)
    persist['me_sy_cnt'] = len(board.players[me].shipyard_ids)
    persist['opp_sy_cnt'] = len(board.players[opp].shipyard_ids)
    persist['me_fl_cnt'] = len(board.players[me].fleet_ids)
    persist['me_ship_cnt'] = sum([sy.ship_count for sy in board.players[me].shipyards]) + \
                            sum([fl.ship_count for fl in board.players[me].fleets])
    persist['opp_ship_cnt'] = sum([sy.ship_count for sy in board.players[opp].shipyards]) + \
                            sum([fl.ship_count for fl in board.players[opp].fleets])

    persist['current_attack'] = []
    persist['no_attack_spawn'] = []
    for i in range(board.step - 2, board.step):
        if i in persist['under_attack'].keys():
            persist['current_attack'] += persist['under_attack'][i]
    persist['avail_ship_count'] = calc_available(persist)
    persist['min_avail_ship_count'] = {}
    for my_pos in persist['my_sy_pos']:
        persist['min_avail_ship_count'][my_pos] = min(persist['avail_ship_count'][my_pos])
    new_sy(persist, board)
    mk_builder(persist)
    persist['sy_border_cnt'] = sy_border_cnt(persist)  # does need frontline
    persist['fut_fls'] = [otf for otf in persist['fut_fls'] if board.step <= otf.launch_round and otf.fid in board.players[opp].fleet_ids
                          and otf.sid in board.players[me].shipyard_ids]
    to_del = [k for k in persist['fp7_sy'].keys() if k not in board.players[me].shipyard_ids]
    for k in to_del:
        del persist['fp7_sy'][k]

    if 65 <= persist['const'].Logger.level:
        for p in persist['avail_ship_count']:
            tmp = []
            for i in range(20):
                tmp.append(persist['sy_future'][p][i].amount)
            log_it(persist, 65, f"avail_ship_count: {p}: {tmp}")
    if 70 <= persist['const'].Logger.level and persist['frontline']:
        mesg = "Frontline: "
        for fl in persist['frontline']:
            mesg += f" {fl}"
        log_it(persist, 70, mesg)


#### internal functions ######

def sy_border_cnt(persist):
    if persist['frontline']:
        min_front = persist['frontline'][0].distance
    else:
        min_front = 20
    conf_len = len(persist['const'].NewShipyard.Ship_vs_shipyard)
    s_vs_sy = persist['const'].NewShipyard.Ship_vs_shipyard[min(conf_len - 1, persist['me_sy_cnt'] - 1)]
    sbc = persist['const'].NewShipyard.Ship_addon + (persist['me_sy_cnt'] * (s_vs_sy - min_front))
    if persist['me_sy_cnt'] == 1 and 130 < persist['round']:
        sbc -= 10
    return sbc


def mk_flight_plan(num: int, max_dist: int) -> dict:
    assert num in (5, 6, 7)
    fp = {}
    if num == 5:
        plans = mk_fp5(max_dist)
    elif num == 6:
        plans = mk_fp6(max_dist)
    else:
        plans = mk_fp7(max_dist)
    for p in plans:
        fp[''.join(p)] = cpath2coords((0, 0), p, [], [])
    return fp


def mk_fp5(max_dist: int) -> list:
    possible = []
    pos = ("NExSW", "NWxSE", "ENxWS", "ESxWN", "SExNW", "SWxNE", "WNxES", "WSxEN")
    for stub in pos:
        for i in range(1, max_dist):
            possible.append(split_path(stub.replace("x", str(i))))
    return possible


def mk_fp6(max_dist: int) -> list:
    possible = []
    for stub in ("NExWxS", "NWxExS", "ENxSxW", "ESxNxW", "SExWxN", "SWxExN", "WNxSxE", "WSxNxE"):
        for i in range(1, max_dist):
            possible.append(split_path(stub.replace("x", str(i))))
    return possible


def mk_fp7(max_dist: int) -> list:
    possible = []
    for stub in ("NaEbWbS", "NaWbEbS", "EaSbNbW", "EaNbSbW", "SaWbEbN", "SaEbWbN", "WaNbSbE", "WaSbNbE", "NaEbSaW", "EaSbWaN", "SaWbNaE", "WaNbEaS"):
        for i in range(0, min(10, max_dist)):
            for j in range(0, min(10, max_dist - i)):
                possible.append(split_path(stub.replace("a", str(i)).replace("b", str(j)).replace("0", "")))
    return possible


def new_sy(persist, board):
    if persist['new_shipyard_ids']:
        log_it(persist, 45, f"New shipyard found.")
        persist['frontline'] = []
        min_dist = 21
        for my_sy in board.players[persist['me']].shipyards:
            for opp_sy in board.players[persist['opp']].shipyards:
                tmp_dist = get_dist(my_sy.position, opp_sy.position)
                if tmp_dist < min_dist:
                    min_dist = tmp_dist
                    persist['frontline'] = [Frontline(me_pos=my_sy.position, opp_pos=opp_sy.position, distance=tmp_dist)]
                elif tmp_dist == min_dist:
                    persist['frontline'].append(Frontline(me_pos=my_sy.position, opp_pos=opp_sy.position, distance=tmp_dist))
    for sid in persist['new_shipyard_ids']:
        if sid in board.players[persist['me']].shipyard_ids:
            sy = board.shipyards[sid]
            persist['fp7_sy'][sid] = []
            for cmd, pat in persist['fp7'].items():
                persist['fp7_sy'][sid].append((cmd, mv_path(sy.position, pat)))


def get_nxt_shipyard(persist) -> dict:

    def work_on(a: list, b: list):
        for my_sy_pos in a:
            min_dist = 22
            for opp_sy_pos in b:
                tmp_dist = get_dist(my_sy_pos, opp_sy_pos)
                if tmp_dist < min_dist:
                    min_dist = tmp_dist
                    nxt_shipyard[my_sy_pos] = [opp_sy_pos]
                elif tmp_dist == min_dist:
                    nxt_shipyard[my_sy_pos].append(opp_sy_pos)

    nxt_shipyard = {}
    if persist['my_sy_pos'] and persist['opp_sy_pos']:
        work_on(persist['my_sy_pos'], persist['opp_sy_pos'])
        work_on(persist['opp_sy_pos'], persist['my_sy_pos'])
    return nxt_shipyard


def calc_available(persist) -> dict:

    def mk_avail(pos_lst: list, is_me: bool):
        multi = {True: 1, False: -1}
        for tmp_pos in pos_lst:
            addon = 0
            spent_kore = 0.0
            pos_sc_future = []
            for i, pos_sc in enumerate(persist['sy_future'][tmp_pos]):
                forcast = pos_sc.amount + (multi[is_me] * addon)
                pos_sc_future.append(forcast)
                if is_me and forcast < 0 and tmp_pos not in [sin[0] for sin in persist['sy_in_need']]:
                    persist['sy_in_need'].append((tmp_pos, pos_sc.rnd, forcast))
                gen_fut = persist['gen_future'][i]
                if is_me and i == 0:
                    spent_kore += (gen_fut.my_spawn - pos_sc.spawn) * 10
                else:
                    if is_me:
                        if gen_fut.my_spawn <= (gen_fut.my_kore - spent_kore) // 10:
                            addon += pos_sc.spawn
                            spent_kore += gen_fut.my_spawn * 10
                    else:
                        if gen_fut.opp_spawn <= (gen_fut.opp_kore - spent_kore) // 10:
                            addon += pos_sc.spawn
                            spent_kore += gen_fut.opp_spawn * 10
            avail_ship_count[tmp_pos] = pos_sc_future

    avail_ship_count = {}
    persist['sy_in_need'] = []
    mk_avail(persist['my_sy_pos'], True)
    mk_avail(persist['opp_sy_pos'], False)
    return avail_ship_count


def mk_builder(persist):
    persist['builder'] = []
    if 4 < len(persist['todo']):
        persist['builder'] = [sy.id for sy in persist['todo'][-2:]]
    elif 2 < len(persist['todo']):
        persist['builder'] = [sy.id for sy in persist['todo'][-1:]]
    elif 2 == len(persist['todo']):
        if persist['todo'][0].max_spawn <= persist['todo'][1].max_spawn // 2:
            persist['builder'] = [sy.id for sy in persist['todo'][-1:]]
    for key in persist['Miner_pause'].keys():
        persist['Miner_pause'][key] += 1
    for sid in persist['builder']:
        if sid not in persist['Miner_pause'].keys():
            persist['Miner_pause'][sid] = 0


def set_sy_future(obs, config, me: int, depth: int, persist: dict):
    n_board = Board(obs, config)
    opp = (me + 1) % 2
    base_round = n_board.step
    persist['sy_future'] = {}
    persist['gen_future'] = []
    ffls = {}
    for i in range(depth):
        my_spawn = 0
        opp_spawn = 0
        curr_step = base_round + i
        for nsy in n_board.shipyards.values():
            if nsy.player_id == me:
                n_amount = nsy.ship_count
                my_spawn += nsy.max_spawn
            else:
                n_amount = -(1 + nsy.ship_count)
                opp_spawn += nsy.max_spawn
            if nsy.position in persist['sy_future'].keys():
                persist['sy_future'][nsy.position].append(Sypos(sid=nsy.id, rnd=curr_step, amount=n_amount, spawn=nsy.max_spawn))
            else:
                persist['sy_future'][nsy.position] = [Sypos(sid=nsy.id, rnd=curr_step, amount=n_amount, spawn=nsy.max_spawn)]
        opp_fleets = {}
        me_fleets = {}
        for fl in n_board.fleets.values():
            if fl.player_id == persist['opp']:
                opp_fleets[fl.position] = (fl.id, fl.ship_count)
            else:
                me_fleets[fl.position] = (fl.id, fl.ship_count)
        persist['gen_future'].append(Metainfo(rnd=(base_round + i),
                                              my_kore=n_board.players[me].kore,
                                              my_spawn=my_spawn,
                                              my_fleets=me_fleets,
                                              opp_kore=n_board.players[(me + 1) % 2].kore,
                                              opp_spawn=opp_spawn,
                                              opp_fleets=opp_fleets))
        if 0 < i:
            ffls[curr_step] = []
            opp_fleet = {}
            for fl in n_board.players[opp].fleets:
                opp_fleet[fl.position] = fl.id
            for sy in n_board.players[me].shipyards:
                syp = sy.position
                for flp in opp_fleet.keys():
                    # if [ff for ff in persist['fut_fls'] if ff.sid == sy.id and ff.fid == opp_fleet[flp]]:
                    if [ff for ff in persist['fut_fls'] if ff.fid == opp_fleet[flp]]:
                        pass
                    else:
                        sf_dist = get_dist(syp, flp)
                        if 0 < sf_dist < i:
                            ffl = OppFleet(distance=sf_dist,
                                           sid=sy.id,
                                           fid=opp_fleet[flp],
                                           step=curr_step,
                                           position=flp,
                                           ship_count=n_board.fleets[opp_fleet[flp]].ship_count,
                                           kore=n_board.fleets[opp_fleet[flp]].kore)
                            ffls[curr_step].append(ffl)
                            if curr_step - 2 in ffls.keys() and (curr_step - 1 in ffls.keys()) \
                                    and [fut for fut in ffls[curr_step - 1] if fut.distance < ffl.distance and fut.sid == ffl.sid and fut.fid == ffl.fid] \
                                    and [fut for fut in ffls[curr_step - 2] if fut.distance == ffl.distance and fut.sid == ffl.sid and fut.fid == ffl.fid]:
                                target = [fut for fut in ffls[curr_step - 1] if fut.sid == ffl.sid and fut.fid == ffl.fid][0]
                                f_pos = target.position
                                for ndir in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                    n_flp = ((ndir[0] + target.position[0]) % 21, (ndir[1] + target.position[1]) % 21)
                                    if 0 < get_dist(syp, n_flp) < sf_dist:
                                        f_pos = n_flp
                                        sf_dist = get_dist(syp, n_flp)
                                persist['fut_fls'].append(OppTargetFleet(launch_round=target.step - sf_dist,
                                                                        position=f_pos,
                                                                        fid=target.fid,
                                                                        sid=sy.id))

        n_board = n_board.next()


def new_sy_area(persist):
    coord_shift = []
    multplier = []
    for x in range(persist['const'].NewShipyard.New_chk_dist):
        for y in range(persist['const'].NewShipyard.New_chk_dist + 1 - x):
            if 0 < x + y:
                mult = pow(persist['const'].NewShipyard.New_chk_decay, x + y)
                coord_shift.append((x, y))
                multplier.append(mult)
                if y == 0:
                    coord_shift.append((-x, y))
                    multplier.append(mult)
                elif x == 0:
                    coord_shift.append((x, -y))
                    multplier.append(mult)
                else:
                    coord_shift.append((x, -y))
                    multplier.append(mult)
                    coord_shift.append((-x, -y))
                    multplier.append(mult)
                    coord_shift.append((-x, y))
                    multplier.append(mult)
    persist['nsy_coords'] = coord_shift
    persist['nsy_multiplier'] = multplier
