from .base_funcs.logger import log_it
from .base_funcs.funcs import one_way_path, get_threshold, get_dist, get_nearest_pos
from .flight_plan import spawn_fp3, spawn_fp, spawn_fp7, ship_transfer
from .base_funcs.spawn_ships import mk_ships
from .new_shipyard import spawn_sy
from .radar import fleet_combat


def weak_opponent(persist, board, sy):
    avail_ship_count = persist['min_avail_ship_count'][sy.position]
    if 0 < persist['gen_future'][0].opp_spawn:
        mod_asc = avail_ship_count * max(1.0, persist['gen_future'][0].my_spawn / persist['gen_future'][0].opp_spawn)
    else:
        return None, None
    if avail_ship_count < 21:
        return None, None

    opps_weak = [(pos, persist['sy_future'][pos][0].spawn) for pos in persist['sy_future'].keys()
                 if persist['sy_future'][pos][0].amount < 0]
    min_sc = mod_asc
    opp_spawn = 0
    for (o_pos, o_spawn) in opps_weak:
        tmp_sc = get_threshold(persist, board, sy, o_pos)
        if tmp_sc < avail_ship_count and opp_spawn < o_spawn:
            weak = o_pos
            min_sc = tmp_sc
            opp_spawn = o_spawn
    if min_sc < mod_asc:
        log_it(persist, 45, f"Weakest Opponent {sy.position} {avail_ship_count} => {weak} {min_sc}")
        return weak, min_sc
    return None, None


def attack(persist, board):
    todo_lst = persist['todo'].copy()
    for sy in todo_lst:
        if sy.id not in persist['no_attack_spawn']:
            avail_ship_count = persist['min_avail_ship_count'][sy.position]
            (target_pos, min_sc) = weak_opponent(persist, board, sy)
            if target_pos:
                cpaths = [p for p in one_way_path(sy.position, target_pos)]
                if cpaths:
                    log_it(persist, 50, f"attack: {sy.position} => {target_pos}")
                    safety = min_sc < avail_ship_count
                    ship_transfer(persist, board, sy, target_pos,  max(21, min(avail_ship_count, 2 * min_sc)), safety, safety)
                    if sy not in persist['todo']:
                        if board.step in persist['under_attack'].keys():
                            persist['under_attack'][board.step].append((sy.position, target_pos))
                        else:
                            persist['under_attack'][board.step] = [(sy.position, target_pos)]
                    else:
                        log_it(persist, 25, f"attack: No save path found {sy.position} => {target_pos}")
                else:
                    log_it(persist, 25, f"attack: No path found {sy.position} => {target_pos}")


def ship_in_need(persist, board):
    if persist['sy_in_need']:
        log_it(persist, 50, f"Shipyards in need: {persist['sy_in_need']} {[s.position for s in persist['todo']]}")
        for shipyard in [sy for sy in persist['todo'] if sy.position in [sin[0] for sin in persist['sy_in_need']]]:
            mk_ships(persist, board, shipyard)
        todo_lst = persist['todo'].copy()
        for shipyard in todo_lst:
            avail_ship_count = persist['min_avail_ship_count'][shipyard.position]
            if shipyard.position not in [sin[0] for sin in persist['sy_in_need']]:
                for (sin_pos, sin_rnd, sin_need) in persist['sy_in_need']:
                    if board.step + get_dist(shipyard.position, sin_pos) == sin_rnd and \
                            20 < avail_ship_count and \
                            shipyard in persist['todo']:
                        ship_transfer(persist, board, shipyard, sin_pos, avail_ship_count, False)
                if shipyard in persist['todo']:
                    if any(board.step + get_dist(shipyard.position, sin_pos) < sin_rnd for sin_pos, sin_rnd, sin_need in persist['sy_in_need']):
                        mk_ships(persist, board, shipyard)
                        if shipyard in persist['todo']:     # if no ship created, still disable any other action
                            persist['todo'].remove(shipyard)
            else:
                mk_ships(persist, board, shipyard)


def spawn_miner(persist, board):
    todo_lst = persist['todo'].copy()
    for shipyard in todo_lst:
        avail_ship_count = persist['min_avail_ship_count'][shipyard.position]
        # log_it(persist, 10, f"  SpawnMiner {shipyard.position}: avail = {avail_ship_count} ({persist['kore']}, {persist['Miner_pause']})")
        is_builder = persist['builder'] and shipyard.id in persist['builder']
        if is_builder and \
                persist['Miner_pause'][shipyard.id] <= persist['const'].BuildShip.Miner_pause_for_Builder and \
                shipyard.max_spawn * 10 < persist['kore']:
            log_it(persist, 55, f"Pausing Miner {shipyard.position}: {persist['Miner_pause'][shipyard.id]}")
        else:
            if 21 <= avail_ship_count:
                # log_it(persist, 10, f"    SpawnMiner: start spawn_fp7 {avail_ship_count}")
                spawn_fp7(persist, board, shipyard, avail_ship_count, persist['const'].Mine.fp7_plan_count, persist['round'] < persist['const'].Mine.Strict_mining)
                if is_builder:
                    persist['Miner_pause'][shipyard.id] = 0
            elif 13 <= avail_ship_count:
                spawn_fp(persist, board, shipyard, 'fp6')
            elif 8 <= avail_ship_count:
                spawn_fp(persist, board, shipyard, 'fp5')


def spawn_shipyard(persist, board):
    tmp_mk_sy = []
    todo_lst = persist['todo'].copy()
    for sy in todo_lst:
        if sy.id not in persist['no_attack_spawn']:
            if persist['me_sy_cnt'] < persist['const'].NewShipyard.Max_shipyard and \
                    persist['const'].NewShipyard.Round_dist < persist['no_shipyard_spawned'] and \
                    persist['sy_border_cnt'] + (persist['opp_ship_cnt'] - persist['me_ship_cnt']) // persist['const'].NewShipyard.Opp_vs_me_factor < persist['me_ship_cnt']:
                log_it(persist, 55, f"SpawnShipyard {sy.position}: {persist['sy_border_cnt']} + {(persist['opp_ship_cnt'] - persist['me_ship_cnt']) // persist['const'].NewShipyard.Opp_vs_me_factor} < {persist['me_ship_cnt']}")
                if persist['const'].NewShipyard.New_sy_threshold <= persist['min_avail_ship_count'][sy.position]:
                    if sy.position not in persist['mk_shipyard'] and \
                            sy.max_spawn * 10 < persist['kore']:
                        log_it(persist, 50, f"SpawnShipyard {sy.position}: Wait one more round")
                        mk_ships(persist, board, sy)
                        tmp_mk_sy.append(sy.position)
                    else:
                        spawn_sy(persist, board, sy)
                        persist['todo'].remove(sy)
                elif persist['const'].NewShipyard.New_sy_threshold <= min(persist['avail_ship_count'][sy.position][1:]) + min(persist['kore'] // 10, sy.max_spawn):
                    log_it(persist, 50, f"SpawnShipyard {sy.position}: Wait for next move")
                    mk_ships(persist, board, sy)
    persist['mk_shipyard'] = tmp_mk_sy


def ship_shortage(persist, board, sy):
    if 20 < persist['min_avail_ship_count'][sy.position]:
        spawn_fp7(persist, board, sy, persist['min_avail_ship_count'][sy.position], persist['const'].Mine.fp7_plan_count, persist['round'] < persist['const'].Mine.Strict_mining)
    elif 13 <= persist['min_avail_ship_count'][sy.position]:
        spawn_fp(persist, board, sy, 'fp6')
    elif 8 <= persist['min_avail_ship_count'][sy.position]:
        spawn_fp(persist, board, sy, 'fp5')
    elif 3 <= persist['min_avail_ship_count'][sy.position] and board.current_player.kore < 10:
        spawn_fp3(persist, sy)
    elif 10 <= board.current_player.kore:
        mk_ships(persist, board, sy)
    else:
        log_it(persist, 40, f"Round {board.step}: cannot do anything")


def low_ship_count(persist, board):
    if persist['me_fl_cnt'] <= persist['me_sy_cnt']:
        todo_lst = persist['todo'].copy()
        for shipyard in todo_lst:
            if shipyard.id in [t.sid for t in persist['fut_fls']]:  # Fleet combat
                fleet_combat(persist, board, shipyard)
            if shipyard in persist['todo']:
                ship_shortage(persist, board, shipyard)


def launch_fleet_combat(persist, board):
    todo_lst = persist['todo'].copy()
    for shipyard in todo_lst:
        if shipyard.id in [t.sid for t in persist['fut_fls']]:  # Fleet combat
            fleet_combat(persist, board, shipyard)


def planned_shipyards(persist, board):
    new_opp = [pos for pos in persist['sy_future'].keys()
               if board.step < persist['sy_future'][pos][0].rnd and persist['sy_future'][pos][0].amount < 0]
    if new_opp:
        me_dist = []
        todo_lst = persist['todo'].copy()
        for sy in todo_lst:
            me_dist.append((min([get_dist(sy.position, no) for no in new_opp]), sy))
        me_dist.sort(key=lambda md: md[0])
        opp_wait = persist['const'].NewShipyard.Opp_wait[min(len(persist['const'].NewShipyard.Opp_wait), persist['me_sy_cnt']) - 1]
        for i in range(min(opp_wait, len(me_dist))):
            log_it(persist, 55, f"PlannedShipyard wait {i}: {me_dist[i][1].position}")
            mk_ships(persist, board, me_dist[i][1])
            persist['no_attack_spawn'].append(me_dist[i][1].id)


def mv2frontline(persist, board):
    todo_lst = persist['todo'].copy()
    for sy in todo_lst:
        avail_ship_count = persist['min_avail_ship_count'][sy.position] - persist['const'].BuildShip.M2f_base
        if persist['const'].BuildShip.M2f_threshold < avail_ship_count:
            target_lst = mk_alarm(persist, board)
            if not target_lst and persist['frontline']:
                target_lst = [front.me_pos for front in persist['frontline']]
            if target_lst:
                t_dist, target = get_nearest_pos(sy.position, target_lst)
                if target != sy.position:
                    log_it(persist, 50, f"Mv2Frontline: Frontline-Transfer {sy.position} {avail_ship_count} => {target}")
                    ship_transfer(persist, board, sy, target, avail_ship_count, False)
            else:
                log_it(persist, 10, f"Mv2Frontline issue: no Frontline found")


def mk_alarm(persist, board):
    spos_alarm = []
    for sy in board.players[persist['me']].shipyards:
        if any(persist['avail_ship_count'][sy.position][mo_dist] + sy.max_spawn * mo_dist < o_sc for
               o_pos, o_sc, mo_dist in [(opp.position, opp.ship_count, get_dist(opp.position, sy.position)) for opp in board.shipyards.values() if opp.player_id == persist['opp']]):
            spos_alarm.append(sy.position)
    return spos_alarm


def create_ships(persist, board, mk_all: bool):
    todo_lst = persist['todo'].copy()
    for shipyard in todo_lst:
        if shipyard.max_spawn == 1 or mk_all:
            mk_ships(persist, board, shipyard)


def create_ships_new(persist, board):
    create_ships(persist, board, False)


def create_ships_all(persist, board):
    create_ships(persist, board, True)
