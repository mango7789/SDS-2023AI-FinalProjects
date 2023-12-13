def coord2int(coord: tuple) -> int:
    return coord[0] + (20 - coord[1]) * 21


def ind2coord(ind: int) -> tuple:
    return ind % 21, 20 - (ind // 21)


def get_dist_coord(coord: tuple, num: int) -> list:
    res = []
    for i in range(1, num):
        res.append(((coord[0] + i) % 21, (coord[1] + num - i) % 21))
        res.append(((coord[0] + i) % 21, (coord[1] - num + i) % 21))
        res.append(((coord[0] - i) % 21, (coord[1] + num - i) % 21))
        res.append(((coord[0] - i) % 21, (coord[1] - num + i) % 21))
    return res


def mdiff(a: int, b: int) -> int:
    return min(abs(a - b), abs(a - b - 21), abs(a - b + 21))


def get_dist(c1: tuple, c2: tuple) -> int:
    return mdiff(c1[0], c2[0]) + mdiff(c1[1], c2[1])


def not_zero(s: str) -> bool:
    return s != "0" and s != ""


def path_components(c1: tuple, c2: tuple) -> tuple:
    assert c1 != c2
    veri = mdiff(c1[0], c2[0])
    hori = mdiff(c1[1], c2[1])
    if (c1[0] + veri) % 21 == c2[0]:
        veri_dir = "E"
    else:
        veri_dir = "W"
    if (c1[1] + hori) % 21 == c2[1]:
        hori_dir = "N"
    else:
        hori_dir = "S"

    if veri == 0:
        return "", 0, hori_dir, hori
    if hori == 0:
        return veri_dir, veri, "", 0
    return veri_dir, veri, hori_dir, hori


def one_way_path(c1: tuple, c2: tuple) -> list:
    # will create all possible one way path ( max length = 6 )
    res = []
    if c1 == c2:
        return res
    veri_dir, veri, hori_dir, hori = path_components(c1, c2)

    if veri == 0:
        return [[hori_dir]]
    if hori == 0:
        return [[veri_dir]]

    for i in range(veri - 1):
        res.append(list(filter(not_zero, [veri_dir, str(i), hori_dir, str(hori - 1), veri_dir, str(veri - 2 - i)])))
    res.append(list(filter(not_zero, [veri_dir, str(veri - 1), hori_dir, str(hori - 1)])))
    for i in range(hori - 1):
        res.append(list(filter(not_zero, [hori_dir, str(i), veri_dir, str(veri - 1), hori_dir, str(hori - 2 - i)])))
    res.append(list(filter(not_zero, [hori_dir, str(hori - 1), veri_dir, str(veri - 1)])))
    return res


def two_way_path(c1: tuple, c2: tuple) -> list:
    opp_way = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
    res = []
    if c1 == c2:
        return res
    veri_dir, veri, hori_dir, hori = path_components(c1, c2)

    if veri == 0:
        return [[hori_dir, str(hori - 1), opp_way[hori_dir]]]
    if hori == 0:
        return [[veri_dir, str(veri - 1), opp_way[veri_dir]]]
    res.append(list(filter(not_zero, [hori_dir, str(hori - 1), veri_dir, str(veri - 1), opp_way[hori_dir], str(hori - 1), opp_way[veri_dir]])))
    res.append(list(filter(not_zero, [veri_dir, str(veri - 1), hori_dir, str(hori - 1), opp_way[veri_dir], str(veri - 1), opp_way[hori_dir]])))
    return res


def add_coord(x, y, dir_coord, coords):
    x = (x + dir_coord[0]) % 21
    y = (y + dir_coord[1]) % 21
    coords.append((x, y))
    return x, y, coords


def cpath2coords(start: tuple, cpath: list, my_sy_pos: list, opp_sy_pos: list) -> list:
    dirs = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}
    (x, y) = start
    dir_coord = (0, 0)
    coords = [(x, y)]
    for c in cpath:
        if "NESW".__contains__(c):
            dir_coord = dirs[c]
            (x, y, coords) = add_coord(x, y, dir_coord, coords)
            if (x, y) in my_sy_pos or (x, y) in opp_sy_pos:
                return coords[1:]
        elif c == "C":
            return coords[1:]
        else:
            for _ in range(int(c)):
                (x, y, coords) = add_coord(x, y, dir_coord, coords)
                if (x, y) in my_sy_pos or (x, y) in opp_sy_pos:
                    return coords[1:]
    cnt = 0
    while (x, y) != coords[0] and cnt < 22:
        (x, y, coords) = add_coord(x, y, dir_coord, coords)
        if (x, y) in my_sy_pos or (x, y) in opp_sy_pos:
            return coords[1:]
        cnt += 1
    if coords[0] != coords[-1]:
        #print("Found non-ending kpath:", kpath)
        return coords[1:]
    return coords[1:-1]


def kore_sum(kore: list, k_indexes: list) -> float:
    return sum(kore[c] for c in k_indexes)


def split_path(path: str) -> list:
    res = []
    tmp = ""
    for c in path:
        if "NESWC".__contains__(c):
            if tmp:
                res.append(tmp)
                tmp = ""
            res.append(c)
        else:
            tmp += c
    if tmp:
        res.append(tmp)
    return res


def cpath_max_kore(persist: dict, board, strt: tuple, cpaths: list, collision_detect=False, maximize=True):
    min_kore = 1000000.0
    max_kore = -1.0
    m_cmd = ""
    for cpath in cpaths:
        collision = False
        coords = cpath2coords(strt, cpath, persist['my_sy_pos'], persist['opp_sy_pos'])
        if maximize:
            if collision_detect:
                for i, coord in enumerate(coords):
                    if i + 1 < len(persist['gen_future']) and \
                        any(get_dist(coord, of_pos) < 2 for of_pos in persist['gen_future'][i + 1].opp_fleets.keys()):
                        #print(f"cpath_max_kore: Collision detected: Round {board.step + i}: {coord} in {coords}")
                        collision = True
            if not (collision_detect and collision):
                k_sum = kore_sum(board.observation['kore'], [coord2int(c) for c in coords])
                if max_kore < k_sum:
                    max_kore = k_sum
                    m_cmd = ''.join(cpath)
        else:
            k_sum = kore_sum(board.observation['kore'], [coord2int(c) for c in coords])
            if k_sum < min_kore:
                min_kore = k_sum
                m_cmd = ''.join(cpath)
    return m_cmd


def mv_path(start: tuple, path: list) -> list:
    return list([coord2int(((start[0] + p[0]) % 21, (start[1] + p[1]) % 21)) for p in path])


def get_threshold(persist, board, my_sy, opp_sy_pos):
    distance = get_dist(my_sy.position, opp_sy_pos)
    if opp_sy_pos in persist['opp_sy_pos']:
        thresh = - persist['avail_ship_count'][opp_sy_pos][distance + 1]
    else:
        o_sypos = persist['sy_future'][opp_sy_pos][0]
        thresh = - o_sypos.amount + (board.step + distance - o_sypos.rnd)
    for surround_sy in [sy for sy in board.players[persist['opp']].shipyards if sy.position != opp_sy_pos]:
        surr_dist = get_dist(surround_sy.position, opp_sy_pos)
        if surr_dist < distance:
            thresh -= persist['avail_ship_count'][surround_sy.position][distance - surr_dist + 1]
    return thresh


def get_nearest_pos(my_pos: tuple, o_pos_lst: list):
    assert o_pos_lst
    min_dist = 50
    n_pos = (21, 21)
    for o_pos in o_pos_lst:
        tmp_dist = get_dist(my_pos, o_pos)
        if tmp_dist < min_dist:
            min_dist = tmp_dist
            n_pos = o_pos
    return min_dist, n_pos
