def Agent_c4_1(obs, config):
    import random
    import numpy as np

    WIDTH, N_STEPS = 2, 12

    map4q = '''
        30,31,32,33; 31,32,33,34; 32,33,34,35; 33,34,35,36; 
        40,41,42,43; 41,42,43,44; 42,43,44,45; 43,44,45,46; 
        50,51,52,53; 51,52,53,54; 52,53,54,55; 53,54,55,56; 
        04,14,24,34; 14,24,34,44; 24,34,44,54; 
        05,15,25,35; 15,25,35,45; 25,35,45,55; 
        06,16,26,36; 16,26,36,46; 26,36,46,56; 
        03,12,21,30; 
        04,13,22,31; 13,22,31,40; 
        05,14,23,32; 14,23,32,41; 23,32,41,50; 
        06,15,24,33; 15,24,33,42; 24,33,42,51; 
        16,25,34,43; 25,34,43,52; 
        26,35,44,53; 
        03,13,23,33; 13,23,33,43; 23,33,43,53; 
        20,31,42,53; 
        10,21,32,43; 21,32,43,54; 
        00,11,22,33; 11,22,33,44; 22,33,44,55; 
        01,12,23,34; 12,23,34,45; 23,34,45,56; 
        02,13,24,35; 13,24,35,46; 
        03,14,25,36; 
        00,10,20,30; 10,20,30,40; 20,30,40,50; 
        01,11,21,31; 11,21,31,41; 21,31,41,51;
        02,12,22,32; 12,22,32,42; 22,32,42,52;
        00,01,02,03; 01,02,03,04; 02,03,04,05; 03,04,05,06;
        10,11,12,13; 11,12,13,14; 12,13,14,15; 13,14,15,16;
        20,21,22,23; 21,22,23,24; 22,23,24,25; 23,24,25,26'''
    # ================================================

    Rows, Cols, weights = 6, 7, [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]

    AT2 = (np.array([a for a in range(42)])).reshape(Rows, Cols)
    def yx(s): return [int(s[0]), int(s[1])]
    Ss = [x.split(",") for x in [x for x in map4q.replace(" ", "").replace("\n", "").split(";")]]
    iMap4q = [list(map(yx, s)) for s in Ss]
    # ================================================
    def agent_NStep(obs, config):
        # valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
        io = it(obs.board, obs.mark)
        tfd_moves = io["moves"]
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        # Use the heuristic to assign a score to each possible board in the next step
        scores = dict(zip(tfd_moves, [score_move4(grid, col, obs.mark, config, N_STEPS) for col in tfd_moves]))
        # Get a list of columns (moves) that maximize the heuristic
        max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
        # Select at random from the maximizing columns
        return random.choice(max_cols)
    # ================================================
    def get_heuristic(grid, mark):
        q = qos(grid, mark)
        score = q[0] * 10000000 + q[1] * 1 - q[2] * 100 - q[3] * 100000
        return score
    # ================================================
    def minimax(node, depth, maximizingPlayer, mark, config):
        # ---------------------------------------------------------------
        if depth == 0 or win_4o_in(node): return get_heuristic(node, mark)
        # ---------------------------------------------------------------
        if mark == 1:
            tfd_moves = iTDF_e(node, mark)["moves"]
        elif mark == 2:
            tfd_1 = iTDF_e(node, mark)
            tfd_2 = iTDF_e(node, 3-mark)
            tfd_moves = tfd_1["moves"] if tfd_1["sum"] > tfd_2["sum"] else tfd_2["moves"]
        # ---------------------------------------------------------------
        if maximizingPlayer:
            value = -np.Inf
            for col in tfd_moves:
                child = drop_piece(node, col, mark, config)
                value = max(value, minimax(child, depth - 1, False, mark, config))
            return value
        else:
            value = np.Inf
            for col in tfd_moves:
                child = drop_piece(node, col, mark % 2 + 1, config)
                value = min(value, minimax(child, depth - 1, True, mark, config))
            return value
    # ================================================
    # Uses minimax to calculate value of dropping piece in selected column
    def score_move4(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        score = minimax(next_grid, nsteps - 1, False, mark, config)
        return score
    # ================================================
    def gpiece(grid, col, nrows=config.rows):
        for row in range(nrows - 1, -1, -1):
            if grid[row][col] == 0: break
        return [row, col]
    # ================================================
    def ipiece(col_move, mark, board, config):
        cboard = np.asarray(board.copy())
        grid = cboard.reshape(config.rows, config.columns)
        yx = gpiece(grid, col_move, nrows=config.rows)
        ia = AT2[yx[0], yx[1]]
        cboard[ia] = mark
        return cboard
    # ================================================
    def gip(col_move, mark, board, config):
        def gpiece2(grid, col, nrows=config.rows):
            for row in range(nrows - 1, -1, -1):
                if grid[row][col] == 0: break
            return [row, col]
        AT2 = (np.array([a for a in range(42)])).reshape(config.rows, config.columns)
        cboard = board.copy()
        grid = np.asarray(cboard).reshape(config.rows, config.columns)
        yx = gpiece2(grid, col_move, nrows=config.rows)
        ia = AT2[yx[0], yx[1]]
        cboard[ia] = mark
        return cboard
    # ================================================
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid
    # ================================================
    def win_4o_in(grid):
        for i in iMap4q:
            v = grid[i[0][0], i[0][1]] * grid[i[1][0], i[1][1]] * grid[i[2][0], i[2][1]] * grid[i[3][0], i[3][1]]
            if v == 1 or v == 16: return True
        return False
    # ================================================
    def qos(g, mark):
        gameVecs = [[g[i[0][0],i[0][1]],g[i[1][0],i[1][1]],g[i[2][0],i[2][1]],g[i[3][0],i[3][1]]] for i in iMap4q]
        _mark = 3 - mark
        q4, q3, _q3, _q4 = 0, 0, 0, 0
        for vec in gameVecs:
            _q, q = 0, 0
            if vec[0] == mark: q += 1;
            elif vec[0] == _mark: _q += 1;
            if vec[1] == mark: q += 1;
            elif vec[1] == _mark: _q += 1;
            if vec[2] == mark: q += 1;
            elif vec[2] == _mark: _q += 1;
            if vec[3] == mark:  q += 1;
            elif vec[3] == _mark: _q += 1;
            if q == 3: q3 += 1
            elif q == 4: q4 += 1
            if _q == 3: _q3 += 1
            elif _q == 4: _q4 += 1
        return [q4, q3, _q3, _q4]
    # ================================================
    def bathos2(pie, board, AT2):
        y = pie[0]
        x = pie[1]
        j = AT2[y][x]
        s = -1;
        for i in range(j, len(board), 7):
            if board[i] == 0: s += 1
        return s
    # ================================================
    def ish_4(v, mark, mayak=8):
        if len(v) < 4: return False
        for i in range(0, len(v) - 4 + 1):
            Q = True
            for j in range(0, 4):
                Q &= ((v[i + j][0] == mark) or (v[i + j][0] == mayak))
            if Q:
                return True
        return False
    # ================================================
    def ish_ns_35(v, board, mark, mayak=8):
        for i in range(len(v)):
            x = [e.copy() for e in v]
            if v[i][0] == 0:
                N = x.copy()
                N[i][0] = mark
                ish = ish_4(N, mark)
                if ish:
                    N[i][0] = 4
                    for j in range(len(v)):
                        if N[j][0] == mayak:
                            if bathos2(N[j][1], board, AT2) == 0:
                                N[j][0] = 35
                            else:
                                N[j][0] = 34
                    return [True, N]
        return [False, []]
    # ================================================
    def line(V, a, b):
        A = V[a]["Vector"]
        A.reverse()
        B = V[b]["Vector"][1:]
        return A + B
    # ================================================
    def Total(vector, mark, weights):
        opomark = mark % 2 + 1
        j, ves = 0, 0
        for i in range(1, len(vector)):
            if vector[i][0] == opomark:
                break
            elif vector[i][0] == mark:
                ves += 1
            elif vector[i][0] == 0:
                ves += weights[j]
                j += 1
        return ves
    # ================================================
    def clocs(vpie, grid, mark, weights, AT2, calc_DC=True, Rs=Rows, Cs=Cols):
        mayak = 8
        M = grid
        Y, X = vpie[0], vpie[1]
        V = {}
        Ro = M[Y, X]
        M[Y, X] = mayak
        v = [[M[i][X], [i, X]] for i in range(Y, -1, -1)]
        V["Noth"] = {"Total": Total(v, mark, weights), "Vector": v}
        v = [[M[i][X], [i, X]] for i in range(Y, Rs, 1)]
        V["South"] = {"Total": Total(v, mark, weights), "Vector": v}
        v = [[M[Y][i], [Y, i]] for i in range(X, -1, -1)]
        V["West"] = {"Total": Total(v, mark, weights), "Vector": v}
        v = [[M[Y][i], [Y, i]] for i in range(X, Cs, 1)]
        V["East"] = {"Total": Total(v, mark, weights), "Vector": v}
        v, i, j = [], vpie[0], vpie[1]
        while i >= 0 and j < Cs:
            v.append([M[i][j], [i, j]])
            i, j = i - 1, j + 1
        V["Noth-East"] = {"Total": Total(v, mark, weights), "Vector": v}
        v, i, j = [], vpie[0], vpie[1]
        while i >= 0 and j >= 0:
            v.append([M[i][j], [i, j]])
            i, j = i - 1, j - 1
        V["Noth-West"] = {"Total": Total(v, mark, weights), "Vector": v}
        v, i, j = [], vpie[0], vpie[1]
        while i < Rs and j < Cs:
            v.append([M[i][j], [i, j]])
            i, j = i + 1, j + 1
        V["South-East"] = {"Total": Total(v, mark, weights), "Vector": v}
        v, i, j = [], vpie[0], vpie[1]
        while i < Rs and j >= 0:
            v.append([M[i][j], [i, j]])
            i, j = i + 1, j - 1
        V["South-West"] = {"Total": Total(v, mark, weights), "Vector": v}
        # Noth-West   NOTH   Noth-East
        #             \ | /
        #    WEST    -- o --    EAST
        #             / | \
        # South-West  SOUTH  South-East
        V["Horizont"] = {"Total": V["West"]["Total"] + V["East"]["Total"], "Vector": line(V, "West", "East"), "35": []}
        V["Vertical"] = {"Total": V["Noth"]["Total"] + V["South"]["Total"], "Vector": line(V, "Noth", "South"),"35": []}
        V["Positive"] = {"Total": V["South-West"]["Total"] + V["Noth-East"]["Total"],
                         "Vector": line(V, "South-West", "Noth-East"), "35": []}
        V["Negative"] = {"Total": V["Noth-West"]["Total"] + V["South-East"]["Total"],
                         "Vector": line(V, "Noth-West", "South-East"), "35": []}
        k_H = (len(V["Horizont"]["Vector"]) / 4) ** 2
        k_V = (len(V["Vertical"]["Vector"]) / 4) ** 2
        k_P = (len(V["Positive"]["Vector"]) / 4) ** 2
        k_N = (len(V["Negative"]["Vector"]) / 4) ** 2
        V["Horizont"]["Total"] *= k_H
        V["West"]["Total"] *= k_H
        V["East"]["Total"] *= k_H
        V["Vertical"]["Total"] *= k_V
        V["Noth"]["Total"] *= k_V
        V["South"]["Total"] *= k_V
        V["Positive"]["Total"] *= k_P
        V["South-West"]["Total"] *= k_P
        V["Noth-East"]["Total"] *= k_P
        V["Negative"]["Total"] *= k_N
        V["Noth-West"]["Total"] *= k_N
        V["South-East"]["Total"] *= k_N
        tfd = 0
        tfd += V["Noth"]["Total"]
        tfd += V["Noth-East"]["Total"]
        tfd += V["East"]["Total"]
        tfd += V["South-East"]["Total"]
        tfd += V["South"]["Total"]
        tfd += V["South-West"]["Total"]
        tfd += V["West"]["Total"]
        tfd += V["Noth-West"]["Total"]
        V["TFD"] = {"Total": tfd, "yx": vpie}
        # ///////////////////////////////////////////////////////////////////
        if not calc_DC:
            M[Y, X] = Ro
            return V
        # ///////////////////////////////////////////////////////////////////
        board = np.asarray(grid).reshape(-1)
        V["Horizont"]['35'] = ish_ns_35(V["Horizont"]["Vector"], board, mark)
        V["Vertical"]['35'] = ish_ns_35(V["Vertical"]["Vector"], board, mark)
        V["Positive"]['35'] = ish_ns_35(V["Positive"]["Vector"], board, mark)
        V["Negative"]['35'] = ish_ns_35(V["Negative"]["Vector"], board, mark)
        qdc = 0
        if V["Horizont"]['35'][0]: qdc += 1
        if V["Vertical"]['35'][0]: qdc += 1
        if V["Positive"]['35'][0]: qdc += 1
        if V["Negative"]['35'][0]: qdc += 1
        V["DC"] = {"qnt": qdc, "yx": vpie}
        M[Y, X] = Ro
        return V
    # ================================================
    def iclocs(board, mark, weights, AT2, calc_DC=True, Rs=Rows, Cs=Cols):
        T2 = (np.array(board)).reshape(Rs, Cs)
        vmoves = [c for c in range(Cs) if board[c] == 0]
        vpies = [gpiece(T2, move) for move in vmoves]

        Clocs = [clocs(vpies[i], T2, mark, weights, AT2) for i in range(len(vpies))]

        QntDC, Adrs = 0, []
        SumTfd, queTfd = 0, []
        for cloc in Clocs:
            SumTfd += cloc["TFD"]['Total']
            queTfd.append([cloc["TFD"]['Total'], cloc["TFD"]['yx'][1]])

            if calc_DC == True:

                if cloc["DC"]['qnt'] >= 2:
                    QntDC += 1
                    Adrs.append(cloc["DC"]['yx'])
                    # if cloc["DC"]['qnt'] == 2:   # not here to compare, but you can here but first а в _4_1 == _4_2
                    #     if len(cloc["DC"]['adrs']) >= 2 and cloc["DC"]['adrs'][0] != cloc["DC"]['adrs'][1]:
                    #         QntDC += 1
                    #         Adrs.append(cloc["DC"]['yx'])

        queTfd.sort(reverse=True)

        ItogClocs = {"TFD": {"sum": SumTfd, "que": queTfd}, "DC": {"qnt": QntDC, "adrs": Adrs}}
        Clocs.insert(0, ItogClocs)
        return Clocs
    # ================================================
    def iTDF(clocs, width=WIDTH):
        i = {}
        i["sum"] = clocs[0]["TFD"]['sum']
        i["que"] = [clocs[0]["TFD"]['que'][i] for i in range(len(clocs[0]["TFD"]['que'])) if i < width]
        i["moves"] = [q[1] for q in i["que"]]
        i["qdc"] = clocs[0]["DC"]["qnt"]
        i["idc"] = clocs[0]["DC"]['adrs']
        i["mdc"] = i["idc"][0][1] if i["qdc"] != 0 else []
        # 'sum'    = 36.4375
        # 'que'    = [[8.4375, 5], [8.0, 4], [7.25, 2]]
        # 'moves'  = [5, 4, 2]
        # 'qdc'    = 2
        # 'idc'    = [[3,2], [3,4]]
        # 'mdc'    = 2
        return i
    # ================================================
    def it(board, mark, width=WIDTH, calc_DC=True, Rs=Rows, Cs=Cols):
        clocs = iclocs(board, mark, weights, AT2, calc_DC)
        i = iTDF(clocs, width)
        return i
    # ================================================
    def iTDF_e(grid, mark, weights=weights, AT2=AT2, width=WIDTH):
        vmoves = [c for c in grid[0] if c == 0]
        vpies = [gpiece(grid, move) for move in vmoves]
        clocses = [clocs(vpies[i], grid, mark, weights, AT2) for i in range(len(vpies))]
        SumTfd, queTfd = 0, []
        for cloc in clocses:
            SumTfd += cloc["TFD"]['Total']
            queTfd.append([cloc["TFD"]['Total'], cloc["TFD"]['yx'][1]])
        queTfd.sort(reverse=True)
        ItogClocs = {"TFD": {"sum": SumTfd, "que": queTfd}}
        clocses.insert(0, ItogClocs)
        i = {}
        i["que"] = [clocses[0]["TFD"]['que'][i] for i in range(len(clocses[0]["TFD"]['que'])) if i < width]
        i["sum"]   = clocses[0]["TFD"]['sum'] # 'sum'    = 36.4375
        i["moves"] = [q[1] for q in i["que"]] # 'moves'  = [5, 4, 2]
        return i
    # ================================================
    def deco(X):
        def decomp(x):
            d0, d1, d2, d3 = x[3], x[2], x[1], x[0]
            d4 = [x[3][0], x[2][1], x[1][2], x[0][3]]
            d5 = [x[3][3], x[2][2], x[1][1], x[0][0]]
            d6 = [x[3][0], x[2][0], x[1][0], x[0][0]]
            d7 = [x[3][1], x[2][1], x[1][1], x[0][1]]
            d8 = [x[3][2], x[2][2], x[1][2], x[0][2]]
            d9 = [x[3][3], x[2][3], x[1][3], x[0][3]]
            return [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]
        x11 = [X[0: 4], X[7:11], X[14:18], X[21:25]]
        x12 = [X[1: 5], X[8:12], X[15:19], X[22:26]]
        x13 = [X[2: 6], X[9:13], X[16:20], X[23:27]]
        x14 = [X[3: 7], X[10:14], X[17:21], X[24:28]]
        x21 = [X[7:11], X[14:18], X[21:25], X[28:32]]
        x22 = [X[8:12], X[15:19], X[22:26], X[29:33]]
        x23 = [X[9:13], X[16:20], X[23:27], X[30:34]]
        x24 = [X[10:14], X[17:21], X[24:28], X[31:35]]
        x31 = [X[14:18], X[21:25], X[28:32], X[35:39]]
        x32 = [X[15:19], X[22:26], X[29:33], X[36:40]]
        x33 = [X[16:20], X[23:27], X[30:34], X[37:41]]
        x34 = [X[17:21], X[24:28], X[31:35], X[38:42]]
        XXX = [[x11, x12, x13, x14], [x21, x22, x23, x24], [x31, x32, x33, x34]]
        Y = []
        for XX in XXX:
            for X in XX: Y.append(decomp(X))
        return Y
    # ================================================
    def f1z(A, m):
        if m == 1:
            if A == [0, 1, 1, 1]: return 0
            if A == [1, 0, 1, 1]: return 1
            if A == [1, 1, 0, 1]: return 2
            if A == [1, 1, 1, 0]: return 3
        if m == 2:
            if A == [0, 2, 2, 2]: return 0
            if A == [2, 0, 2, 2]: return 1
            if A == [2, 2, 0, 2]: return 2
            if A == [2, 2, 2, 0]: return 3
        return -1
    # ================================================
    def f2o(A, m):
        if m == 1:
            if A == [0, 1, 1, 0]: return [0, 3]
        if m == 2:
            if A == [0, 2, 2, 0]: return [0, 3]
        return [-1, -1]
    # ================================================
    def fo0o(A, m):
        if m == 1:
            if A == [0, 1, 0, 1]: return [2]
            if A == [1, 0, 1, 0]: return [1]
        if m == 2:
            if A == [0, 2, 0, 2]: return [2]
            if A == [2, 0, 2, 0]: return [1]
        return [-1]
    # ================================================
    def f2z(A, m):
        if m == 1:
            if A == [1, 1, 0, 0]: return [2, 3]
            if A == [0, 0, 1, 1]: return [0, 1]
            # if A == [1,0,0,1]: return [1,2]
        if m == 2:
            if A == [2, 2, 0, 0]: return [2, 3]
            if A == [0, 0, 2, 2]: return [0, 1]
            # if A == [2,0,0,2]: return [1,2]
        return [-1, -1]
    # ================================================
    def bathos(G, j):
        if j == -1: return -1
        s = -1;
        for i in range(j, len(G), 7):
            if G[i] == 0: s += 1
        return s
    # ================================================
    def shield_1s(G, g, a, m):
        u0, u1, u2 = [], [], []
        for i in range(len(a)):
            for r in range(len(g[i])):
                j = f1z(g[i][r], m)
                u = a[i][r][j]
                if j != - 1:
                    if bathos(G, u) == 0:  u0.append(u)
                    if bathos(G, u) == 1:
                        u1.append(u)
                    else:
                        u2.append(u)

        return [u0, u1, u2]
    # ================================================
    def shield_2s(G, g, a, m):
        T = []
        for i in range(len(a)):
            for r in range(len(g[i])):
                t2 = f2o(g[i][r], m)  # ==============
                j, jj = t2[0], t2[1]
                if (j != -1) and (jj != -1):
                    a1, a2 = a[i][r][j], a[i][r][jj]
                    if (bathos(G, a1) == 0 and bathos(G, a2) == 0):
                        T.append(a1)
                        T.append(a2)
                t1 = fo0o(g[i][r], m)  # ==============
                j = t1[0]
                if (j != -1):
                    a1 = a[i][r][j]
                    if bathos(G, a1) == 0: T.append(a1)
        return T
    # ================================================
    def shield_2z(G, g, a, m):
        Z = []
        for i in range(len(a)):
            for r in range(len(g[i])):
                t = f2z(g[i][r], m)
                j, jj = t[0], t[1]
                if (j != -1) and (jj != -1):
                    a1, a2 = a[i][r][j], a[i][r][jj]
                    if bathos(G, a1) == 0: Z.append(a1)
                    if bathos(G, a2) == 0: Z.append(a2)
        if len(Z) > 0:
            A2d = [[Z.count(s), s] for s in set(Z)]
            A2d.sort()
            E = A2d[-1:][0][1]
            return [E % 7]
        return []
    # ================================================
    def shield_dc(u, io, obs, config):
        idc = [i[1] for i in io['idc'] if i[1] not in u]
        if len(idc) == 1: return idc[0]
        for move in idc:
            obs_board = gip(move, obs.mark, obs.board, config)
            way = deco(obs_board)
            aix = deco([n for n in range(42)])
            I = shield_1s(obs_board, way, aix, obs.mark)
            if len(I[0]) > 0: return I[0][0] % 7
        return random.choice(idc)
    # ================================================
    def select_move(T, U, Z, r, valid_moves):
        u = list(set([u % 7 for u in U[1]]))
        t = list(set([t % 7 for t in T]))
        # -----------
        if len(t) > 0 and (t[0] not in u): return t[0]
        if (r != -1)  and (r not in u):    return valid_moves[r]
        if len(Z) > 0 and (Z[0] not in u): return Z[0]
        # -----------
        for move in valid_moves:
            if move not in u: return move
        # -----------
        return random.choice(valid_moves)
    # =============================================================
    way = deco(obs.board)
    aix = deco([n for n in range(42)])
    I = shield_1s(obs.board, way, aix, obs.mark)
    T = shield_2s(obs.board, way, aix, obs.mark % 2 + 1)
    U = shield_1s(obs.board, way, aix, obs.mark % 2 + 1)
    Z = shield_2z(obs.board, way, aix, obs.mark % 2 + 1)
    Ti= shield_2s(obs.board, way, aix, obs.mark)
    Zi= shield_2z(obs.board, way, aix, obs.mark)
    u = list(map(lambda x: x % 7, U[1]))  # U[1] : o_oo = deep 1
    # ---------------------------------------------------------------------------
    if len(I[0]) > 0: return I[0][0] % 7  # 100% win                # attack  1m
    if len(U[0]) > 0: return U[0][0] % 7  # 100% loss               # defense 1m
    # ---------------------------------------------------------------------------
    io1 = it(obs.board, obs.mark)
    if (io1['qdc'] >= 1 and io1['mdc'] not in u):                   # attack  DC
        return io1['mdc']
    # -----------------------------------------------------
    io2 = it(obs.board, obs.mark % 2 + 1)
    if (io2['qdc'] >= 1):
        if (io2['qdc'] == 1 and io2['mdc'] not in u):               # defense DC
            return io2['mdc']
        elif io2['qdc'] > 1:
            return shield_dc(u, io2, obs, config)
    # ---------------------------------------------------------------------------
    vms = [c for c in range(config.columns) if obs.board[c] == 0]
    _7steps = [[] for c in range(len(vms))]

    for i in range(len(vms)):
        cboard = ipiece(vms[i], obs.mark, obs.board, config)
        io = it(cboard, obs.mark % 2 + 1)
        _7steps[i] = [io['qdc'], io, cboard]
    # ---------------------------------------------------------------------------
    if len(Ti) > 0 and (Ti[0] % 7) not in u: return Ti[0] % 7
    if len(T)  > 0 and (T [0] % 7) not in u: return T [0] % 7
    # ---------------------------------------------------------------------------
    Re, agent_move, i = -1, -1, np.sum(np.array(obs.board))

    if (i > 11): agent_move = agent_NStep(obs, config)

    if (agent_move != -1) and (agent_move not in u): return agent_move

    Re = 3 if i <= 7 else 4 if i <= 11 else -1

    return select_move(T, U, Z, Re, [c for c in range(config.columns) if obs.board[c] == 0])