def log_it(persist: dict, lvl: int, msg: str):
    if lvl <= persist['log_lvl']:
        msg = f"Round {persist['round']}: " + msg
        if persist['log_f_name'] == 'None':
            print(msg)
        else:
            f = open(persist['log_f_name'], "a")
            print(msg, file=f)


def time_info(persist, board, elapsed_time):
    persist['sum_time'] += elapsed_time
    if persist['max_rnd_time'][0] < elapsed_time:
        persist['max_rnd_time'] = (elapsed_time, board.step)
    if board.step % 50 == 49:
        (rnd_t, rnd) = persist['max_rnd_time']
        log_it(persist, 30, f"Playa {board.current_player.id}: CalcTime: {persist['sum_time']:.2f}, Slowest Round {rnd}: {rnd_t:.2f}")
        persist['sum_time'] = 0.0
        persist['max_rnd_time'] = (0.0, 0)


def prt_time(persist, elapsed_time, msg):
    log_it(persist, 70, f"After {msg} {elapsed_time:.2f}: {[sy.position for sy in persist['todo']]}")
