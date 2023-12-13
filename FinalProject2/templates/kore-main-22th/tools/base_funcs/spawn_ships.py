from kaggle_environments.envs.kore_fleets.helpers import ShipyardAction


def mk_ships(persist, board, sy):
    if board.step < persist['const'].BuildShip.Last:
        to_spawn = min(persist['kore'] // 10, sy.max_spawn)
        if 0 < to_spawn:
            sy.next_action = ShipyardAction.spawn_ships(to_spawn)
            persist['todo'].remove(sy)
            persist['kore'] -= 10 * to_spawn
