class Constant:

    def __init__(self):
        pass

    class NewShipyard:
        Ship_addon = 30
        Ship_vs_shipyard = (128, 118, 118, 113, 108)
        Max_shipyard = 10
        New_dist = 6
        New_dist_first = 5
        Min_dist_me = 5
        Min_dist_opp = 7
        New_chk_dist = 6
        New_chk_decay = 0.85
        Round_dist = 7
        Opp_wait = (1, 2, 2, 3)  # How much own shipyard should stop and produce ships, during opponent ship creation
        Opp_vs_me_factor = 12
        New_sy_threshold = 53

    class Mine:
        fp5_max_ship_dist = 8
        fp6_max_ship_dist = 8
        fp7_max_ship_dist = 10
        fp7_plan_count = 5
        max_fp7_len = (0, 21, 21, 19, 19, 17)
        Strict_mining = 150     # No collisions with own ships allowed
        Risk_factor = 0.9
        Ships_per_shipyard = 6

    class BuildShip:
        Last = 375
        Miner_pause_for_Builder = 3
        M2f_threshold = 27
        M2f_base = 21

    class PreCog:
        attack_detection = 24

    class Logger:
        level = 50
        f_name = 'None'

