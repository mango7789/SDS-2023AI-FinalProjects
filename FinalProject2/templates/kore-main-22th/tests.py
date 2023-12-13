#!/usr/bin/python3
#

import tools.base_funcs.funcs as funcs
import tools.flight_plan as flightplan
import unittest


class TestStringMethods(unittest.TestCase):

    def test_one_way_path(self):
        print("tools.funcs.one_way_path")
        self.assertEqual(funcs.one_way_path((18, 10), (3, 12)), [['E', 'N', '1', 'E', '4'], ['E', '1', 'N', '1', 'E', '3'],
                              ['E', '2', 'N', '1', 'E', '2'], ['E', '3', 'N', '1', 'E', '1'], ['E', '4', 'N', '1', 'E'],
                              ['E', '5', 'N', '1'], ['N', 'E', '5', 'N'], ['N', '1', 'E', '5']], "one_way_path failed for (18, 10) => (3, 12)")
        for (a, b) in (((0, 0), (1, 2)), ((3, 4), (4, 6)), ((20, 4), (0, 6)), ((20, 19), (0, 0)), ((0, 20), (1, 1))):
            self.assertEqual(funcs.one_way_path(a, b),
                             [['E', 'N', '1'], ['N', 'E', 'N'], ['N', '1', 'E']],
                             "one_way_path failed for (0, 0) => (1, 2)")
        self.assertEqual(funcs.one_way_path((1, 19), (1, 11)), [['S']], "one_way_path failed for (1, 19) => (1, 11)")

    def test_two_way_path(self):
        print("tools.funcs.two_way_path")
        self.assertEqual(funcs.two_way_path((15, 5), (8, 5)), [['W', '6', 'E']], "two_way_path failed for (15, 5) => (8, 5)")

    def test_split_path(self):
        print("tools.funcs.split_path")
        self.assertEqual(funcs.split_path("S8E"), ['S', '8', 'E'], "split_path failed for S8E => ['S', '8', 'E']")
        self.assertEqual(funcs.split_path("E2S"), ['E', '2', 'S'], "split_path failed for E2S => ['E', '2', 'S']")
        self.assertEqual(funcs.split_path("NS"), ['N', 'S'], "split_path failed for NE => ['N', 'S']")

    def test_cpath2coords(self):
        print("tools.funcs.cpath2coords")
        self.assertEqual(funcs.cpath2coords((5, 15), ['S', '8', 'E'], [(15, 5), (8, 6)], [5, 15]),
                         [(5, 14), (5, 13), (5, 12), (5, 11), (5, 10), (5, 9), (5, 8), (5, 7), (5, 6), (6, 6), (7, 6), (8, 6)],
                         "cpath2coords failed")
        self.assertEqual(funcs.cpath2coords((5, 15), ['E', '2', 'S'], [(15, 5), (8, 6)], [5, 15]),
                         [(6, 15), (7, 15), (8, 15), (8, 14), (8, 13), (8, 12), (8, 11), (8, 10), (8, 9), (8, 8), (8, 7), (8, 6)],
                         "cpath2coords failed")

    def test_coord_int(self):
        print("tools.funcs.coord2int tools.funcs.int2coord")
        for x in range(21):
            for y in range(21):
                self.assertEqual(funcs.ind2coord(funcs.coord2int((x, y))), (x, y), "coord => int => coord")
        for ind in range(442):
            self.assertEqual(funcs.coord2int(funcs.ind2coord(ind)), ind, "int => coord => int")

    def test_calc_fp7_addon(self):
        print("tools.flight_plan.calc_addon")
        for fp, ms, res in [(32, 3, 1), (35, 7, 1), (15, 7, 2), (5, 7, 4), (5, 5, 5)]:
            self.assertEqual(flightplan.calc_fp7_addon(fp, ms), res, f"ship_count: {fp}, max_spawn: {ms} => {res}")

if __name__ == '__main__':
    unittest.main()
