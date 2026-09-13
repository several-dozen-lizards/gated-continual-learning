import unittest
from world import NAMES, TRAIN_TEMPLATES, EVAL_TEMPLATES, make_world, route, matched_random


class WorldTests(unittest.TestCase):
    def test_corrections_and_partition(self):
        self.assertFalse(set(NAMES['development']) & set(NAMES['test']))
        self.assertFalse(set(TRAIN_TEMPLATES) & set(EVAL_TEMPLATES))
        w = make_world('test', 17)
        for name, group in w['groups'].items():
            if group == 'correction':
                self.assertNotEqual(w['initial'][name], w['final'][name])
            if group == 'retention':
                self.assertEqual(w['initial'][name], w['final'][name])

    def test_router_does_not_require_answers(self):
        w = make_world('test', 17)
        for e in w['stream']:
            minimal = {k:e[k] for k in ('relevant','verified')}
            self.assertEqual(route(e), route(minimal))
        useful = [e for e in w['stream'] if w['evaluator_labels'][e['id']]['useful']]
        self.assertEqual(sum(route(e)['route']=='formative' for e in useful), 6)
        false = [e for e in w['stream'] if w['evaluator_labels'][e['id']]['kind'] in ('rumor','spoof')]
        self.assertTrue(all(route(e)['route']=='informational' for e in false))

    def test_budget_match_and_order(self):
        w = make_world('test', 29)
        selected = [i for i,e in enumerate(w['stream']) if route(e)['route']=='formative']
        cost = lambda e: (len(e['name']),len(e['color']))
        chosen = matched_random(w['stream'], selected, cost, 43)
        self.assertEqual(chosen, sorted(set(chosen)))
        self.assertEqual(sorted(cost(w['stream'][i]) for i in selected), sorted(cost(w['stream'][i]) for i in chosen))


if __name__ == '__main__':
    unittest.main()
