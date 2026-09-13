import unittest
from collections import Counter
from selection import rank_losses,matched_random

class SelectionTests(unittest.TestCase):
    def test_ranks_loss_not_absolute_difficulty(self):
        before={'stable_hard':[.1,.1],'forgotten':[.9,.9],'improved':[.1,.1]}
        after={'stable_hard':[.1,.1],'forgotten':[.2,.2],'improved':[.8,.8]}
        result=rank_losses(before,after,1)
        self.assertEqual(result['selected_names'],['forgotten'])
        self.assertEqual(next(r['score'] for r in result['ranking'] if r['name']=='improved'),0)

    def test_ties_are_deterministic(self):
        before={'B':[1.0],'A':[1.0]}
        self.assertEqual(rank_losses(before,before,1)['selected_names'],['A'])

    def test_random_matches_cost_without_duplicates(self):
        claims=[dict(name=n,color='red') for n in ['A','B','CC','DD','EEE','FFF']]
        selected=[claims[0],claims[2],claims[4]]
        cost=lambda c:(len(c['name']),len(c['name'])+1)
        result=matched_random(claims,selected,cost,17)
        self.assertEqual(Counter(map(cost,result)),Counter(map(cost,selected)))
        self.assertEqual(len({c['name'] for c in result}),3)
        self.assertEqual(result,matched_random(claims,selected,cost,17))

    def test_rejects_invalid_probe_pair(self):
        with self.assertRaises(AssertionError):rank_losses({'A':[.1]},{'B':[.1]})
        with self.assertRaises(AssertionError):rank_losses({'A':[float('nan')]},{'A':[.1]})

if __name__=='__main__':unittest.main()
