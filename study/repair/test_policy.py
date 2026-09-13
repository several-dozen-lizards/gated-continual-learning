import copy,json,unittest
from pathlib import Path
from policy import repair
ROOT=Path(__file__).resolve().parent

class RepairTests(unittest.TestCase):
    def test_withdrawal_uncertainty_and_replacement(self):
        for seed in (17,29,43):
            p=json.loads((ROOT/'fixtures'/f'{seed}.json').read_text())
            result=repair(p)
            self.assertEqual(result['initial'][p['old_wrong_color']]['route'],'formative')
            self.assertEqual(result['receipts'][0]['after'][p['old_wrong_color']]['route'],'informational')
            p['repair_events']=p['repair_events'][:2]
            withdrawn=repair(p)
            self.assertFalse(withdrawn['selected'])
            self.assertEqual(len(withdrawn['ledger']),11)
            self.assertEqual(len(result['ledger']),12)

    def test_evaluator_and_truth_labels_do_not_control_policy(self):
        p=json.loads((ROOT/'fixtures'/'17.json').read_text())
        expected=repair(p)
        altered=copy.deepcopy(p)
        altered['world']={}
        altered['evidence']['evaluator']={}
        altered['old_wrong_color']='not a color'
        self.assertEqual(expected,repair(altered))

    def test_sham_preserves_ledger_content(self):
        p=json.loads((ROOT/'fixtures'/'17.json').read_text())
        result=repair(p,sham=True)
        self.assertEqual({c['name']:c['color'] for c in result['ledger']},
            {c['name']:c['color'] for c in p['accepted_ledger']})

if __name__=='__main__':unittest.main()
