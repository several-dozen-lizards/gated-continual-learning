import copy,json,unittest
from pathlib import Path
from selection import schedule
ROOT=Path(__file__).resolve().parent

class SelectionTests(unittest.TestCase):
    def test_frozen_schedule_and_no_future_leakage(self):
        for seed in (17,29,43):
            p=json.loads((ROOT/'fixtures'/f'{seed}.json').read_text())
            available={c['name'] for c in p['initial_ledger']}
            result=schedule(p['initial_ledger'],p['waves'],'small',seed)
            self.assertEqual(result,p['schedules']['small'])
            for stage in result:
                names={c['name'] for c in stage['buffer']}
                self.assertEqual(len(names),4)
                self.assertLessEqual(names,available)
                self.assertFalse(names&{c['name'] for c in stage['updates']})
                available.update(c['name'] for c in stage['updates'])

    def test_color_values_do_not_change_membership(self):
        p=json.loads((ROOT/'fixtures'/'17.json').read_text())
        altered=copy.deepcopy(p)
        for c in altered['initial_ledger']+sum(altered['waves'],[]):c['color']='unused'
        first=schedule(p['initial_ledger'],p['waves'],'small',17)
        second=schedule(altered['initial_ledger'],altered['waves'],'small',17)
        self.assertEqual([[c['name'] for c in s['buffer']] for s in first],[[c['name'] for c in s['buffer']] for s in second])

    def test_training_contents(self):
        p=json.loads((ROOT/'fixtures'/'17.json').read_text())
        for arm,counts in [('none',[4,4]),('small',[8,8]),('full',[16,20])]:
            stages=schedule(p['initial_ledger'],p['waves'],arm,17)
            self.assertEqual([len(s['training']) for s in stages],counts)

if __name__=='__main__':unittest.main()
