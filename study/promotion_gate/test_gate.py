import hashlib,json,tempfile,unittest
from pathlib import Path
from gate import decide,choose,activate

def row(name,prediction='red',expected='red'):
    colors=('red','blue','green','gold')
    return dict(name=name,template='probe',prediction=prediction,expected=expected,
        probabilities=[float(c==prediction) for c in colors])

class GateTests(unittest.TestCase):
    def test_improvement_only_promotes(self):
        self.assertTrue(decide([row('A','blue'),row('B')],[row('A'),row('B')])['promote'])

    def test_net_improvement_does_not_hide_collateral(self):
        before=[row('A','blue'),row('B','blue'),row('C')]
        after=[row('A'),row('B'),row('C','blue')]
        result=decide(before,after)
        self.assertFalse(result['promote']);self.assertEqual(len(result['gains']),2);self.assertEqual(len(result['losses']),1)

    def test_sham_no_gain(self):
        result=decide([row('A')],[row('A')])
        self.assertFalse(result['promote']);self.assertEqual(result['reason'],'no_demonstrated_gain')

    def test_invalid_receipts_fail_closed(self):
        for bad in [[row('B')],[row('A'),row('A')],[dict(row('A'),probabilities=[float('nan'),0,0,0])],
                    [dict(row('A'),prediction='blue')],[row('A',expected='blue')]]:
            with self.assertRaises(ValueError):decide([row('A')],bad)

    def test_fallback(self):
        self.assertEqual(choose(dict(drift=dict(promote=False),random=dict(promote=True))),'random')
        self.assertEqual(choose(dict(drift=dict(promote=False),random=dict(promote=False))),'baseline')

    def test_activation_preserves_immutable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);weights=root/'weights';weights.write_bytes(b'original')
            checkpoints=dict(baseline=dict(path=str(root),hashes={'weights':hashlib.sha256(b'original').hexdigest()}))
            activate(root/'active.json','baseline',checkpoints)
            self.assertEqual(json.loads((root/'active.json').read_text())['selected'],'baseline')
            self.assertEqual(weights.read_bytes(),b'original')
            with self.assertRaises(FileExistsError):activate(root/'active.json','baseline',checkpoints)
            weights.write_bytes(b'changed')
            with self.assertRaises(ValueError):activate(root/'rejected.json','baseline',checkpoints)
            self.assertFalse((root/'rejected.json').exists())

if __name__=='__main__':unittest.main()
