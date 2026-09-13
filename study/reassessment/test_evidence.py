import unittest
from evidence import EvidenceGate,route_stream
from fixture import build_fixture


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.claim={'id':'c','name':'Site','color':'red','relevant':True}
        self.history=[dict(predicted='red',observed='red' if i<13 else 'blue') for i in range(16)]
        self.registry={'a':'one','alias':'one','b':'two','c':'three'}
        self.histories={n:self.history for n in ('one','two','three')}

    def gate(self):
        return EvidenceGate([self.claim],self.registry,self.histories)

    def vote(self,g,source,color='red'):
        return g.ingest(dict(id=source,name='Site',source=source,color=color))

    def test_duplicate_not_independent(self):
        g=self.gate()
        self.vote(g,'a');self.vote(g,'alias')
        self.assertAlmostEqual(g.decision(self.claim)['probability'],0.7)
        self.assertEqual(g.decision(self.claim)['route'],'informational')
        self.vote(g,'b')
        self.assertAlmostEqual(g.decision(self.claim)['probability'],49/52)
        self.assertEqual(g.decision(self.claim)['route'],'formative')

    def test_later_conflict_reopens_decision(self):
        g=self.gate();self.vote(g,'a');self.vote(g,'b')
        result=self.vote(g,'c','blue')
        self.assertEqual(result['before']['c']['route'],'formative')
        self.assertEqual(result['after']['c']['route'],'informational')

    def test_unknown_source_and_sham_do_not_create_support(self):
        g=self.gate();self.vote(g,'unknown');self.vote(g,'a',None)
        self.assertAlmostEqual(g.decision(self.claim)['probability'],0.25)

    def test_origin_internal_conflict_is_not_last_writer_wins(self):
        g=self.gate();self.vote(g,'a');self.vote(g,'alias','blue')
        self.assertAlmostEqual(g.decision(self.claim)['probability'],0.25)

    def test_gate_is_not_a_truth_oracle(self):
        # Even when a claim is false, agreeing sources can pass the threshold.
        g=self.gate();self.vote(g,'a');self.vote(g,'b')
        self.assertEqual(g.decision(self.claim)['route'],'formative')


if __name__=='__main__':
    unittest.main()
