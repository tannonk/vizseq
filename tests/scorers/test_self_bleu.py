#!/usr/bin/python3
# -*- coding: utf-8 -*-


import unittest
from vizseq.scorers.self_bleu import SelfBLEUScorer

class SelfBLEUScorerTestCase(unittest.TestCase):
    
    def test(self):
        """
        test case for Distinct-1 (sentence level and corpus level)
        """        
        hyps = ["the cat sat on the mat .", "a cat sat on the mat .", "the cat sat under the mat .", "the fat cat sat on the mat .", "the cat sat on the small mat .", "the cat sat on the mat !"]
    
        scorer = SelfBLEUScorer(corpus_level=True, sent_level=True)

        result = scorer.score(hyps)

        self.assertAlmostEquals(result.corpus_score, 62.15, places=2)
        self.assertListEqual(result.sent_scores, [80.911, 80.911, 0.0, 70.711, 59.46, 80.911])


if __name__ == '__main__':
    unittest.main()