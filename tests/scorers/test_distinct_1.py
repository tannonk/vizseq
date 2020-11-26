#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from vizseq.scorers.distinct_n import Distinct1Scorer

class Distinct1ScorerTestCase(unittest.TestCase):
    
    def test(self):
        """
        test case for Distinct-1 (sentence level and corpus level)
        """        
        hyp1 = "this is a simple test sentence for measuring distinct n-grams"
        hyp2 = ""
        hyp3 = "test test test test"
        hyps = [hyp1, hyp2, hyp3]

        scorer = Distinct1Scorer(corpus_level=True, sent_level=True)

        result = scorer.score(hyps)

        self.assertAlmostEqual(result.corpus_score, 0.417)
        self.assertListEqual(result.sent_scores, [1.0, 0.0, 0.25])


if __name__ == '__main__':
    unittest.main()