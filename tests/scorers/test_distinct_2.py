#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from vizseq.scorers.distinct_n import Distinct2Scorer
import numpy as np


class Distinct2ScorerTestCase(unittest.TestCase):
    
    def test(self):
        """
        test case for Distinct-2 (sentence level and corpus level)
        """        
        
        scorer = Distinct2Scorer(corpus_level=True, sent_level=True)

        # empty string
        hyps = [
            'this is a duplicate test sentence .',
            '',
            'this is not a duplicate test sentence .']

        result = scorer.score(hyps)
        self.assertTrue(np.allclose(np.array(result.sent_scores), np.array([0.8, 1e-7, 0.8]), atol=0.1))
        

if __name__ == '__main__':
    unittest.main()