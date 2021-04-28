#!/usr/bin/python3
# -*- coding: utf-8 -*-

import unittest
from vizseq.scorers.distinct_n import Distinct1Scorer
import numpy as np

class Distinct1ScorerTestCase(unittest.TestCase):
    
    def test(self):
        """
        test case for Distinct-1 (sentence-level (intra) and
        corpus-level (inter))
        """    

        scorer = Distinct1Scorer(corpus_level=True, sent_level=True)

        # sentence repetition
        hyps = [
            'this is a duplicate test sentence .',
            'this is a duplicate test sentence .',
            'this is a duplicate test sentence .'
            ]

        result_1 = scorer.score(hyps)

        self.assertAlmostEqual(result_1.corpus_score, 1.0/len(hyps), 2)
        self.assertTrue(np.allclose(np.array(result_1.sent_scores), np.array([1.0, 1.0, 1.0])))

        # empty string
        hyps = [
            'this is a duplicate test sentence .',
            '',
            'this is not a duplicate test sentence .']

        result_2 = scorer.score(hyps)
        self.assertGreater(result_2.corpus_score, result_1.corpus_score)
        self.assertTrue(np.allclose(np.array(result_2.sent_scores), np.array([1.0, 1e-7, 1.0]), atol=0.1))


        # token-level repetition
        hyps = [
            'this is a sentence sentence sentence sentence',
            'this is another sentence sentence sentence sentence',
            'this is yet another sentence sentence sentence sentence sentence']

        result_3 = scorer.score(hyps)
        self.assertLess(result_3.corpus_score, result_2.corpus_score)
        print(result_3.sent_scores)
        self.assertTrue(np.allclose(np.array(result_3.sent_scores), np.array([0.5, 0.5, 0.5]), atol=0.1))

        # high variability
        hyps = [
            'this a simple test sentence .',
            'the cat sat on that mat ;',
            'he was n\'t drinking milk !']

        result_4 = scorer.score(hyps)
        self.assertAlmostEqual(result_4.corpus_score, 1.0, 2)
        self.assertTrue(np.allclose(np.array(result_4.sent_scores), np.array([1.0, 1.0, 1.0]),atol=0.1))
        

        # low variability
        hyps = [
            'this is a simple test sentence .',
            'this is another simple test sentence .',
            'and this is yet another simple test sentence .']

        result_5 = scorer.score(hyps)
        self.assertLess(result_5.corpus_score, result_4.corpus_score)
        # self.assertTrue(np.allclose(np.array(result_5.sent_scores), np.array([1.0, 1e-7, 1.0])))        

if __name__ == '__main__':
    unittest.main()