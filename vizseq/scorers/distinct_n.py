#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Author: Tannon Kew
Email: kew@cl.uzh.ch
Date: 26.11.20
"""

from typing import Optional, List, Dict
from vizseq.scorers import register_scorer, VizSeqScorer, VizSeqScore


def _get_sent_distinct(
    hypothesis: List[str],
    N: int = 2,
    extra_args: Optional[Dict[str, str]] = None
) -> List[float]:
    """
    Compute distinct-N for a single hypothesis sentence.

    :param sentence: a list of tokens
    :param N: int, ngram.
    :return: float, the metric value.
    """

    scores = []
    for h in hypothesis:
        h = h.split()
        
        if len(h) < N:
            scores.append(0.0)

        else:
            grams = [tuple(h[i:i+N]) for i in range(len(h)-N+1)]
            uniq_grams = set(grams)
            score = len(uniq_grams) / len(grams)
            scores.append(score)

    return scores

def _get_sent_distinct_1(
    hypothesis: List[str],
    references: Optional[List[List[str]]] = None,
    tags: Optional[List[List[str]]] = None,
    extra_args: Optional[Dict[str, str]] = None
) -> List[float]:
    return _get_sent_distinct(hypothesis, N=1)    

def _get_sent_distinct_2(
    hypothesis: List[str],
    references: Optional[List[List[str]]] = None,
    tags: Optional[List[List[str]]] = None,
    extra_args: Optional[Dict[str, str]] = None
) -> List[float]:
    return _get_sent_distinct(hypothesis, N=2)    


@register_scorer('distinct_1', 'Distinct-1')
class Distinct1Scorer(VizSeqScorer):
    def score(
            self, hypothesis: List[str], references: Optional[List[List[str]]] = None,
            tags: Optional[List[List[str]]] = None
    ) -> VizSeqScore:
        
        references = []

        return self._score_multiprocess_averaged(
            hypothesis, references, tags, sent_score_func=_get_sent_distinct_1
        )

@register_scorer('distinct_2', 'Distinct-2')
class Distinct2Scorer(VizSeqScorer):
    def score(
            self, hypothesis: List[str], references: Optional[List[List[str]]] = None,
            tags: Optional[List[List[str]]] = None
    ) -> VizSeqScore:
        
        references = []

        return self._score_multiprocess_averaged(
            hypothesis, references, tags, sent_score_func=_get_sent_distinct_2
        )
