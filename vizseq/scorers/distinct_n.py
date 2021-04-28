#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Author: Tannon Kew
Email: kew@cl.uzh.ch
Date: 28.04.21
"""

from typing import Optional, List, Dict
from vizseq.scorers import register_scorer, VizSeqScorer, VizSeqScore

def get_word_count(hypothesis: List[str]):
    wc = 0
    for hyp in hypothesis:
        wc += len(hyp.split())
    return wc

def compute_distinct_N(
    hypothesis: List[str],
    N: int = 2,
    extra_args: Optional[Dict[str, str]] = None
) -> List[float]:
    """
    Proposed by Li et al. (2016) - http://arxiv.org/abs/1510.03055
    
    Calculates the number of distinct unigrams and bigrams
    in generated responses and scales this by the total
    number of generated tokens to avoid favoring long
    sentences.

    I.e. "the number of distinct n-grams divided by the
    total number of generated words" Martins et al. (2020) http://arxiv.org/abs/2004.02644

    "Distinct-N quantifies the intra-text diversity based on
    distinct n-grams in each text." Choi et al. (2020) http://arxiv.org/abs/2009.09417

    :param hypothesis: a list of hypotheses
    :param N: int, ngram to use for calculating
    :return: float, the metric value
    """

    scores = []

    total_wc = get_word_count(hypothesis)
    
    for hyp in hypothesis:
        hyp_tokens = hyp.split()
        
        if len(hyp_tokens) < N:
            scores.append(0.0)

        else:
            n_grams = [tuple(hyp_tokens[i:i+N]) for i in range(len(hyp_tokens)-N+1)]
            distinct_n_grams = set(n_grams)
            
            score = len(distinct_n_grams) / total_wc
            scores.append(score)

    return scores

def get_sent_distinct_1(
    hypothesis: List[str],
    references: Optional[List[List[str]]] = None,
    tags: Optional[List[List[str]]] = None,
    extra_args: Optional[Dict[str, str]] = None
) -> List[float]:
    return compute_distinct_N(hypothesis, N=1)    

def get_sent_distinct_2(
    hypothesis: List[str],
    references: Optional[List[List[str]]] = None,
    tags: Optional[List[List[str]]] = None,
    extra_args: Optional[Dict[str, str]] = None
) -> List[float]:
    return compute_distinct_N(hypothesis, N=2)    


@register_scorer('distinct_1', 'Distinct-1')
class Distinct1Scorer(VizSeqScorer):
    def score(
            self, hypothesis: List[str], references: Optional[List[List[str]]] = None,
            tags: Optional[List[List[str]]] = None
    ) -> VizSeqScore:
        
        references = []

        return self._score_multiprocess_averaged(
            hypothesis, references, tags, sent_score_func=get_sent_distinct_1
        )

@register_scorer('distinct_2', 'Distinct-2')
class Distinct2Scorer(VizSeqScorer):
    def score(
            self, hypothesis: List[str], references: Optional[List[List[str]]] = None,
            tags: Optional[List[List[str]]] = None
    ) -> VizSeqScore:
        
        references = []

        return self._score_multiprocess_averaged(
            hypothesis, references, tags, sent_score_func=get_sent_distinct_2
        )
