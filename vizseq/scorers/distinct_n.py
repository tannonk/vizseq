#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Author: Tannon Kew
Email: kew@cl.uzh.ch
Date: 28.04.21
"""

from typing import Optional, List, Dict
from vizseq.scorers import register_scorer, VizSeqScorer, VizSeqScore
import numpy as np
from collections import Counter

def compute_distinct_N(
    hypothesis: List[str],
    N: int = 2,
    extra_args: Optional[Dict[str, str]] = None
) -> List[float]:
    """
    Originally proposed by Li et al. (2016) - http://arxiv.org/abs/1510.03055
    
        Calculates the number of distinct unigrams and bigrams
        in generated responses and scales this by the total
        number of generated tokens to avoid favoring long
        sentences.

        I.e. "the number of distinct n-grams divided by the
        total number of generated words" Martins et al. (2020) http://arxiv.org/abs/2004.02644

        NOTE: Choi et al. (2020) http://arxiv.org/abs/2009.09417
        use a version of Distinct-N to quantify the intra-text diversity based on
        distinct n-grams in each text (essentially
        type-token-ratio for ngrams)

    
    This implementation calculates both inter and
    intra distinct-N for a set of hypothesis texts. 
    Instead of normalising by all words (i.e. unigrams), we
    normalise by all ngrams (i.e. unigrams or bigrams).

    NOTE: This implementation is adapted from Baidu's implementation
    (https://github.com/PaddlePaddle/models/blob/release/1.6/PaddleNLP/Research/Dialogue-PLATO/plato/metrics/metrics.py)

    :param hypothesis: a list of hypotheses
    :param N: int, ngram to use for calculating
    :return: float, the metric value
    """

    # sentence-level score, i.e. 
    # distinct(Ngrams in sentence) / # all(ngrams in sentence)
    intra_dist = []

    all_ngrams = Counter()
    for hypo in hypothesis:
        hypo_tokens = hypo.split()
        hypo_ngrams = [tuple(hypo_tokens[i:i+N]) for i in range(len(hypo_tokens)-N+1)]
        hypo_ngrams = Counter(hypo_ngrams)
        
        intra_dist.append((len(hypo_ngrams)+1e-12) / (len(hypo_tokens)+1e-5))

        all_ngrams.update(hypo_ngrams)
    
    inter_dist = (len(all_ngrams)+1e-12) / (sum(all_ngrams.values())+1e-5)
    
    # NOTE: we avoid computing averaging of intra scores
    # here since most other VizSeq scorers return a list of sentence scores
    # intra_dist = np.average(intra_dist)

    return inter_dist, intra_dist


@register_scorer('distinct_1', 'Distinct-1')
class Distinct1Scorer(VizSeqScorer):
    def score(
            self, hypothesis: List[str], references: Optional[List[List[str]]] = None,
            tags: Optional[List[List[str]]] = None
    ) -> VizSeqScore:
        
        inter_dist, intra_dist = compute_distinct_N(hypothesis, N=1)

        return VizSeqScore(
            corpus_score = inter_dist,
            sent_scores = intra_dist
        )

@register_scorer('distinct_2', 'Distinct-2')
class Distinct2Scorer(VizSeqScorer):
    def score(
            self, hypothesis: List[str], references: Optional[List[List[str]]] = None,
            tags: Optional[List[List[str]]] = None
    ) -> VizSeqScore:
        
        inter_dist, intra_dist = compute_distinct_N(hypothesis, N=2)

        return VizSeqScore(
            corpus_score = inter_dist,
            sent_scores = intra_dist
        )
