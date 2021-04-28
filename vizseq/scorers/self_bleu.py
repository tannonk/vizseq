#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Author: Tannon Kew
Email: kew@cl.uzh.ch
Date: 27.11.20
"""

from typing import Optional, List, Dict
from vizseq.scorers import register_scorer, VizSeqScorer, VizSeqScore
from vizseq.scorers.bleu import BLEUScorer
from tqdm import tqdm
import numpy as np


def compute_self_bleu(hyps: List[str]) -> List[float]:

    # initialise new BLEUScorer 
    # NOTE: setting corpus_level=True and n_workers > 1 will error
    bleu_for_self = BLEUScorer(corpus_level=False, sent_level=True, n_workers=1, verbose=False, extra_args=None)

    selfbleu_scores = []
    for i, h in enumerate(hyps):
        hyp_refs = [hyps[0:i]+hyps[i+1:]] # references is List[List[str]]
        h = [h] # hypothesis is List[str]    
        selfbleu = bleu_for_self.score(h, hyp_refs)
        # take the single element in sent-level BLEU score, 
        # i.e. selfbleusent_scores[0]
        # since we always only have 1 hyp sentence, 
        # sent-level BLEU score is a List[float] of len() 1
        selfbleu_scores.append(selfbleu.sent_scores[0])

    return selfbleu_scores

@register_scorer('self_bleu', 'Self-BLEU')
class SelfBLEUScorer(VizSeqScorer):
    def score(
            self, hypothesis: List[str], references: Optional[List[List[str]]] = None,
            tags: Optional[List[List[str]]] = None
    ) -> VizSeqScore:
        
        corpus_score, group_scores, sent_scores = None, None, None
        
        selfbleu_scores = compute_self_bleu(hypothesis)
            
        if self.corpus_level:
            # implement corpus-level score
            corpus_score = np.mean(selfbleu_scores)

        if self.sent_level:
            # implement sentence-level score
            sent_scores = selfbleu_scores
        
        if tags is not None:
            raise NotImplementedError
            # tag_set = self._unique(tags)
            # implement group-level (by sentence tags) score
            # group_scores={t: 99.9 for t in tag_set}
        
        return VizSeqScore.make(
            corpus_score=corpus_score, sent_scores=sent_scores,
            group_scores=group_scores
        )
