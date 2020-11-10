from typing import List, Optional, Dict
from nltk.util import ngrams

# from vizseq.scorers import register_scorer, VizSeqScorer
# from vizseq.scorers import VizSeqScore

# ----------
# DISTINCT-N (adapted from https://github.com/neural-dialogue-metrics/Distinct-N)
# ----------


def get_sentence_distinct(
    hypo: List[str],
    n: int = 2
) -> float:
    """
    Compute distinct-N for a single hypothesis sentence.

    :param sentence: a list of tokens
    :param n: int, ngram.
    :return: float, the metric value.
    """

    if isinstance(hypo, str):
        hypo = hypo.split()

    assert isinstance(hypo, list)

    if len(hypo) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(hypo, n))
    # print(distinct_ngrams, len(sentence))
    return (len(distinct_ngrams) / len(hypo))*100


def calc_sentence_distinct_average(
    hypos: List[str], n: int = 2
) -> float:

    return sum(get_sentence_distinct(hypo, n) for hypo in hypos) / len(hypos)


def get_corpus_distinct(hypos: List[str], n: int = 2) -> float:
    """
    Compute average distinct-N of a list of sentences (the
    corpus).

    Adapted from https://github.com/neural-dialogue-metrics/Distinct-N

    Distinct-N, most notably distinct-1 and distinct-2, is
    metric that measures the diversity of a sentence. It
    focuses on the number of distinct n-gram of a sentence
    and thus penalizes sentences with lots of repeated
    words. The metric is free of any reference or ground
    truth sentence and devotes totally to the property of a
    sentence (generated by the system). 

    (Jiwei Li et.al in the paper A Diversity-Promoting Objective Function for Neural Conversation Models.)


    :param hypos: list of sentences, where each
    sentence is a single string (tokenizing happens here on whitespace)
    :param n: (int) ngram
    :return: (float) score.
    """

    # ensure each hypo is a tokenized list of strings
    if isinstance(hypos[0], str):
        hypos = [hypo.split() for hypo in hypos]

    assert isinstance(hypos[0], list)

    # collect all n-grams in corpus sentences
    corpus_ngrams = [ngram for hypo in hypos for ngram in ngrams(hypo, n)]

    # get total number of unique n-grams
    distinct_corpus_ngrams = set(corpus_ngrams)

    return (len(distinct_corpus_ngrams)/len(corpus_ngrams))*100


# @register_scorer('distinct_1', 'DISTINCT-1')
# class NISTScorer(VizSeqScorer):
#     def score(
#             self, hypothesis: List[str],
#             tags: Optional[List[List[str]]] = None,
#     ) -> VizSeqScore:

#         corpus_score = get_corpus_distinct(hypothesis, 1)
#         sent_scores = [get_sentence_distinct(hyp, 1) for hyp in hypothesis]

#         return VizSeqScore(corpus_score, sent_scores, None)


# @register_scorer('distinct_2', 'DISTINCT-2')
# class NISTScorer(VizSeqScorer):
#     def score(
#             self, hypothesis: List[str],
#             tags: Optional[List[List[str]]] = None,
#     ) -> VizSeqScore:

#         corpus_score = get_corpus_distinct(hypothesis, 2)
#         sent_scores = [get_sentence_distinct(hyp, 2) for hyp in hypothesis]

#         return VizSeqScore(corpus_score, sent_scores, None)


# @register_scorer('distinct_3', 'DISTINCT_3')
# class NISTScorer(VizSeqScorer):
#     def score(
#             self, hypothesis: List[str],
#             tags: Optional[List[List[str]]] = None,
#     ) -> VizSeqScore:
#         # c_score = get_corpus_distinct(hypothesis, 3)
    # s_sorces = [get_sentence_distinct(hyp, 3) for hyp in hypothesis]
    # return c_sore, s_sorces, None

if __name__ == "__main__":
    pass
