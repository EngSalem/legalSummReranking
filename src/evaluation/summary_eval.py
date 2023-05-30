from rouge_score import rouge_scorer
from rouge_score import scoring
import sacrebleu
from absl import logging
from bert_score_metric import BertScoreMetric
from BARTScore.bart_score import BARTScorer
from moverscore_v2 import get_idf_dict, word_mover_score
from typing import List, Union, Iterable
from itertools import zip_longest
from collections import defaultdict
import numpy as np


def rouge(targets, predictions, score_keys=None):
  """Computes rouge score.
  Args:
    targets: list of strings
    predictions: list of strings
    score_keys: list of strings with the keys to compute.
  Returns:
    dict with score_key: rouge score across all targets and predictions
  """

  if score_keys is None:
    score_keys = ["rouge1", "rouge2", "rougeLsum"]
  scorer = rouge_scorer.RougeScorer(score_keys)
  aggregator = scoring.BootstrapAggregator()

  def _prepare_summary(summary):
    # Make sure the summary is not bytes-type
    # Add newlines between sentences so that rougeLsum is computed correctly.
    summary = summary.replace(" . ", " .\n")
    return summary

  for prediction, target in zip(predictions, targets):
    target = _prepare_summary(target)
    prediction = _prepare_summary(prediction)
    aggregator.add_scores(scorer.score(target=target, prediction=prediction))
  result = aggregator.aggregate()
  for key in score_keys:
    logging.info(
        "%s = %.2f, 95%% confidence [%.2f, %.2f]",
        key,
        result[key].mid.fmeasure*100,
        result[key].low.fmeasure*100,
        result[key].high.fmeasure*100,
    )
  return {key: result[key].mid.fmeasure*100 for key in score_keys}

def bleu(targets, predictions):
  """Computes BLEU score.
  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings
  Returns:
    bleu_score across all targets and predictions
  """
  if isinstance(targets[0], list):
    targets = [[x for x in target] for target in targets]
  else:
    # Need to wrap targets in another list for corpus_bleu.
    targets = [targets]

  bleu_score = sacrebleu.corpus_bleu(predictions, targets,
                                     smooth_method="exp",
                                     smooth_value=0.0,
                                     force=False,
                                     lowercase=False,
                                     tokenize="intl",
                                     use_effective_order=False)
  return {"bleu": bleu_score.score}

def chrfpp(targets, predictions):
    """ Compute CHRF score (character n-gram)
    :param targets:
    :param predictions:
    :return: chrf score (character n-gram overlap)
    """
    score = sacrebleu.corpus_chrf(predictions, targets, 6, 2)
    score_dict = {"chrf": score.score}
    return score_dict

def bertScore(targets, predictions):
    """ Computes BertScore
    :param targets: list of strings or list of list of strings if multiple references are presented
    :param predictions:
    :return:
    """
    bert_metric = BertScoreMetric()
    return bert_metric.evaluate_batch(predictions, targets)

def moverScore(targets, predictions):
    """ Computes mover Score
    :param targets: list of strings or list of list of strings
    :param predictions: predicted summaries
    :return: word mover score
    """
    idf_dict_hyp = get_idf_dict(predictions)  # idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = get_idf_dict(targets)  # idf_dict_ref = defaultdict(lambda: 1.)
    scores = word_mover_score(targets, predictions, idf_dict_ref, idf_dict_hyp,
                              stop_words=[], n_gram=1, remove_subwords=True)
    return {'mover_score': sum(scores)/len(scores)}

def bartScore(targets, predictions):
    """
    :param targets:
    :param predictions:
    :return:
    """
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    score = bart_scorer.score(predictions, targets)
    score_dict = {"bart_score": np.mean(score)}
    return score_dict