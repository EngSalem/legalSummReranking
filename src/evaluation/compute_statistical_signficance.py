import pandas as pd
import argparse
#from summ_eval.rouge_metric import RougeMetric
#from rouge_score import rouge_scorer
#from summ_eval.bert_score_metric import BertScoreMetric
import nltk
import numpy as np
import os
from summ_eval.rouge_metric import RougeMetric


#scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge = RougeMetric()
def bootstrap_rogue(references, baseline_hypo, new_hypo, num_samples=1000, sample_size=200):
    """
    Original Code adapted from https://github.com/pytorch/translate/tree/master/pytorch_translate
    Attributes:
        references -> the reference data
        baseline_hypo -> the baseline model output
        new_hypo -> new mdoel_hypo
    """
    num_sents = len(baseline_hypo)
    assert len(baseline_hypo) == len(new_hypo) == len(references)

    indices = np.random.randint(
        low=0, high=num_sents, size=(num_samples, sample_size))
    out = {}

    baseline_better = 0
    new_better = 0
    num_equal = 0
    for index in indices:
        sub_base_hypo = [y for i, y in enumerate(
            baseline_hypo) if i in index]
        sub_new_hypo = [y for i, y in enumerate(
            new_hypo) if i in index]
        sub_ref = [y for i, y in enumerate(
            references) if i in index]
        baseline_rogue = rouge.evaluate_batch(sub_ref, sub_base_hypo)['rouge1']['rouge_1_f_score']
        new_rogue = rouge.evaluate_batch(sub_ref, sub_new_hypo)['rouge1']['rouge_1_f_score']

        if new_rogue > baseline_rogue:
            new_better += 1
        elif baseline_rogue > new_rogue:
            baseline_better += 1
        else:
            num_equal += 1
    print("------DOWN ROGUE COMPUTATION------")
    print("{} New better confidence : {:.2f}".format(
        "ROUGUE", new_better/num_samples))
    out = {"new_pvalue": 1 - (new_better/num_samples)}
    return out

##################################
DIR = '../../src/summarization/outputs'
## combine raw folds

dfs = []
for fold in range(1,6):
    dfs.append(pd.read_csv(os.path.join(DIR, f'test_raw_led_fold{fold}.csv')))

raw_test = pd.concat(dfs).drop_duplicates()

## combine our proposed system predictions

dfs  = []
for fold in range(1,6):
    dfs.append(pd.read_csv(os.path.join(DIR, f'test_full_voted_beams_argLED_pred_model_fold{fold}.csv')))

proposed = pd.concat(dfs).drop_duplicates()

###
print(bootstrap_rogue(raw_test['oracle'].tolist(),raw_test['generated_summary'].tolist(), proposed['generated_summary'].tolist()))

