## author: Mohamed Salem Elaraby
## mail: mse30@pitt.edu


import pandas as pd
import summary_eval as summEval
import json

####### Evaluate Baseline #####
## read validation and test output

extBERT_valid = pd.read_csv('../summarization/eval_results/extbert_valid_out.csv')
extBERT_test = pd.read_csv('../summarization/eval_results/extbert_test_out.csv')

scores = {}


## compute rouge score
rouge_score_valid = summEval.rouge(extBERT_valid['decoded_summary'].tolist(),extBERT_valid['summary'].tolist())
rouge_score_test = summEval.rouge(extBERT_test['decoded_summary'].tolist(), extBERT_test['summary'].tolist())

## compute bleu score
bleu_score_valid = summEval.bleu(extBERT_valid['decoded_summary'].tolist(), extBERT_valid['summary'].tolist())
bleu_score_test = summEval.bleu(extBERT_test['decoded_summary'].tolist(), extBERT_test['summary'].tolist())

## compute bert_score
bert_score_valid = summEval.bertScore(extBERT_valid['decoded_summary'].tolist(), extBERT_valid['summary'].tolist())
bert_score_test = summEval.bertScore(extBERT_test['decoded_summary'].tolist(), extBERT_test['summary'].tolist())

scores['validation'] = {'rouge': rouge_score_valid,
                        'bleu': bleu_score_valid,
                        'bertscore': bert_score_valid}

scores['test'] = {'rouge': rouge_score_test,
                  'bleu': bleu_score_test,
                  'bertscore': bert_score_test}


with open('./evaluation_outputs/extractive_bert_baseline_results.json', 'w') as fp:
    json.dump(scores, fp)