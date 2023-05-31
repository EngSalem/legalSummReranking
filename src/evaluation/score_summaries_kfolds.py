import pandas as pd
import argparse
from summ_eval.rouge_metric import RougeMetric
from summ_eval.bert_score_metric import BertScoreMetric
import nltk
import numpy as np
import ast

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-summary_out", type=str)
my_parser.add_argument("-k_folds", type=str)
my_parser.add_argument("-oracle_col",type=str)
my_parser.add_argument("-summ_col", type=str)
my_parser.add_argument("-phrase_summ", type=bool)
args = my_parser.parse_args()

## get rouge

rouge = RougeMetric()
bertscore = BertScoreMetric( model_type='roberta-base')
## read summaries

def get_phrase_summ(list_of_phrase_dicts):
    try:
       dict_list = ast.literal_eval(list_of_phrase_dicts)
       phrases = []
       for dict in dict_list:
           print(dict)
           phrases.extend(dict.keys())
       return '.\n'.join(phrases)    
    except:
       return '' 

#df['Predicted Phrase Summary'] = df.apply(lambda row: get_phrase_summ(row['Predicted Phrase Summary']), axis=1)


rouge_1, rouge_2, rouge_L , BERTscores= [], [], [],[]

for fold in args.k_folds.split(','):
    df_summaries = pd.read_csv(args.summary_out+fold+'.csv')
    print(df_summaries.columns)
    if args.phrase_summ:
       df_summaries['tokenized_summary'] = df_summaries.apply(lambda row: '.\n'.join(nltk.sent_tokenize(row[args.summ_col])), axis=1)
    else:
       df_summaries['summaries'] = df_summaries.apply(lambda row:  get_phrase_summ(row[args.summ_col]), axis=1)
       df_summaries['tokenized_summary'] = df_summaries.apply(lambda row: '.\n'.join(nltk.sent_tokenize(row['summaries'])), axis=1)

    df_summaries['tokenized_oracle'] = df_summaries.apply(lambda  row: '.\n'.join(nltk.sent_tokenize(row[args.oracle_col])), axis=1)
    rouge_dict = rouge.evaluate_batch(df_summaries['tokenized_summary'].tolist(), df_summaries['tokenized_oracle'].tolist())

    
    rouge_1.append(rouge_dict['rouge']['rouge_1_f_score'])
    rouge_2.append(rouge_dict['rouge']['rouge_2_f_score'])
    rouge_L.append(rouge_dict['rouge']['rouge_l_f_score'])
    ## compute BERTscore

    bertscore_dict = bertscore.evaluate_batch(df_summaries['tokenized_summary'].tolist(), df_summaries['tokenized_oracle'].tolist())
    BERTscores.append(bertscore_dict['bert_score_f1'])


print({'rouge 1': np.mean(rouge_1),
        'rouge 2': np.mean(rouge_2),
        'rouge L': np.mean(rouge_L)})

print({'bertScore-f1': np.mean(BERTscores)})

