import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import ngrams
from ast import literal_eval
import numpy as np

sns.set_style('whitegrid')

my_parser = argparse.ArgumentParser()

my_parser.add_argument("-summary_out", type=str)
my_parser.add_argument("-k_folds", type=str)
my_parser.add_argument("-summ_types", type=str)
args = my_parser.parse_args()
########################################################

GRAM_RANGE = [1, 2, 3, 4]

def get_grams(text, gram):
    return [' '.join(ngram) for ngram in ngrams(text.split(), gram)]

def get_ngram_overlap(article_grams, summary_grams):
    return len([gram for gram in summary_grams if gram not in article_grams])/len(article_grams)*100


final_dfs = []
for summ_out, model_type in zip(args.summary_out.split(','), args.summ_types.split(',')):
    full_dfs = []
    for fold in args.k_folds.split(','):
        df_summ_fold = pd.read_csv(summ_out+fold+'.csv')
        df_ref_fold = pd.read_csv(f'../../data/sum_pair/test_set_with_grams_fold{fold}.csv')
        
        try:
           df_summ_fold = df_summ_fold.merge(df_ref_fold, left_on='oracle', right_on='summary')
        except:
           df_summ_fold =  df_summ_fold.merge(df_ref_fold, on='summary')   
        for gram in GRAM_RANGE:
            try:
               df_summ_fold[str(gram)+'_gram_generated_summary'] = df_summ_fold.apply(lambda row: get_grams(row['generated_summary'], gram), axis=1)
            except:
               df_summ_fold[str(gram)+'_gram_generated_summary'] = df_summ_fold.apply(lambda row: get_grams(row['summary'], gram), axis=1)    

        full_dfs.append(df_summ_fold)

    df_full = pd.concat(full_dfs)

    for gram in GRAM_RANGE:
        df_full[f'{gram}_gram_article'] = df_full[f'{gram}_gram_article'].apply(literal_eval)
        df_full[f'{gram}_gram_novel'] = df_full.apply(lambda row:get_ngram_overlap(article_grams=row[f'{gram}_gram_article'],
                                                                      summary_grams=row[f'{gram}_gram_generated_summary']), axis=1)

    data_dict = df_full[[f'{gram}_gram_novel'  for gram in GRAM_RANGE]].to_dict('list')

    for k, v in data_dict.items():
        data_dict[k] = [np.mean(v)]
    
   

    data_dict['model'] = [model_type]
    final_dfs.append(pd.DataFrame.from_dict(data_dict))


final_df = pd.concat(final_dfs)
final_df = pd.melt(final_df, id_vars ='model')
final_df = final_df.rename(columns={'variable':'ngrams'})

sns.factorplot(x = 'model', y='value', 
               hue = 'ngrams',data=final_df, kind='bar')
plt.show()

#sns.barplot(x= [k for k,_ in data_dict.items()], y= [v for _,v in data_dict.items()])

#plt.show()



