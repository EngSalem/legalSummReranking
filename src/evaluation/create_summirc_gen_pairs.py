import pandas as pd
import argparse
my_parser = argparse.ArgumentParser()
my_parser.add_argument("-k_folds", type=str)
my_parser.add_argument("-summary_out", type=str)

args = my_parser.parse_args()



df_full = pd.concat([pd.read_csv('../../data/IRC_classification/summ_articles_train.csv'),
                     pd.read_csv('../../data/IRC_classification/summ_articles_validation.csv'),
                     pd.read_csv('../../data/IRC_classification/summ_articles_test.csv')])

fnames, oracl_ircs, oracles = [], [], []
for  name , df_tmp in df_full.groupby(by=['name']):
     fnames.append(name)
     oracl_ircs.append(' '.join(df_tmp[df_tmp['IRC_type'] != 'Non_IRC']['sentence'].tolist()))
     oracles.append(' '.join(df_tmp['sentence'].tolist()))

df_summ = pd.DataFrame(data={'name': fnames, 'irc_oracle': oracl_ircs, 'oracle': oracles})
###########################################################################################

for fold in args.k_folds.split(','):
    df_out_summ = pd.read_csv(args.summary_out +fold+'.csv')
    df_fold_summary = pd.merge(df_out_summ,  df_summ, on='oracle')
    ###############

    df_fold_summary.drop_duplicates().to_csv(args.summary_out+'_with_irc_summary_'+ fold+'.csv', index=False)
