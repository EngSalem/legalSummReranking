import pandas as pd
import nltk
import numpy as np
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-file", type=str)
my_parser.add_argument("-k_folds", type=str)
my_parser.add_argument("-summary_columns", type=str)
args = my_parser.parse_args()

fold_lengths = []
for fold in args.k_folds.split(','):
    df_fold = pd.read_csv(args.file+fold+'.csv')
    df_fold['summary_length'] = df_fold.apply(lambda row: len(nltk.tokenize.word_tokenize(row[args.summary_columns])), axis=1)

    fold_lengths.append(np.median(df_fold['summary_length'].tolist()))

print('median Summary Length ', np.mean(fold_lengths))

