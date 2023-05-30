import pandas as pd
from nltk.tokenize import word_tokenize

dfs = []
for fold in range(1,6):
    dfs.append(pd.read_csv(f'../../data/sum_pair/train_set_fold{fold}.csv'))

full_data = pd.concat(dfs)
full_data['article_token_length'] = full_data.apply(lambda row: len(word_tokenize(row['article'])), axis=1)

full_data['summary_token_length'] = full_data.apply(lambda row: len(word_tokenize(row['summary'])), axis=1)

##############3

print('Min article length', full_data['article_token_length'].min())
print('Max article length', full_data['article_token_length'].max())
print('Mean article length', full_data['article_token_length'].mean())

#####

print('Min summary length', full_data['summary_token_length'].min())
print('Max summary length', full_data['summary_token_length'].max())
print('Mean summary length', full_data['summary_token_length'].mean())

