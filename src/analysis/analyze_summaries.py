## auhor: Mohamed Salem
## email: mse30@pitt.edu

## This script analyzes the summaries of the legal artices.

## read datasets
import pandas as pd

train_summ = pd.read_csv('../../data/sum_pair/train_raw.csv')
valid_summ = pd.read_csv('../../data/sum_pair/valid_raw.csv')
test_summ = pd.read_csv('../../data/sum_pair/test_raw.csv')
############ def analyze summary lengths ####################

def get_summary_length(summary_text):
    '''
    :param summary_text:
    :return: summary length (in words).
    '''
    ##Todo: substitute string splitter with a proper word tokenizer
    return len(summary_text.strip().split(' '))

## concat summaries dataframes and extract lengths
full_summ_df = pd.concat([train_summ, valid_summ, test_summ])
full_summ_df['summary_length'] = full_summ_df.apply(lambda row: get_summary_length(row['summary']), axis=1)
## get ength stats
max_length = full_summ_df['summary_length'].max()
min_length = full_summ_df['summary_length'].min()
median_length = full_summ_df['summary_length'].median()
mean_length = full_summ_df['summary_length'].mean()


## visualize length as a histogram
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



sns.set_style("whitegrid")

sns.histplot(data=full_summ_df, x="summary_length")

stats_text = ['Maximum Summary Length: '+str(max_length), 'Minimum Summary Length: '+str(min_length),
              'Median Summary Length: '+ str(int(median_length)), 'Mean Summary Length: '+str(int(mean_length))]

patches = [mpatches.Patch(color='w', label=stats_text[j]) for j in range(len(stats_text))]

plt.legend(handles=patches)
plt.xlabel('Summary Length Distribution')
plt.ylabel('Summary Length Frequency')
plt.title('Summary Length Histogram')
plt.show()

### Analyze amount of generated vocabulary

def get_vocabulary(text):
    '''
    :param text:
    :return: unique vocabulary list
    '''
    ##Todo: substitute string splitter with a proper word tokenizer
    return list(set(text.strip().split(' ')))

def get_excluded_vocabulary(check_vocabulary_list, reference_vocabulary_list):
    generated_vocabulary = [w for w in check_vocabulary_list if w not in reference_vocabulary_list]
    return generated_vocabulary, len(generated_vocabulary)

added_vocabulary, new_vocab_lens = [], []
for article, summary in zip(full_summ_df['article'].tolist(), full_summ_df['summary'].tolist()):
    article_vocab = get_vocabulary(article)
    summary_vocab = get_vocabulary(summary)
    generated_vocab, vocab_size = get_excluded_vocabulary(summary_vocab, article_vocab)

    added_vocabulary.extend(generated_vocab)
    new_vocab_lens.append(vocab_size)


import numpy as np

sns.histplot(new_vocab_lens)

stats_text = ['Maximum Added Vocabulary size: '+str(max(new_vocab_lens)), 'Minimum Added Vocabulary: '+str(min(new_vocab_lens)),
              'Median Added Vocabulary size: '+str(int(np.median(new_vocab_lens))), 'Mean Added Vocabulary size: '+str(int(np.mean(new_vocab_lens)))]

patches = [mpatches.Patch(color='w', label=stats_text[j]) for j in range(len(stats_text))]

plt.legend(handles=patches)
plt.xlabel('New Vocabulary Size Distribution')
plt.ylabel('Frequency')
plt.title('New Vocabulary Size Histogram')
plt.show()

################################################################
#### Analyze the amount the percentile of IRCS in summaries ####
################################################################
df_summaries = pd.concat([pd.read_csv('../../data/sum_pair/train_summ.csv'),
                          pd.read_csv('../../data/sum_pair/valid_summ.csv'),
                          pd.read_csv('../../data/sum_pair/test_summ.csv')])

issues_percentiles, reasons_percentiles, conclusion_percentiles, nonircs_percentiles = [],[],[],[]

for article_name, summ_df in df_summaries.groupby(by = ['name']):
    issues_percentiles.append(summ_df[summ_df['IRC_type'] == 'Issue'].shape[0]/summ_df.shape[0])
    reasons_percentiles.append(summ_df[summ_df['IRC_type'] == 'Reason'].shape[0]/summ_df.shape[0])
    conclusion_percentiles.append(summ_df[summ_df['IRC_type'] == 'Conclusion'].shape[0]/summ_df.shape[0])
    nonircs_percentiles.append(summ_df[summ_df['IRC_type'] == 'Non_IRC'].shape[0]/summ_df.shape[0])

## plot histogram of distributions
fig, axes = plt.subplots(2, 2)

sns.histplot(x=issues_percentiles, ax=axes[0,0])
axes[0,0].set_title('Issues distribution in summaries')
axes[0,0].set_xlabel('percentiles')

sns.histplot(x=reasons_percentiles, ax=axes[0,1])
axes[0,1].set_title('Reasons distribution in summaries')
axes[0,1].set_xlabel('percentiles')

sns.histplot(x=conclusion_percentiles, ax=axes[1,0])
axes[1,0].set_title('Conclusion distribution in summaries')
axes[1,0].set_xlabel('percentiles')

sns.histplot(x=nonircs_percentiles, ax=axes[1,1])
axes[1,1].set_title('Non-IRCs distribution in summaries')
axes[1,1].set_xlabel('percentiles')

plt.show()


