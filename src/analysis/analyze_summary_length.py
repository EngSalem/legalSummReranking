
import pandas as pd
def get_length(summary_text):
    '''
    :param summary_text:
    :return: summary length (in words).
    '''
    ##Todo: substitute string splitter with a proper word tokenizer
    return len(summary_text.strip().split(' '))


valid_summ = pd.read_csv('~/PhD/LegalDocSummarization/src/summarization/outputs/test_big_model_voted.csv')
## concat summaries dataframes and extract lengths

valid_summ['summary_length'] = valid_summ.apply(lambda row: get_length(row['generated_summary']), axis=1)
valid_summ['article_length'] = valid_summ.apply(lambda row: get_length(row['oracle']), axis=1)
## get ength stats
summary_max_length = valid_summ['summary_length'].max()
summary_min_length = valid_summ['summary_length'].min()
summary_median_length = valid_summ['summary_length'].mean()
summary_mean_length = valid_summ['summary_length'].mean()

## get article length stats

article_max_length = valid_summ['article_length'].max()
article_min_length = valid_summ['article_length'].min()
article_median_length = valid_summ['article_length'].mean()
article_mean_length = valid_summ['article_length'].mean()




print('Mean summary length ', summary_mean_length)
print('Mean article length ', article_mean_length)
print('Max summary length', summary_max_length)
print('Max article length', article_max_length)
print('Min summary length', summary_min_length)
print('Min article length', article_min_length)





