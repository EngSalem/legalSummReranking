import pandas as pd


test_df = '../../data/sum_pair/test_raw.csv'
test_df_with_scores = '../../data/sum_pair/test_summs_with_ircs_led_ircs_tags_with_rouge1.csv'


def get_length(summary_text):
    '''
    :param summary_text:
    :return: summary length (in words).
    '''
    ##Todo: substitute string splitter with a proper word tokenizer
    return len(summary_text.strip().split(' '))


df_test_raw = pd.read_csv(test_df)
df_test_scores = pd.read_csv(test_df_with_scores)
##
df_test_raw['article_length'] = df_test_raw.apply(lambda row: get_length(row['article']), axis=1)
df_test_scores = df_test_scores.merge(df_test_raw[['fname','article_length']], left_on='name', right_on='fname')
df_test_scores.to_csv('../../data/sum_pair/test_summs_with_ircs_led_ircs_tags_with_rouge1_and_articles_lens.csv', index=False)