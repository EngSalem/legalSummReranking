import pandas as pd
from rouge_score import rouge_scorer
import numpy as np

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
## get predictions
test_predictions = [tag.strip() for tag in
                    open('../../data/IRC_classification/legal_bert_predictions_fold1.txt', 'r').readlines()]
test_df = pd.read_csv('../../data/IRC_classification/test_classification_fold1.csv')
##
test_df['predicted_labels'] = test_predictions
##
predicted_ircs, name = [], []


def argumentative_ensemble(summaries, scores):
    return summaries[np.argmax(scores)]


for _name, _df_grp in test_df.groupby(by='name'):
    name.append(_name)
    predicted_ircs.append(' '.join(_df_grp[~(_df_grp['predicted_labels'] == 'Non_IRC')]['sentence'].tolist()))

df_ircs_preds_beam1 = pd.read_csv('../../src/summarization/outputs/test_ircs_tags_led_retested_beam1.csv')
df_ircs_preds_beam2 = pd.read_csv('../../src/summarization/outputs/test_ircs_tags_led_retested_beam2.csv')
df_ircs_preds_beam3 = pd.read_csv('../../src/summarization/outputs/test_ircs_tags_led_retested_beam3.csv')
df_ircs_preds_beam4 = pd.read_csv('../../src/summarization/outputs/test_ircs_tags_led_retested_beam4.csv')
df_ircs_preds_beam5 = pd.read_csv('../../src/summarization/outputs/test_ircs_tags_led_retested_beam5.csv')

df_ircs_preds_beam1 = df_ircs_preds_beam1.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                                                on='name')
df_ircs_preds_beam2 = df_ircs_preds_beam2.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                                                on='name')
df_ircs_preds_beam3 = df_ircs_preds_beam3.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                                                on='name')
df_ircs_preds_beam4 = df_ircs_preds_beam4.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                                                on='name')
df_ircs_preds_beam5 = df_ircs_preds_beam5.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                                                on='name')

df_ircs_preds_beam1['rouge1_ircs_led_ircs_beam1'] = df_ircs_preds_beam1.apply(
    lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
df_ircs_preds_beam1['rouge1_summs_led_ircs'] = df_ircs_preds_beam1.apply(
    lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)

df_ircs_preds_beam2['rouge1_ircs_led_ircs_beam2'] = df_ircs_preds_beam2.apply(
    lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
df_ircs_preds_beam2['rouge1_summs_led_ircs'] = df_ircs_preds_beam2.apply(
    lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)

df_ircs_preds_beam3['rouge1_ircs_led_ircs_beam3'] = df_ircs_preds_beam3.apply(
    lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
df_ircs_preds_beam3['rouge1_summs_led_ircs'] = df_ircs_preds_beam3.apply(
    lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)

df_ircs_preds_beam4['rouge1_ircs_led_ircs_beam4'] = df_ircs_preds_beam4.apply(
    lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
df_ircs_preds_beam4['rouge1_summs_led_ircs'] = df_ircs_preds_beam4.apply(
    lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)

df_ircs_preds_beam5['rouge1_ircs_led_ircs_beam5'] = df_ircs_preds_beam5.apply(
    lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
df_ircs_preds_beam5['rouge1_summs_led_ircs'] = df_ircs_preds_beam5.apply(
    lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)

############merge all############
df_ircs_preds_beam1 = df_ircs_preds_beam1.rename(columns={'generated_summary': 'generated_summary_beam1'})
df_ircs_preds_beam2 = df_ircs_preds_beam2.rename(columns={'generated_summary': 'generated_summary_beam2'})
df_ircs_preds_beam3 = df_ircs_preds_beam3.rename(columns={'generated_summary': 'generated_summary_beam3'})
df_ircs_preds_beam4 = df_ircs_preds_beam4.rename(columns={'generated_summary': 'generated_summary_beam4'})
df_ircs_preds_beam5 = df_ircs_preds_beam5.rename(columns={'generated_summary': 'generated_summary_beam5'})

######################################
df_test_full = df_ircs_preds_beam1.merge(df_ircs_preds_beam2, on='name').merge(df_ircs_preds_beam3, on='name').merge(
    df_ircs_preds_beam4, on='name').merge(df_ircs_preds_beam5, on='name')

print(df_test_full.columns)
df_test_full['generated_summary'] = df_test_full.apply(lambda row: argumentative_ensemble(
    [row['generated_summary_beam1'], row['generated_summary_beam2'], row['generated_summary_beam3'],
     row['generated_summary_beam4'], row['generated_summary_beam5']],
    [row['rouge1_ircs_led_ircs_beam1'], row['rouge1_ircs_led_ircs_beam2'], row['rouge1_ircs_led_ircs_beam3'],
     row['rouge1_ircs_led_ircs_beam4'],
     row['rouge1_ircs_led_ircs_beam5']]), axis=1)

df_test_full['rouge1_ensembeled'] = df_test_full.apply(
   lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
#print(df_test_full['rouge1_ensembeled'].mean())
df_test_full.to_csv('../../src/summarization/outputs/test_full_voted_beams.csv', index=False)
