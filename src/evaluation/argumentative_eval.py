import pandas as pd
from rouge_score import rouge_scorer
import numpy as np

fold='4'
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
## get predictions
test_predictions = [tag.strip() for tag in
                    open(f'../../data/IRC_classification/legal_bert_predictions_fold{fold}.txt', 'r').readlines()]
test_df = pd.read_csv(f'../../data/IRC_classification/test_classification_fold{fold}.csv')
##
test_df['predicted_labels'] = test_predictions
##
predicted_ircs, name = [], []


def argumentative_ensemble(summaries, scores):
    return summaries[np.argmax(scores)]


for _name, _df_grp in test_df.groupby(by='name'):
    name.append(_name)
    predicted_ircs.append(' '.join(_df_grp[~(_df_grp['predicted_labels'] == 'Non_IRC')]['sentence'].tolist()))

df_ircs_preds = pd.read_csv(f'../../src/summarization/outputs/test_irc_tags_led_fold{fold}.csv')
df_raw = pd.read_csv(f'../../src/summarization/outputs/test_raw_led_fold{fold}.csv')
df_binary_preds = pd.read_csv(f'../../src/summarization/outputs/test_set_binary_tags_fold{fold}.csv')

print(df_ircs_preds.columns)
df_ircs_preds = df_ircs_preds.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}), on='name')
df_ircs_raw = df_raw.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}), on='name')
df_binary_preds = df_binary_preds.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}), on='name')

df_ircs_preds['rouge1_ircs_led_ircs'] = df_ircs_preds.apply(
    lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
df_ircs_preds['rouge1_summs_led_ircs'] = df_ircs_preds.apply(
    lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)

df_ircs_raw['rouge1_ircs_led_raw'] = df_ircs_raw.apply(
    lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
df_ircs_raw['rouge1_summs_led_raw'] = df_ircs_raw.apply(
    lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)

df_binary_preds['rouge1_ircs_led_binary'] = df_binary_preds.apply(
    lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
df_binary_preds['rouge1_summs_led_binary'] = df_binary_preds.apply(
    lambda row: scorer.score(row['oracle'], row['generated_summary'])['rouge1'].fmeasure, axis=1)

############merge all############
df_ircs_preds = df_ircs_preds.rename(columns={'generated_summary': 'generated_summary_ircs'})
df_ircs_raw = df_ircs_raw.rename(columns={'generated_summary': 'generated_summary_raw'})
df_ircs_binary = df_binary_preds.rename(columns={'generated_summary': 'generated_summary_binary'})
######################################
df_test_full = df_ircs_preds.merge(df_ircs_raw, on='name').merge(df_ircs_binary, on='name')

df_test_full['generated_summary'] = df_test_full.apply(lambda row: argumentative_ensemble(
    [row['generated_summary_ircs'], row['generated_summary_raw'], row['generated_summary_binary']],
    [row['rouge1_ircs_led_ircs'], row['rouge1_ircs_led_raw'], row['rouge1_ircs_led_binary']]), axis=1)

df_test_full['rouge1_ensembeled'] = df_test_full.apply(
    lambda row: scorer.score(row['oracle_x'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
print(df_test_full['rouge1_ensembeled'].mean())
df_test_full.to_csv(f'../../src/summarization/outputs/test_ensemble_model_fold{fold}.csv', index=False)
