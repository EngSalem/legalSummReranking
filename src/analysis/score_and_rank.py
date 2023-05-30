import pandas as pd
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

test_file = '../summarization/outputs/egal_led_r1_irc_tags_jsd.csv'

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

df = pd.read_csv(test_file)
df['rouge1'] = df.apply(lambda row: scorer.score(row['oracle'],row['generated_summary'])['rouge1'].fmeasure, axis=1)
df = df.sort_values(by=['rouge1'], ascending=False)
df.to_csv('../summarization/outputs/legal_led_r1_irc_tags_jsd_with_rouge1.csv', index=False)

print(pearsonr(df['rouge1'].tolist(), df['jsd_distance'].tolist()))