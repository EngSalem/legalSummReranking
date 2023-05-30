import pandas as pd
from rouge_score import rouge_scorer
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
penalties = [0.34, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

df_test = pd.read_csv('../../data/IRC_classification/summ_articles_test.csv')

pearson_corrs, kendal_taus = [], []

for penalty in penalties:
    df_with_jsd = pd.read_csv('../summarization/outputs/legal_led_r1_irc_tags_jsd_pen_' + str(penalty) + '.csv')
    oracle, ircs = [], []
    for name, df_name in df_test.groupby(by='name'):
        oracle.append(' '.join(df_name['sentence'].tolist()))
        ircs.append(' '.join(df_name[~(df_name['sentence'] == 'Non_IRC')]['sentence'].tolist()))

    df_full = pd.merge(df_with_jsd, pd.DataFrame(data={'oracle': oracle, 'summ_ircs': ircs}), on='oracle')
    df_full['rouge1_ircs'] = df_full.apply(
        lambda row: scorer.score(row['summ_ircs'], row['generated_summary'])['rouge1'].fmeasure, axis=1)
    df_full['neg_jsd'] = df_full.apply(lambda row: 1 - row['jsd_distance'], axis=1)

    pearson_corrs.append(pearsonr(df_full['jsd_distance'].tolist(), df_full['rouge1_ircs'].tolist())[0])
    kendal_taus.append(kendalltau(df_full['jsd_distance'].tolist(), df_full['rouge1_ircs'].tolist())[0])

data_correlations = pd.DataFrame(data={'correlations': pearson_corrs + kendal_taus,
                                       'penalty range': penalties* 2,
                                       'type': ['pearson'] * len(pearson_corrs) + ['kendal'] * len(kendal_taus)})
sns.lineplot(data=data_correlations, x="penalty range", y="correlations", hue="type")
plt.show()

#print(data_correlations)