import pandas as pd
from rouge_score import rouge_scorer
import numpy as np

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)
rank_criteria = 'rougeL'

for fold in [1, 2, 3, 4, 5]:
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

    df_raw_fold = pd.read_csv(f'../../src/summarization/outputs/test_raw_led_fold{fold}.csv')
    df_irc_led_fold = pd.read_csv(f'../../src/summarization/outputs/test_pred_irc_tags_fold{fold}.csv')
    df_binary_led_fold = pd.read_csv(f'../../src/summarization/outputs/test_pred_binary_tags_fold{fold}.csv')

    ############

    df_raw_fold = df_raw_fold.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}), on='name')
    df_irc_led_fold = df_binary_led_fold.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                                               on='name')
    df_binary_led_fold = df_binary_led_fold.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                                                  on='name')
    #########

    df_raw_fold['rouge1_raw'] = df_raw_fold.apply(
        lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)

    df_irc_led_fold['rouge1_irc'] = df_irc_led_fold.apply(
        lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)

    df_binary_led_fold['rouge1_binary'] = df_irc_led_fold.apply(
        lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)

    #########

    ############merge all############
    df_raw_fold = df_raw_fold.rename(columns={'generated_summary': 'generated_summary_raw'})
    df_irc_led_fold = df_irc_led_fold.rename(columns={'generated_summary': 'generated_summary_irc_tags'})
    df_binary_led_fold = df_binary_led_fold.rename(columns={'generated_summary': 'generated_summary_binary_tags'})


    ######################################
    df_test_full = df_raw_fold.merge(df_irc_led_fold, on='name').merge(df_binary_led_fold,
                                                                                   on='name')


    df_test_full['generated_summary'] = df_test_full.apply(lambda row: argumentative_ensemble(
        [row['generated_summary_raw'], row['generated_summary_irc_tags'], row['generated_summary_binary_tags']],
        [row['rouge1_raw'], row['rouge1_irc'], row['rouge1_binary']]), axis=1)

    df_test_full['rouge1_ensembeled'] = df_test_full.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)
    # print(df_test_full['rouge1_ensembeled'].mean())
    df_test_full.to_csv(f'../../src/summarization/outputs/test_full_voted_ensemble_pred_{rank_criteria}_fold{fold}.csv', index=False)
