import pandas as pd
from rouge_score import rouge_scorer
import numpy as np

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)
rank_criteria = 'rouge1'

scorer = rouge_scorer.RougeScorer([rank_criteria], use_stemmer=True)


def return_beam_df_with_scores(base_file_path, fold):
    try:
      df_beam1 = pd.read_csv(f'{base_file_path}_beam_1_fold{fold}.csv')
      df_beam2 = pd.read_csv(f'{base_file_path}_beam_2_fold{fold}.csv')
      df_beam3 = pd.read_csv(f'{base_file_path}_beam_3_fold{fold}.csv')
      df_beam4 = pd.read_csv(f'{base_file_path}_beam_4_fold{fold}.csv')
      df_beam5 = pd.read_csv(f'{base_file_path}_beam_5_fold{fold}.csv')

    except:
        df_beam1 = pd.read_csv(f'{base_file_path}_beam1_fold{fold}.csv')
        df_beam2 = pd.read_csv(f'{base_file_path}_beam2_fold{fold}.csv')
        df_beam3 = pd.read_csv(f'{base_file_path}_beam3_fold{fold}.csv')
        df_beam4 = pd.read_csv(f'{base_file_path}_beam4_fold{fold}.csv')
        df_beam5 = pd.read_csv(f'{base_file_path}_beam5_fold{fold}.csv')

    ##########################

    df_beam1 = df_beam1.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                              on='name')
    df_beam2 = df_beam2.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                              on='name')
    df_beam3 = df_beam3.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                              on='name')
    df_beam4 = df_beam4.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                              on='name')
    df_beam5 = df_beam5.merge(pd.DataFrame(data={'name': name, 'predicted_ircs': predicted_ircs}),
                              on='name')
    ############

    df_beam1['rouge1_ircs_led_ircs_beam1'] = df_beam1.apply(
        lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)
    df_beam1['rouge1_summs_led_ircs'] = df_beam1.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)

    df_beam2['rouge1_ircs_led_ircs_beam2'] = df_beam2.apply(
        lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)
    df_beam2['rouge1_summs_led_ircs'] = df_beam2.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)

    df_beam3['rouge1_ircs_led_ircs_beam3'] = df_beam3.apply(
        lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)
    df_beam3['rouge1_summs_led_ircs'] = df_beam3.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)

    df_beam4['rouge1_ircs_led_ircs_beam4'] = df_beam4.apply(
        lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)
    df_beam4['rouge1_summs_led_ircs'] = df_beam4.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)

    df_beam5['rouge1_ircs_led_ircs_beam5'] = df_beam5.apply(
        lambda row: scorer.score(row['predicted_ircs'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)
    df_beam5['rouge1_summs_led_ircs'] = df_beam5.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary'])[rank_criteria].fmeasure, axis=1)

    ############merge all############
    df_beam1 = df_beam1.rename(columns={'generated_summary': 'generated_summary_beam1'})
    df_beam2 = df_beam2.rename(columns={'generated_summary': 'generated_summary_beam2'})
    df_beam3 = df_beam3.rename(columns={'generated_summary': 'generated_summary_beam3'})
    df_beam4 = df_beam4.rename(columns={'generated_summary': 'generated_summary_beam4'})
    df_beam5 = df_beam5.rename(columns={'generated_summary': 'generated_summary_beam5'})

    return df_beam1, df_beam2, df_beam3, df_beam4, df_beam5


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

    ######################################
    df_ircs_preds_beam1, df_ircs_preds_beam2, df_ircs_preds_beam3, df_ircs_preds_beam4, df_ircs_preds_beam5 = return_beam_df_with_scores(
        '../../src/summarization/outputs/test_set_pred_argLEDAug_irc_tags', fold)

    df_test_full_ircs = df_ircs_preds_beam1.merge(df_ircs_preds_beam2, on='name').merge(df_ircs_preds_beam3,
                                                                                   on='name').merge(df_ircs_preds_beam4, on='name').merge(df_ircs_preds_beam5, on='name')

    df_test_full_ircs['generated_summary_ircs'] = df_test_full_ircs.apply(lambda row: argumentative_ensemble(
        [row['generated_summary_beam1'], row['generated_summary_beam2'], row['generated_summary_beam3'],
         row['generated_summary_beam4'], row['generated_summary_beam5']],
        [row['rouge1_ircs_led_ircs_beam1'], row['rouge1_ircs_led_ircs_beam2'], row['rouge1_ircs_led_ircs_beam3'],
         row['rouge1_ircs_led_ircs_beam4'],
         row['rouge1_ircs_led_ircs_beam5']]), axis=1)

    df_test_full_ircs['rouge1_ensembeled_ircs'] = df_test_full_ircs.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary_ircs'])[rank_criteria].fmeasure, axis=1)

    #########################################################

    df_binary_preds_beam1, df_binary_preds_beam2, df_binary_preds_beam3, df_binary_preds_beam4,\
    df_binary_preds_beam5 = return_beam_df_with_scores('../../src/summarization/outputs/test_set_pred_argLEDAug_binary', fold)

    df_test_full_binary = df_binary_preds_beam1.merge(df_binary_preds_beam2, on='name').merge(df_binary_preds_beam3,
                                                                                              on='name').merge(df_binary_preds_beam4,
        on='name').merge(df_binary_preds_beam5,
                         on='name')

    df_test_full_binary['generated_summary_binary'] = df_test_full_binary.apply(lambda row: argumentative_ensemble(
        [row['generated_summary_beam1'], row['generated_summary_beam2'], row['generated_summary_beam3'],
         row['generated_summary_beam4'], row['generated_summary_beam5']],
        [row['rouge1_ircs_led_ircs_beam1'], row['rouge1_ircs_led_ircs_beam2'], row['rouge1_ircs_led_ircs_beam3'],
         row['rouge1_ircs_led_ircs_beam4'],
         row['rouge1_ircs_led_ircs_beam5']]), axis=1)

    df_test_full_binary['rouge1_ensembeled_binary'] = df_test_full_binary.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary_binary'])[rank_criteria].fmeasure, axis=1)

    ###########################################################

    df_raw_preds_beam1, df_raw_preds_beam2, df_raw_preds_beam3, df_raw_preds_beam4,\
    df_raw_preds_beam5 = return_beam_df_with_scores('../../src/summarization/outputs/test_set_raw_combined', fold)

    df_test_full_raw = df_raw_preds_beam1.merge(df_raw_preds_beam2, on='name').merge(df_raw_preds_beam3,
                                                                                              on='name').merge(
        df_raw_preds_beam4,
        on='name').merge(df_raw_preds_beam5,
                         on='name')

    df_test_full_raw['generated_summary_raw'] = df_test_full_raw.apply(lambda row: argumentative_ensemble(
        [row['generated_summary_beam1'], row['generated_summary_beam2'], row['generated_summary_beam3'],
         row['generated_summary_beam4'], row['generated_summary_beam5']],
        [row['rouge1_ircs_led_ircs_beam1'], row['rouge1_ircs_led_ircs_beam2'], row['rouge1_ircs_led_ircs_beam3'],
         row['rouge1_ircs_led_ircs_beam4'],
         row['rouge1_ircs_led_ircs_beam5']]), axis=1)


    df_test_full_raw['rouge1_ensembeled_raw'] = df_test_full_raw.apply(
        lambda row: scorer.score(row['oracle'], row['generated_summary_raw'])[rank_criteria].fmeasure, axis=1)
    ##############################################################

    df_test_final = df_test_full_ircs.merge(df_test_full_binary, on='name').merge(df_test_full_raw, on='name')

    print(df_test_final.columns)

    df_test_final['generated_summary'] = df_test_final.apply(lambda row: argumentative_ensemble([row['generated_summary_raw'],
                                                                                                 row['generated_summary_binary'],
                                                                                                 row['generated_summary_ircs']],
                                                             [row['rouge1_ensembeled_raw'],row['rouge1_ensembeled_binary'],row['rouge1_ensembeled_ircs']]), axis=1)



    df_test_final = df_test_final[['generated_summary', 'oracle']]

    # print(df_test_full['rouge1_ensembeled'].mean())
    df_test_final.drop_duplicates().to_csv(f'../../src/summarization/outputs/test_full_voted_beams_argLED_{rank_criteria}_pred_model_fold{fold}.csv', index=False)
