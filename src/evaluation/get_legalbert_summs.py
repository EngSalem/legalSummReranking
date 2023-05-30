import pandas as pd
import numpy as np

for fold in [1, 2, 3, 4, 5]:
    ## get predictions
    test_predictions = [tag.strip() for tag in
                        open(f'../../data/IRC_classification/legal_bert_predictions_fold{fold}.txt', 'r').readlines()]
    test_df = pd.read_csv(f'../../data/IRC_classification/test_classification_fold{fold}.csv')
    ##
    test_df['predicted_labels'] = test_predictions


    name, generated_summary = [], []
    for _name, _df_grp in test_df.groupby(by='name'):
        name.append(_name)
        generated_summary.append(' '.join(_df_grp[~(_df_grp['predicted_labels'] == 'Non_IRC')]['sentence'].tolist()))

    df =pd.merge(pd.DataFrame(data={'fname':name, 'generated_summary': generated_summary}),
             pd.read_csv(f'../../data/sum_pair/test_set_fold{fold}.csv'), on='fname').rename(columns={'summary':'oracle'})

    df.to_csv(f'../../src/summarization/outputs/irc_summ_fold{fold}.csv', index=False)




