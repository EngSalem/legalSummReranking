import pandas as pd
import os

DATA_DIR = '../../data/extractive/'
TEST_DIR = '../../data/sum_pair/'
##############
HipoRank = 'named_hiporank_0622.csv'
HipoRank_reweighted = 'named_theme1_dynamic_1049_no_header.csv'
extractive_bert = 'named_1049_test_output_bert_extract_selection.csv'
##################
Models = {'HipoRank':HipoRank,
          'HipoRank_reweighted': HipoRank_reweighted,
          'ExtractiveBERT': extractive_bert}

##################

for fold in range(1,6):
    for model, file in Models.items():

        df_full_summs = pd.read_csv(os.path.join(DATA_DIR, file))
        pd.merge(pd.read_csv(os.path.join(TEST_DIR, f'test_set_fold{fold}.csv')).drop(columns={'summary'}),
                                df_full_summs, on='fname').to_csv(os.path.join(DATA_DIR, f'{model}_fold{fold}.csv'), index=False)
