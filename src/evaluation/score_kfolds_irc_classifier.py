from sklearn.metrics import classification_report
import pandas as pd

for fold in [1, 2, 3, 4, 5]:
    ## get predictions
    test_predictions = [tag.strip() for tag in
                        open(f'../../data/IRC_classification/legal_bert_predictions_fold{fold}.txt', 'r').readlines()]

    ## oracle
    oracle_predictions = [tag for tag in list(pd.read_csv(f'../../data/IRC_classification/test_classification_fold{fold}.csv')['IRC_type'])]

    print(classification_report(oracle_predictions, test_predictions))

##############################################################
              Precision recall f

