import pandas as pd
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from summ_eval.rouge_metric import RougeMetric
from summ_eval.bert_score_metric import BertScoreMetric
from scipy.stats import pearsonr

rouge = RougeMetric()

def get_rouge_per_example(oracle, model1, model2, model3):
   r_model1 =  rouge.evaluate_example(oracle,model1)['rouge1'].fmeasure
   r_model2 = rouge.evaluate_example(oracle,model2)['rouge1'].fmeasure
   r_model3 = rouge.evaluate_example(oracle,model3)['rouge1'].fmeasure

   return r_model1, r_model2, r_model3

    
    

def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
    of adoption rating."""
    w = np.zeros((N,N))
    O = confusion_matrix(actuals, preds)
    
    for i in range(len(w)): 
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)
    
    act_hist=np.zeros([N])
    for item in actuals: 
        act_hist[item]+=1
    
    pred_hist=np.zeros([N])
    for item in preds: 
        pred_hist[item]+=1
                         
    E = np.outer(act_hist, pred_hist);
    E = E/E.sum();
    O = O/O.sum();
    
    num=0
    den=0
    print(len(w), len(O))
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*O[i][j]
            den+=w[i][j]*E[i][j]
    return (1 - (num/den))



def normalize_rating(rating):
    if rating <=2:
       return 0
    elif rating ==3:
       return 1
    else:
       return 2   


annotations = pd.read_csv("../../data/human_eval/annotations_table.csv")
human_ratings = pd.read_csv("../../data/human_eval/human_annotations.csv")
human_30_samples = pd.read_csv("../../data/human_eval/human_eval_30_samples.csv")

rouge1, rouge2, rougel, entry_ixs = [], [], [],[]
entry_ix = 0
for oracle, model1, model2, model3 in zip(human_30_samples['oracle'].tolist(),
                                          human_30_samples['model1_summary'].tolist(),
                                          human_30_samples['model2_summary'].tolist(),
                                          human_30_samples['model3_summary'].tolist()):
    
    rouge1.append(rouge.evaluate_batch([oracle], [model1])['rouge']['rouge_1_f_score'])
    rouge2.append(rouge.evaluate_batch([oracle], [model1])['rouge']['rouge_2_f_score'])
    rougel.append(rouge.evaluate_batch([oracle], [model1])['rouge']['rouge_l_f_score'])
    entry_ixs.append(entry_ix)
    entry_ix +=1

    rouge1.append(rouge.evaluate_batch([oracle], [model2])['rouge']['rouge_1_f_score'])
    rouge2.append(rouge.evaluate_batch([oracle], [model2])['rouge']['rouge_2_f_score'])
    rougel.append(rouge.evaluate_batch([oracle], [model2])['rouge']['rouge_l_f_score'])
    entry_ixs.append(entry_ix)
    entry_ix +=1

    rouge1.append(rouge.evaluate_batch([oracle], [model3])['rouge']['rouge_1_f_score'])
    rouge2.append(rouge.evaluate_batch([oracle], [model3])['rouge']['rouge_2_f_score'])
    rougel.append(rouge.evaluate_batch([oracle], [model3])['rouge']['rouge_l_f_score'])
    entry_ixs.append(entry_ix)
    entry_ix+=1
    print(entry_ixs)
 
df_rouge = pd.DataFrame(data={'rouge1': rouge1,
                        'rouge2': rouge2,
                        'rougel': rougel,
                        'entry_ix': entry_ixs})


# for oracle, summary in zip(human_ratings['oracle'].tolist(), human_ratings['summaries'].tolist()):
#     print('oracle', oracle)
#     print('Summary', summary)
#     print(rouge.evaluate_batch([oracle], [summary])['rouge']['rouge_1_f_score'])

#human_ratings['rouge-1'] = human_ratings.apply(lambda row: rouge.evaluate_example(row['oracle'], 
#                                                                                 row['summaries'])['rouge1'].fmeasure, axis=1)
usernames = ['ashley_nsf', 'matt', 'morgan gray']

annotations = annotations[['username',
                           'reference_summary_id',
                           'generated_summary_id',
                           'question_id',
                           'annotator_rating',
                           'feedback_comment']]

annotations = annotations[annotations['username'].isin(usernames)]
annotations['normalized_ratings'] = annotations.apply(lambda row: normalize_rating(row['annotator_rating']), axis=1)
# annotations_df = None
# for annotator, annotator_df in annotations.groupby(['username']):
#     if annotations_df is None:
#         annotations_df = annotator_df
#     else:
#         annotations_df = pd.merge(annotations_df, annotator_df, on=['reference_summary_id', 'generated_summary_id', 'question_id'], how='inner')

question_ixs,question_ids, annotator1, annotator2, annotator3, reference_summary_ids = [],[],[],[],[], []


#print(annotations_df.head())
## check for argumentation only
annotations = annotations[annotations['question_id']==2]
for question_info, question_ratings_df in annotations.groupby(['reference_summary_id',
                                                                'generated_summary_id',
                                                                'question_id']):
    if len(list(set(question_ratings_df['username'].tolist()))) == 3:
       question_ixs.append(question_info)
       question_ids.append(question_info[-1])
       reference_summary_ids.append(question_info[0])
       annotator1.append(question_ratings_df[question_ratings_df['username']==usernames[0]]['normalized_ratings'].tolist()[0])
       annotator2.append(question_ratings_df[question_ratings_df['username']==usernames[1]]['normalized_ratings'].tolist()[0])
       annotator3.append(question_ratings_df[question_ratings_df['username']==usernames[2]]['normalized_ratings'].tolist()[0])
    
data = pd.DataFrame(data={'question_ix': question_ixs,
                          'question_id': question_ids,
                          'entry_ix': reference_summary_ids,
                          'annotator1':annotator1,
                          'annotator2': annotator2,
                          'annotator3': annotator3})

print(data.head())
data = data.dropna()
# Convert the annotator columns to listsclear
annotator1 = [int(x)  for x in data['annotator1'].tolist()]
annotator2 = [int(x) for x in data['annotator2'].tolist()]
annotator3 = [int(x) for x in data['annotator3'].tolist()]

# Calculate pairwise Cohen's Kappa between annotators
kappa_1_2 = cohen_kappa_score(annotator1, annotator2)
kappa_1_3 = cohen_kappa_score(annotator1, annotator3)
kappa_2_3 = cohen_kappa_score(annotator2, annotator3)

# Calculate pairwise weighted kappa between annotators
QWK_1_2 = quadratic_kappa(annotator1,annotator2, N=3)
QWK_1_3 = quadratic_kappa(annotator1,annotator3, N=3)
QWK_2_3 = quadratic_kappa(annotator2, annotator3, N=3)

QWK_avg_kappa = (QWK_1_2+ QWK_1_3+ QWK_2_3)/ 3
print("QWK Kappa (Annotator 1 vs Annotator 2):", QWK_1_2)
print("QWK Kappa (Annotator 1 vs Annotator 3):", QWK_1_3)
print("QWK Kappa (Annotator 2 vs Annotator 3):", QWK_2_3)
print("QWK Kappa:", QWK_avg_kappa)


# Average the pairwise Cohen's Kappa values
avg_kappa = (kappa_1_2 + kappa_1_3 + kappa_2_3) / 3

print("Cohen's Kappa (Annotator 1 vs Annotator 2):", kappa_1_2)
print("Cohen's Kappa (Annotator 1 vs Annotator 3):", kappa_1_3)
print("Cohen's Kappa (Annotator 2 vs Annotator 3):", kappa_2_3)
print("Average Cohen's Kappa:", avg_kappa)

data = data.merge(df_rouge, on='entry_ix')
#print(set(data['question_id'].tolist()))
# ## question 1 correlations
# data_q1 = data[data['question_id']==0]
# print(data_q1.head())
data['agg_rating'] = data.apply(lambda row: (row['annotator1']+row['annotator3'])/2, axis=1)
print(data.head())
print('Correlation analysis with rouge-1', pearsonr(data['agg_rating'], data['rouge1']))
print('Correlation analysis with rouge-2', pearsonr(data['agg_rating'], data['rouge2']))
print('Correlation analysis with rouge-L', pearsonr(data['agg_rating'], data['rougel']))

