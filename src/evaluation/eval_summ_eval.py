import pandas as pd
from summ_eval.rouge_metric import RougeMetric
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output', action='store', dest='output',
                    help='train csv')
args = parser.parse_args()

rouge = RougeMetric()
print('Loading test ...')

df = pd.read_csv(args.output)
summaries = df['generated_summary'].tolist()
references = df['oracle'].tolist()

rouge_dict = rouge.evaluate_batch(summaries, references)

rouge_1 = rouge_dict['rouge']['rouge_1_f_score']
rouge_2 = rouge_dict['rouge']['rouge_2_f_score']
rouge_L = rouge_dict['rouge']['rouge_l_f_score']

print('Scores', {'rouge 1': rouge_1,
                 'rouge 2': rouge_2,
                 'rouge-L': rouge_L})
