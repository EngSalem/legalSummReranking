import pandas as pd
import re

def get_marked_text(text):
    corefs = []
    for t in text.split("<Marked>"):
        if "</Marked>" in t:
            corefs.append(t.split("</Marked>")[0])
    return corefs

def check_coref(marked_ref, article):
    marked2exist = {}
    for marked in marked_ref:
        if marked in article:
           marked2exist[marked] = True
        else:
           marked2exist[marked] = False

    return marked2exist


df_test_annotated = pd.read_excel('../../data/unfactual/test_output_irc_tags_led_annotated.xlsx')
df_test = pd.read_csv('../../data/sum_pair/test_raw.csv')[['summary','article']]
df_test_annotated = df_test_annotated.merge(df_test, left_on='oracle', right_on='summary')

df_test_annotated['marked_references'] = df_test_annotated.apply(lambda row: get_marked_text(row['generated_summary']), axis=1)
df_test_annotated['existreference'] = df_test_annotated.apply(lambda row: check_coref(row['marked_references'],row['article']), axis=1)

df_test_annotated.to_csv('../../data/unfactual/test_annotated_with_markers.csv', index=False)