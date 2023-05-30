import pandas as pd 
#import sacrerouge 
from sacrerouge.metrics import BertScore, MoverScore, Meteor #from sacrerouge.metrics.bertscore import BertScore
import os
from bert_score import score
from nltk.tokenize import sent_tokenize
import re

all_files = os.listdir("bertScore_dir/generation/DEC_generations_yes/")

#MV =dd Meteor()
fin =  open("eval_results/"+ "new_result.txt", "w")

def remove_tag(input_str):
    pattern1 = re.sub(r"<sent.+sep( )?(>|]|\)|»|;)", "", input_str)
    pattern2 = re.sub(r".+sep ", "", pattern1)
    pattern3 = re.sub(r"<sent()?", "", pattern2)
    pattern4 = re.sub(r"<\D+(>|]) ", "", pattern3)
    pattern5 = re.sub(r"Sent ", "", pattern4)
    pattern6 = re.sub(r"<text.*>", "", pattern5)
    pattern7 = re.sub(r"<", "", pattern6)
    return pattern7

for fname in all_files:
    #if not "trained" in fname:
         #   continue
    try:
        filein = pd.read_csv("bertScore_dir/generation/DEC_generations_yes/" + fname)
    except:
        continue
    raw_summary = list(filein['generated_summary'])
    summary = []
    for item in raw_summary:
        item = remove_tag(item)
        summary.append("\n".join(sent_tokenize(remove_tag(item))))

    refs = list(filein['oracle'])
    refs = ["\n".join(sent_tokenize(y)) for y in refs]
#print(refs)

#rouge = Rouge()
#print(rouge.evaluate(summary, refs)[0])
    print("LOAD -- > ", fname)
    P, R, F1 = score(summary, refs, lang="en", model_type="roberta-large", verbose=True)
    fin.write(fname+"\n")
    bs_result = f"System level F1 , P, R are : {F1.mean()*100:.2f}"
    bs_p = f"& {P.mean()*100:.2f}"
    bs_R = f"& {R.mean()*100:.2f}"
    
    #mv_result = MV.evaluate(summary, refs)
    fin.write("BERTScore " +  str(bs_result) + str(bs_p) + str(bs_R) +"\n")
    fin.write("\n")
       # fin.write("Meteor ", str(mv_result))
        #print(f"System level F1 score: {F1.mean():.3f}")
#print(f"System level P score: {P.mean():.3f}")
#print(f"System level Recall score: {R.mean():.3f}")
