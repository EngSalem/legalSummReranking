from operator import ge
import rouge
import pandas as pd 

class Dataset:
    def __init__(self, name='test'):
        self.article_files = pd.read_csv("../../dataset/NSF_summarization_dataset/1049full_position.csv")
        self.summ_files = pd.read_csv("../../dataset/NSF_summarization_dataset/1049summ_position.csv")
        if name == "all":
            self.test_names = sorted(list(set(list(self.article_files['name']))))
        else:
            self.test_names = sorted(list(set(list(pd.read_csv("../../dataset/summ_src_tgt_2022/%s_summ.csv"%name)['name']))))
    
    def get_doc_list(self):
        all_article = []
        all_summ = []
        no_duplicate_names = []
        for name in self.test_names:
            if name in no_duplicate_names:
                continue
            no_duplicate_names.append(name)
        
            article = self.article_files[self.article_files['name'] == name]['sentence']
            summary = self.summ_files[self.summ_files['name'] == name]['sentence']
            all_article.append(article)
            all_summ.append(summary)
        return all_article, all_summ
        

class OracleSummarizer:
    def __init__(self,
                 metric: str = 'rouge-l',
                 prf: str = 'f',
                 num_words: int = 200,
                 stay_under_num_words: bool = False
                 ):
        self.evaluator = rouge.Rouge(metrics=[metric])
        self.metric = metric
        self.prf = prf
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words


    def get_summary(self, article, ref):
        # build dictionaries for easy book-keeping
        sentences = {}
        indices = {}
        global_idx = 0
        
        for sentence in (article):
            sentences[global_idx] = sentence
            indices[global_idx] = global_idx
            global_idx += 1
    
        ref = "\n".join(ref) # reference summary
        # print(ref)
        c = ""  # candidate summary
        summary = []
        num_words = 0
        selected_indices = []
        
        while True:
            scores  = {}
            for i,s in sentences.items():
                # print([f'{c}{s}\n'])
                # print(self.evaluator.get_scores([f'{c}{s}\n'],[ref]))
                scores[i] = self.evaluator.get_scores([f'{c}{s}\n'],[ref])[0][self.metric][self.prf]
                
            i = max(scores, key=scores.get)
            sentence = sentences.pop(i)
            c = f'{c}{sentence}\n'
            sentence_indices = indices.pop(i)
            num_words += len(sentence.split())
            if self.stay_under_num_words and num_words > self.num_words:
                break
            summary.append(sentence.replace("\n", " "))
            selected_indices.append(str(sentence_indices))
            if num_words >= self.num_words:
                break
            if len(sentences) == 0:
                break
        return "|".join(summary), ref, selected_indices 

def clean(line_list):
    return [y.replace("\n", " ") for y in line_list]

def generate_shuffled_reference():
    test_names = list(pd.read_csv("../../dataset/summ_src_tgt_2022/test_summ.csv")['name'])
    summ_files = pd.read_csv("../../dataset/NSF_summarization_dataset/1049summ_position.csv")
    
    bart_docs = pd.read_csv("../../dataset/summ_src_tgt_2022/test_output_original_beam_2.csv")['oracle']
    
    bart_sentence = list(bart_docs)
    reindex = []
    # recover the order reference for bart models.
    all_test_sentences = []
    
    no_duplicate_names = []
    shffuled_test_sentences = []

    for name in test_names:
        if name in no_duplicate_names:
            continue
        no_duplicate_names.append(name)
        summary = summ_files[summ_files['name'] == name]['sentence']
        g = " ".join(summary)
        all_sents = "\n".join(summary)
        
        all_test_sentences.append(g)
        shffuled_test_sentences.append(all_sents)
    
    new_index = []
    for query in bart_sentence:
        idx = all_test_sentences.index(query)
        new_index.append(idx)
    print(new_index)
    
    final = []
    for idx in new_index:
        final.append(shffuled_test_sentences[idx])
    test_df = pd.DataFrame()
    # valid_df = pd.DataFrame()
    test_df['oracle'] = final  
    test_df.to_csv("../../dataset/summ_src_tgt_2022/test_references_shuffled.csv") 
     
def verify_extractive_oracle():
    test_indexes = list(pd.read_csv("../../dataset/summ_src_tgt_2022/test_output_1049_extractive_oracle.csv")['indexes'])
    test_docs = pd.read_csv("../../dataset/NSF_summarization_dataset/1049full_position.csv")
    
    test_names = sorted(list(set(list(test_docs['name']))))
    no_duplicate_name = []
    
    found = 0
    all = 0
    for name in test_names:
        if name not in no_duplicate_name:
            
            
            labels = list(test_docs[test_docs['name'] == name]['IRC_type'])
            # print((labels))
            found_indexes = [int(y) for y in test_indexes[len(no_duplicate_name)].split(" | ")]
            IRC_indexes = [item for item in found_indexes if labels[item] != "Non_IRC"]
            print(IRC_indexes, len(found_indexes), ([labels[item]for item in found_indexes] ))
            no_duplicate_name.append(name)
            found += len(IRC_indexes)
            all += len([g for g in labels if g != "Non_IRC"])
    print(found, all, found / all)
    
if __name__ == "__main__":
    # verify_extractive_oracle()
    # data = Dataset(name='valid')
    # all_article, all_summ= data.get_doc_list()
    
    # model = OracleSummarizer()
    
    # all_generated = []
    # all_refs = []
    # all_indexes = []
    # count = 0
    # for a,b in zip(all_article, all_summ):
    #     predicted, ref, indexes = model.get_summary(a, b)
    #     all_indexes.append(indexes)
    #     print(indexes)
    #     all_generated.append(predicted)
    #     all_refs.append(ref)
    #     print(count+1)
    #     count += 1
    # new_dataframe = pd.DataFrame()
    # new_dataframe['generated_summary'] = all_generated
    # new_dataframe['oracle'] = all_refs
    # new_dataframe['indexes'] = [" | ".join(y) for y in all_indexes]
    # new_dataframe.to_csv("../../dataset/summ_src_tgt_2022/valid_output_extractive_oracle.csv")
    # # generate_shuffled_reference()
    
    
    data = Dataset(name='all')
    all_article, all_summ= data.get_doc_list()
    
    model = OracleSummarizer()
    
    all_generated = []
    all_refs = []
    all_indexes = []
    count = 0
    for a,b in zip(all_article, all_summ):
        predicted, ref, indexes = model.get_summary(a, b)
        all_indexes.append(indexes)
        print(indexes)
        all_generated.append(predicted)
        all_refs.append(ref)
        print(count+1)
        count += 1
    new_dataframe = pd.DataFrame()
    new_dataframe['generated_summary'] = all_generated
    new_dataframe['oracle'] = all_refs
    new_dataframe['indexes'] = [" | ".join(y) for y in all_indexes]
    new_dataframe.to_csv("../../dataset/summ_src_tgt_2022/test_output_1049_extractive_oracle.csv")
    
    # data_path = "/Users/yangzhong/Desktop/Fall_2021/AI_fairness_research/code_process/billsum_v4_1/pubmed_version_section/us_test_data_final_OFFICIAL.jsonl"
    # files = pd.read_json(data_path, lines=True)
    # write_out_file = open("test_billsum_ext.src", "w", encoding='utf-8')
    # all_article = list(files['sections'])
    # all_summ  = list(files['abstract_text'])
    
    # model = OracleSummarizer()
    
    # all_generated = []
    # all_refs = []
    # all_indexes = []
    # count = 0
    # for a,b in zip(all_article, all_summ):
    #     all_sents = []
        
    #     for sec in a:
    #         all_sents += sec 
        
    #     predicted, ref, indexes = model.get_summary(all_sents, b)
    #     all_indexes.append(indexes)
    #     print(indexes)
    #     all_generated.append(predicted)
    #     all_refs.append(ref)
    #     print(count+1)
    #     count += 1
    # new_dataframe = pd.DataFrame()
    # new_dataframe['generated_summary'] = all_generated
    # new_dataframe['oracle'] = all_refs
    # new_dataframe['indexes'] = [" | ".join(y) for y in all_indexes]
    # new_dataframe.to_csv("../../dataset/summ_src_tgt_2022/test_output_billsum_extractive_oracle.csv")
    