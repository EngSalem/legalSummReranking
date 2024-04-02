# legalSummReranking

* This repository contains the source code for the paper "Towards Argument-Aware Abstractive Summarization of Long Legal
Opinions with Summary Reranking" Accepted for Findings of ACL 2023

## Data

*  To request the annotations of both summaries and articles with argument roles , please contact Dr. Kevin D. Ashley (ashley@pitt.edu).  However, you must first obtain the unannotated data through an agreement with the Canadian Legal Information Institute (CanLII) (https://www.canlii.org/en/)

## Summarization

* The main training , and testing scripts are obtained from [[this repostiory](https://github.com/EngSalem/arglegalsumm)]

## IRC classification
* We used the classification code povided in [[this link](https://github.com/EngSalem/arglegalsumm/tree/master/src/argument_classification)]

## Reranking
* Our reranking script can be found [[here](https://github.com/EngSalem/legalSummReranking/tree/main/src/rankers/)]
* This script leverages multiple beams and multiple folds simultaeously 
* Change the *basefile directories* in the script according to your inputs, and your *final output* to your own choice.

### Reranking is done through ROUGE evaluation
### Requirements to run reranking
- transformers
- pytorch
- SummEval [[link](https://github.com/Yale-LILY/SummEval)]


## Evaluation
* scoring scripts for *rouge , bertscore and botstrapped signficance testing*  can be found [[here](https://github.com/EngSalem/legalSummReranking/tree/main/src/evaluation)]


If you are going to follow up on this project please cite this work using the following bibtext:*
```
@article{elaraby2023towards,
  title={Towards Argument-Aware Abstractive Summarization of Long Legal Opinions with Summary Reranking},
  author={Elaraby, Mohamed and Zhong, Yang and Litman, Diane},
  journal={arXiv preprint arXiv:2306.00672},
  year={2023}
}
```

You can also refer to our previous paper

```
@inproceedings{elaraby-litman-2022-arglegalsumm,
    title = "{A}rg{L}egal{S}umm: Improving Abstractive Summarization of Legal Documents with Argument Mining",
    author = "Elaraby, Mohamed  and
      Litman, Diane",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.540",
    pages = "6187--6194",
    abstract = "A challenging task when generating summaries of legal documents is the ability to address their argumentative nature. We introduce a simple technique to capture the argumentative structure of legal documents by integrating argument role labeling into the summarization process. Experiments with pretrained language models show that our proposed approach improves performance over strong baselines.",
}
```



 


