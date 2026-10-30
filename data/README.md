---
annotations_creators:
- crowdsourced
language:
- en
- ar
- bn
- fi
- ja
- ko
- ru
- te
language_creators:
- crowdsourced
license:
- mit
multilinguality:
- multilingual
pretty_name: XORQA Reading Comprehension
size_categories:
- '10K<n<100K'
source_datasets:
- extended|wikipedia
task_categories:
- question-answering
task_ids:
- extractive-qa
---

# Dataset Card for "tydi_xor_rc"


## Dataset Description

- Homepage: https://ai.google.com/research/tydiqa
- Paper: https://aclanthology.org/2020.tacl-1.30

### Dataset Summary

[TyDi QA](https://github.com/google-research-datasets/tydiqa) is a question answering dataset covering 11 typologically diverse languages. 
[XORQA](https://github.com/AkariAsai/XORQA) is an extension of the original TyDi QA dataset to also include unanswerable questions, where context documents are only in English but questions are in 7 languages.
[XOR-AttriQA](https://github.com/google-research/google-research/tree/master/xor_attriqa) contains annotated attribution data for a sample of XORQA.
This dataset is a combined and simplified version of the [Reading Comprehension data from XORQA](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/tydi_xor_rc_yes_no_unanswerable.zip) and the [in-English data from XOR-AttriQA](https://storage.googleapis.com/gresearch/xor_attriqa/xor_attriqa.zip).

The code to create the dataset is available on [this Colab notebook](https://colab.research.google.com/drive/14s0FEag5FDr-jqjaVLzlU_0Lv0nXHWNg?usp=sharing).

## Dataset Structure

The dataset contains a train and a validation set, with 15343 and 3011 examples, respectively. Access them with

```py
from datasets import load_dataset
dataset = load_dataset("coastalcph/tydi_xor_rc")
df_train = dataset["train"].to_pandas()
df_val = dataset["validation"].to_pandas()
```

### Data Instances

Description of the dataset columns:

| Column name                  | type        |  Description                                                                                                     |
| -----------                  | ----------- | -----------                                                                                                      |
| lang                     | str         | The language of the question                                                                                |
| question                | str         | The question to answer                                                                                           |
| context           | str         | The context, a Wikipedia paragraph in English that might or might not contain the answer to the question                    | 
| answertable | bool | True if the question can be answered given the context, False otherwise |
| answer_start  | int   | The character index in 'context' where the answer starts. If the question is unanswerable given the context, this is -1  |
| answer   | str   | The answer in English, a span of text from 'context'. If the question is unanswerable given the context, this can be 'yes' or 'no'            |
| answer_inlang   | str   | The answer in the same language as the question, only available for some instances (otherwise, NaN)            |


## Useful stuff

Check out the [datasets ducumentations](https://huggingface.co/docs/datasets/quickstart) to learn how to manipulate and use the dataset. Specifically, you might find the following functions useful:

`dataset.filter`, for filtering out data (useful for keeping instances of specific languages, for example).

`dataset.map`, for manipulating the dataset.

`dataset.to_pandas`, to convert the dataset into a pandas.DataFrame format.

## Citations
```
@article{clark-etal-2020-tydi,
    title = "{T}y{D}i {QA}: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages",
    author = "Clark, Jonathan H.  and
      Choi, Eunsol  and
      Collins, Michael  and
      Garrette, Dan  and
      Kwiatkowski, Tom  and
      Nikolaev, Vitaly  and
      Palomaki, Jennimaria",
    editor = "Johnson, Mark  and
      Roark, Brian  and
      Nenkova, Ani",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "8",
    year = "2020",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2020.tacl-1.30",
    doi = "10.1162/tacl_a_00317",
    pages = "454--470",
    abstract = "Confidently making progress on multilingual modeling requires challenging, trustworthy evaluations. We present TyDi QA{---}a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs. The languages of TyDi QA are diverse with regard to their typology{---}the set of linguistic features each language expresses{---}such that we expect models performing well on this set to generalize across a large number of the world{'}s languages. We present a quantitative analysis of the data quality and example-level qualitative linguistic analyses of observed language phenomena that would not be found in English-only corpora. To provide a realistic information-seeking task and avoid priming effects, questions are written by people who want to know the answer, but don{'}t know the answer yet, and the data is collected directly in each language without the use of translation.",
}

@inproceedings{asai-etal-2021-xor,
    title = "{XOR} {QA}: Cross-lingual Open-Retrieval Question Answering",
    author = "Asai, Akari  and
      Kasai, Jungo  and
      Clark, Jonathan  and
      Lee, Kenton  and
      Choi, Eunsol  and
      Hajishirzi, Hannaneh",
    editor = "Toutanova, Kristina  and
      Rumshisky, Anna  and
      Zettlemoyer, Luke  and
      Hakkani-Tur, Dilek  and
      Beltagy, Iz  and
      Bethard, Steven  and
      Cotterell, Ryan  and
      Chakraborty, Tanmoy  and
      Zhou, Yichao",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.46",
    doi = "10.18653/v1/2021.naacl-main.46",
    pages = "547--564",
    abstract = "Multilingual question answering tasks typically assume that answers exist in the same language as the question. Yet in practice, many languages face both information scarcity{---}where languages have few reference articles{---}and information asymmetry{---}where questions reference concepts from other cultures. This work extends open-retrieval question answering to a cross-lingual setting enabling questions from one language to be answered via answer content from another language. We construct a large-scale dataset built on 40K information-seeking questions across 7 diverse non-English languages that TyDi QA could not find same-language answers for. Based on this dataset, we introduce a task framework, called Cross-lingual Open-Retrieval Question Answering (XOR QA), that consists of three new tasks involving cross-lingual document retrieval from multilingual and English resources. We establish baselines with state-of-the-art machine translation systems and cross-lingual pretrained models. Experimental results suggest that XOR QA is a challenging task that will facilitate the development of novel techniques for multilingual question answering. Our data and code are available at \url{https://nlp.cs.washington.edu/xorqa/}.",
}

@inproceedings{muller-etal-2023-evaluating,
    title = "Evaluating and Modeling Attribution for Cross-Lingual Question Answering",
    author = "Muller, Benjamin  and
      Wieting, John  and
      Clark, Jonathan  and
      Kwiatkowski, Tom  and
      Ruder, Sebastian  and
      Soares, Livio  and
      Aharoni, Roee  and
      Herzig, Jonathan  and
      Wang, Xinyi",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.10",
    doi = "10.18653/v1/2023.emnlp-main.10",
    pages = "144--157",
    abstract = "Trustworthy answer content is abundant in many high-resource languages and is instantly accessible through question answering systems {---} yet this content can be hard to access for those that do not speak these languages. The leap forward in cross-lingual modeling quality offered by generative language models offers much promise, yet their raw generations often fall short in factuality. To improve trustworthiness in these systems, a promising direction is to attribute the answer to a retrieved source, possibly in a content-rich language different from the query. Our work is the first to study attribution for cross-lingual question answering. First, we collect data in 5 languages to assess the attribution level of a state-of-the-art cross-lingual QA system. To our surprise, we find that a substantial portion of the answers is not attributable to any retrieved passages (up to 50{\%} of answers exactly matching a gold reference) despite the system being able to attend directly to the retrieved text. Second, to address this poor attribution level, we experiment with a wide range of attribution detection techniques. We find that Natural Language Inference models and PaLM 2 fine-tuned on a very small amount of attribution data can accurately detect attribution. With these models, we improve the attribution level of a cross-lingual QA system. Overall, we show that current academic generative cross-lingual QA systems have substantial shortcomings in attribution and we build tooling to mitigate these issues.",
}

```
