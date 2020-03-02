# Document Similarity Benchmark

## Usage

### The Standard Benchmark

Running the standard benchmark is done in the following fashion.

1.  Run `python benchmark.py -r arguments`
    -   The arguments are:
        -   no argument: run all algorithms
        -   `bow`: Bag-of-Words model
        -   `bow_j`: Bag-of-Words model using Jaccard similarity as metric
        -   `bow_l2`: Bag-of-Words model using Euclidean similarity as metric
        -   `bow_s`: Bag-of-Words model using stopword filtering
        -   `bow_j_s`: Bag-of-Words model using stopword filtering and Jaccard similarity as metric
        -   `bow_l2_s`: Bag-of-Words model using stopword filtering and Euclidean similarity as metric
        -   `bow_l`: Bag-of-Words model using lemmatization
        -   `bow_ls`: Bag-of-Words model using lemmatization and stopword filtering
        -   `bow_j_l`: Bag-of-Words model using lemmatization and Jaccard similarity as metric
        -   `bow_j_ls`: Bag-of-Words model using lemmatization, stopword filtering and Jaccard similarity as metric
        -   `bow_l2_l`: Bag-of-Words model using lemmatization and Euclidean similarity as metric
        -   `bow_l2_ls_`: Bag-of-Words model using lemmatization, stopword filtering and Euclidean similarity as metric
        -   `spacy_w2v`: SpaCy's Word2Vec model
        -   `spacy_bert`: SpaCy transformer's BERT model
        -   `gensim_wmd`: Gensim's Word2Vec model and gensim's word mover's distance as metric
        -   `gensim_d2v`: Gensim's Doc2Vec model
2.  Results will be saved into: `./data/results.json`

### A Custom Benchmark

1.  Create two files:
    1.  One for training
    2.  One for testing

    -   The first line of each file is assumed to be column names and will be discarded.
    -   Each file needs at least two entries.
    -   3 tab separated columns are necessary:
        -   sentence 1
        -   sentence 2
        -   similarity score

2.  run either:
    1.  `python benchmark.py -custom `“`path_to_train_file`”` `“`path_to_test_file`” to run the benchmark only on the custom dataset.
    2.  `python benchmark.py -r -custom `“`path_to_train_file`”` `“`path_to_test_file`” to run the benchmark on the standard and the custom datasets.

### Visualisation

Once a benchmark has been run `python visualize.py` can be run to create the figures seen in the [results](Document_Similarity_Benchmark#Results "wikilink") section. The figures will be saved in the `./figs/`-subfolder.


## Documentation

See wiki
