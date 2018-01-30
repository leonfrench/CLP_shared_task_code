# CLP_shared_task_code
CLPsych shared task feature generation code for the 2017 submissions by Derek Howard, Jacob Ritchie, Geoffrey Woollard and Leon French. This repository contains no task data, only source code. 


Python3 dependences:
gensim 
numpy 
nltk
empath
pandas
beautifulsoup4
You can install these by running pip3 install requirements.txt

1. Set data directory in config.py if the data is not in the data/raw/ folder. Also, set number of cores to use. 
2. Run python3 make_dataset.py to process the posts in raw data and produce all_posts_data.csv in interim folder.
3. Run build_features.py to generate processed_features.csv
4. Run topic_model.py to get the LDA topic features
5. Run make_doc2vec.py to generate doc2vec features.


After setting it up, the directory tree will look like:
```
.
├── build_features.py
├── config.py
├── data
│   ├── interim
│   │   ├── all_posts_data.csv
│   │   ├── empath_features.csv
│   │   ├── processed_features.csv
│   │   ├── processed_features_LDA.csv
│   │   └── processed_features_plus_doc2vec.csv
│   └── raw
│       ├── clpsych16-data (unzip the data here)
│       ├── clpsych17-data (unzip the data here)
│       ├── clpsych17-test (unzip the data here)
│       └── clpsych17-test-labels.tsv
├── feature_groups.py
├── make_dataset.py
├── make_doc2vec.py
└── topic_model.py
```
