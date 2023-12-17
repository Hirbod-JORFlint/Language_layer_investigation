## Instructions to reproduce results

Tested with Python 3.9.1. Needs packages: torch (tested with version 
1.8.0), transformers (4.3.3).

### Steps to create representations and train probes: 

1. Specify the BERT model you want to use in word_representation. 
    Our options: 'bert-base-uncased', 'bert-base-german-cased',  
    'TurkuNLP/bert-base-finnish-cased-v1', 'onlplab/alephbert-base', 
    'KB/bert-base-swedish-cased', 'UWB-AIR/Czert-B-base-cased', 
    'dbmdz/bert-base-turkish-cased'.

2. Specify the names for the .conllu files you want to use and
    for the pickle file you want to create in create_objects_ud.
    We used the following: 
    cs: 'cs_pdt-ud-train.conllu', 'cs_pdt-ud-dev.conllu'
    de: 'de_gsd-ud-train.conllu', 'cs_pdt-ud-dev.conllu'
    en: 'en_ewt-ud-train.conllu', 'en_ewt-ud-dev.conllu'
    fi: 'fi_tdt-ud-train.conllu', 'fi_tdt-ud-dev.conllu'
    he: 'he_htb-ud-train.conllu', 'he_htb-ud-dev.conllu'
    sv: 'sv_talbanken-ud-train.conllu', 'sv_talbanken-ud-dev.conllu'
    tr: 'tr_kenet-ud-train.conllu', 'tr_kenet-ud-dev.conllu'
    The representations are saved in the folder 'pickle2'.
    
3.  Specify the pickle file names in main, and run it. If you are 
    only interested in certain layers or setups you can modify the 
    loop(s) accordingly. 

### Results and visualization

The notebook visualize.ipynb contains lists with all accuracies, 
functions to calculate GBP, GCP, LBP, LCP, EMI and EMI-BL from them, and all
results and plots included in the paper and appendix. 
