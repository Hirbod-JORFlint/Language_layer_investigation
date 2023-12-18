import pickle
import utils
import numpy as np
from load_ud import read_conllu

np.random.seed(42)

'''

save training data as pickle files
specify path to .conllu file and file name

'''


### Change path to UD files here ###
path = 'E:\\Dars\\Research Project\\Language_layer_investigation\\en_ewt-ud-train.conllu'
path_dev = 'E:\\Dars\\Research Project\\Language_layer_investigation\\en_ewt-ud-dev.conllu'



# save them to file as sentence objects
def save_to_file(path, file_name, limit=True):
    sentences = read_conllu(path)
    if limit:
        sentences = np.random.choice(sentences, size=1000)

    p = []
    for i in range(0,13):
        p.append(utils.head_pos(sentences, i,i))
    with open(file_name + 'parent.pickle', 'wb') as file:
        pickle.dump(p, file)

    p_pr = []
    for i in range(1,13):
        p_pr.append(utils.head_pos(sentences, i-1,i))
    with open(file_name + 'parent_pr.pickle', 'wb') as file:
        pickle.dump(p_pr, file)


    p_0i = []
    for i in range(0,13):
        p_0i.append(utils.head_pos(sentences, 0,i))
    with open(file_name + 'parent_0i.pickle', 'wb') as file:
        pickle.dump(p_0i, file)


    gp = []
    for i in range(0,13):
        gp.append(utils.grandparent_pos(sentences, i,i))
    with open(file_name + 'grandparent.pickle', 'wb') as file:
        pickle.dump(gp, file)

    gp_pr = []
    for i in range(1,13):
        gp_pr.append(utils.grandparent_pos(sentences, i-1,i))
    with open(file_name + 'grandparent_pr.pickle', 'wb') as file:
        pickle.dump(gp_pr, file)

    gp_0i = []
    for i in range(0,13):
        gp_0i.append(utils.grandparent_pos(sentences, 0,i))
    with open(file_name + 'grandparent_0i.pickle', 'wb') as file:
        pickle.dump(gp_0i, file)


### Change names for the pickle files here ###
file_train = 'pickle2/en-train-'
save_to_file(path, file_train, limit=True)
file_dev = 'pickle2/en-dev-'
save_to_file(path_dev, file_dev, limit=False)
