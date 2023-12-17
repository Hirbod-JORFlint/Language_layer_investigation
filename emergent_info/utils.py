import torch


'''

functions that create tensors for a ancestors tasks.
for POS tasks, the functions are found in find_mft.py

'''


# convert lists of states/labels to single tensors
def get_tensors(X, y):
    with torch.no_grad():
        X_new = []
        y_new = []
        tensX = torch.Tensor(len(X), 2*768)
        tensy = torch.LongTensor(len(y), 1)

        assert len(X) == len(y)

        for item in X:
            X_new.append(item.unsqueeze(dim=0))
        for item in y:
            y_new.append(item)

        torch.cat(X_new, out=tensX)
        torch.cat(y_new, out=tensy)
        #print(tensX.shape, tensy.shape)
        return tensX, tensy


# get X/y from sentence objects (position of syn.head/parent)
def head_pos(sentences, layer1, layer2):
    X = []
    y = []
    for s in sentences: 
        for d in s.dependencies:
            if (abs(d[1] - d[0]) < 15): # and (d[2].item() == 4):
                X.append(torch.cat((s.states[layer1][d[0]-1], s.states[layer2][d[0]-1]),0))
                y.append(torch.LongTensor([d[1] - d[0]]))
        assert len(X) == len(y)

    return get_tensors(X,y)


# for grandparents_pos()
def get_grandparents(dep_list):
    grandparents = []
    for d in dep_list:
        for d2 in dep_list:
            if d[0] == d2[1]:
                grandparents.append((d2[0], d[1]))
    return grandparents


# get X/y from sentence objects (position of syn.head's head/grandparent)
def grandparent_pos(sentences, layer1, layer2):
    X = []
    y = []
    for s in sentences: 
        for d in get_grandparents(s.dependencies):
            if (abs(d[1] - d[0]) < 15): 
                X.append(torch.cat((s.states[layer1][d[0]-1], s.states[layer2][d[0]-1]),0))
                y.append(torch.LongTensor([d[1] - d[0]]))
        assert len(X) == len(y)
        
    return get_tensors(X,y)


# get X/y from sentence objects (POS)
def pos_tags(sentences, layer1, layer2):
    X = []
    y = []

    for s in sentences:
        if len(s.pos) == len(s.states[layer1][:-1]):
            for i in range(len(s.pos) - 1):
                c = torch.cat((s.states[layer1][i], s.states[layer2][i]),0)
                #print(c.shape)
                X.append(c)
                y.append(s.pos[i])

    return get_tensors(X,y)
