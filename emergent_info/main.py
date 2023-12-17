import pickle, torch
from classifier import train_probe


'''

train the probes with the created pickle objects.
change file names according to language.

'''

print('---SV---')

print('mft - i, i')
for i in range(0,13):
    with open('pickle2/sv-train-mft.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-mft.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)


print('mft - i-1, i')
for i in range(0,12):
    with open('pickle2/sv-train-mft_pr.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-mft_pr.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)


print('mft - 0, i')
for i in range(1,13):
    with open('pickle2/sv-train-mft_0i.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-mft_0i.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)


print('nonmft - i, i')
for i in range(0,13):
    with open('pickle2/sv-train-nonmft.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-nonmft.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]

        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)


print('nonmft - i-1, i')
for i in range(0,12):
    with open('pickle2/sv-train-nonmft_pr.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-nonmft_pr.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)


print('nonmft - 0, i')
for i in range(1,13):
    with open('pickle2/sv-train-nonmft_0i.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-nonmft_0i.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)

print('parent - i, i')
for i in range(0,13):
    with open('pickle2/sv-train-parent.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        y = y + 15
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-parent.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        y_val = y_val + 15
        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)


print('parent - i-1, i')
for i in range(0,12):
    with open('pickle2/sv-train-parent_pr.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        #print(y.shape)
        #print(torch.bincount(y))
        y = y + 15

    with open('pickle2/sv-dev-parent_pr.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        #print(y_val.shape)
        #print(torch.bincount(y_val))
        y_val = y_val + 15

    train_probe(X, y, X_val, y_val)


print('parent - 0, i')
for i in range(1,13):
    with open('pickle2/sv-train-parent_0i.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        y = y + 15
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-parent_0i.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        y_val = y_val + 15
        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)


print('grandparent - i, i')
for i in range(0,13):
    with open('pickle2/sv-train-grandparent.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        y = y + 15
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-grandparent.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        y_val = y_val + 15

        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)


print('grandparent - i-1, i')
for i in range(0,12):
    with open('pickle2/sv-train-grandparent_pr.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        #print(y.shape)
        #print(torch.bincount(y))
        y = y + 15

    with open('pickle2/sv-dev-grandparent_pr.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        #print(y_val.shape)
        #print(torch.bincount(y_val))
        y_val = y_val + 15

    train_probe(X, y, X_val, y_val)


print('grandparent - 0, i')
for i in range(1,13):
    with open('pickle2/sv-train-grandparent_0i.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X = l[i][0]
        y = l[i][1]
        y = y + 15
        #print(y.shape)
        #print(torch.bincount(y))

    with open('pickle2/sv-dev-grandparent_0i.pickle', 'rb') as handle:
        l = pickle.load(handle)
        X_val = l[i][0]
        y_val = l[i][1]
        y_val = y_val + 15
        #print(y_val.shape)
        #print(torch.bincount(y_val))

    train_probe(X, y, X_val, y_val)

