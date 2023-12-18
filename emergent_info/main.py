import pickle
from classifier import train_probe

print('---SV---')

# Define the training scenarios
scenarios = [
    ('mft', range(0, 13), 'pickle2/sv-train-mft.pickle', 'pickle2/sv-dev-mft.pickle'),
    ('mft - i-1, i', range(0, 12), 'pickle2/sv-train-mft_pr.pickle', 'pickle2/sv-dev-mft_pr.pickle'),
    ('mft - 0, i', range(1, 13), 'pickle2/sv-train-mft_0i.pickle', 'pickle2/sv-dev-mft_0i.pickle'),
    ('nonmft - i, i', range(0, 13), 'pickle2/sv-train-nonmft.pickle', 'pickle2/sv-dev-nonmft.pickle'),
    ('nonmft - i-1, i', range(0, 12), 'pickle2/sv-train-nonmft_pr.pickle', 'pickle2/sv-dev-nonmft_pr.pickle'),
    ('nonmft - 0, i', range(1, 13), 'pickle2/sv-train-nonmft_0i.pickle', 'pickle2/sv-dev-nonmft_0i.pickle'),
]
additional_scenarios = [
    ('parent - i, i', range(0, 13), 'pickle2/sv-train-parent.pickle', 'pickle2/sv-dev-parent.pickle', 15),
    ('parent - i-1, i', range(0, 12), 'pickle2/sv-train-parent_pr.pickle', 'pickle2/sv-dev-parent_pr.pickle', 15),
    ('parent - 0, i', range(1, 13), 'pickle2/sv-train-parent_0i.pickle', 'pickle2/sv-dev-parent_0i.pickle', 15),
    ('grandparent - i, i', range(0, 13), 'pickle2/sv-train-grandparent.pickle', 'pickle2/sv-dev-grandparent.pickle', 15),
    ('grandparent - i-1, i', range(0, 12), 'pickle2/sv-train-grandparent_pr.pickle', 'pickle2/sv-dev-grandparent_pr.pickle', 15),
    ('grandparent - 0, i', range(1, 13), 'pickle2/sv-train-grandparent_0i.pickle', 'pickle2/sv-dev-grandparent_0i.pickle', 15),
]

# Loop over the scenarios
for scenario_name, indices, train_pickle, dev_pickle in scenarios:
    print(scenario_name)
    for i in indices:
        with open(train_pickle, 'rb') as handle:
            l = pickle.load(handle)
            X = l[i][0]
            y = l[i][1]

        with open(dev_pickle, 'rb') as handle:
            l = pickle.load(handle)
            X_val = l[i][0]
            y_val = l[i][1]

        train_probe(X, y, X_val, y_val)

# Loop over the additional scenarios
for scenario_name, indices, train_pickle, dev_pickle, y_offset in additional_scenarios:
    print(scenario_name)
    for i in indices:
        with open(train_pickle, 'rb') as handle:
            l = pickle.load(handle)
            X = l[i][0]
            y = l[i][1] + y_offset

        with open(dev_pickle, 'rb') as handle:
            l = pickle.load(handle)
            X_val = l[i][0]
            y_val = l[i][1] + y_offset

        train_probe(X, y, X_val, y_val)