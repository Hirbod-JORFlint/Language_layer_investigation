import tensorflow as tf



# convert lists of states/labels to single tensors
def get_tensors(X, y):
    X_new = []
    y_new = []
    tensX = tf.constant(X, shape=(len(X), 2 * 768))
    tensy = tf.constant(y, shape=(len(y), 1))

    assert len(X) == len(y)

    for item in X:
        X_new.append(tf.expand_dims(item, axis=0))
    for item in y:
        y_new.append(item)

    tensX = tf.concat(X_new, axis=0)
    tensy = tf.concat(y_new, axis=0)
    return tensX, tensy


# get X/y from sentence objects (position of syn.head/parent)
def head_pos(sentences, layer1, layer2):
    X = []
    y = []
    for s in sentences:
        for d in s.dependencies:
            if (abs(d[1] - d[0]) < 15):
                X.append(tf.concat([s.states[layer1][d[0] - 1], s.states[layer2][d[0] - 1]], axis=0))
                y.append(tf.constant([d[1] - d[0]], dtype=tf.int64))
        assert len(X) == len(y)

    return get_tensors(X, y)


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
                X.append(tf.concat([s.states[layer1][d[0] - 1], s.states[layer2][d[0] - 1]], axis=0))
                y.append(tf.constant([d[1] - d[0]], dtype=tf.int64))
        assert len(X) == len(y)

    return get_tensors(X, y)


# get X/y from sentence objects (POS)
def pos_tags(sentences, layer1, layer2):
    X = []
    y = []

    for s in sentences:
        if len(s.pos) == len(s.states[layer1][:-1]):
            for i in range(len(s.pos) - 1):
                c = tf.concat([s.states[layer1][i], s.states[layer2][i]], axis=0)
                X.append(c)
                y.append(s.pos[i])

    return get_tensors(X, y)
