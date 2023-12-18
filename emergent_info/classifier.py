import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from metrics import Evaluator, MeanSD

class ProbingModule(keras.Model):
    """ Simple Probing Classifier """

    def __init__(self, input_dim=1536, hidden_dim=64, output_dim=30):
        super(ProbingModule, self).__init__()
        self.hidden = layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,))
        self.output = layers.Dense(output_dim)

    def call(self, X):
        X = self.hidden(X)
        X = self.output(X)
        return X

# train and evaluate classifier
def train_probe(X, y, X_val, y_val, batch_size=64, get_dev_loss=False, verbose=False):
    evaluators = []
    losses = []
    losses_dev = []
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tf.random.set_seed(seed)
        model = ProbingModule()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        evaluator = Evaluator(model)
        n_steps = 0

        for epoch in range(10):
            loss_seed = []
            for i in range(0, len(X), batch_size):
                state, target = X[i:i+batch_size], y[i:i+batch_size]
                loss, _ = model.train_on_batch(state, target)
                loss_seed.append(loss)

                if n_steps in evaluator.steps:
                    step_acc = evaluator.step_accuracy(X_val, y_val)
                    train_step_acc = evaluator.train_step_accuracy(X, y)
                    if verbose:
                        print("n_steps: {0}, accuracy: {1} ".format(n_steps, step_acc))
                n_steps += 64

            acc = evaluator.accuracy(X_val, y_val)
            if verbose:
                print("Epoch: {0}, Loss {1}, Acc Train: {2}, Acc Dev: {3}".format(epoch, loss,
                                                                              evaluator.accuracy_notrack(X, y),
                                                                              acc))

        losses.append(loss_seed)
        evaluators.append(evaluator)

        # for loss ranking scoring function
        if get_dev_loss:
            loss_seed_dev = []
            for i in range(0, len(X_val)):
                state, target = X_val[i:i+1], y_val[i:i+1]
                loss, _ = model.test_on_batch(state, target)
                loss_seed_dev.append(loss)
            losses_dev.append(loss_seed_dev)

    mean_sd = MeanSD(evaluators)
    if verbose:
        mean_sd.print_all()
    else:
        mean_sd.print_best()
    return evaluators, losses, losses_dev
