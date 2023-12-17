import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from metrics import Evaluator, MeanSD


class ProbingModule(nn.Module):
    """ Simple Probing Classifier """

    def __init__(
            self,
            input_dim=1536 ,
            hidden_dim=64,
            output_dim=30 ,
    ):
        super(ProbingModule, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = F.relu(self.hidden(X))
        X = self.output(X)
        return X


# train and evaluate classifier
def train_probe(X, y, X_val, y_val, batch_size=64, get_dev_loss=False, verbose=False):
    evaluators = []
    losses = []
    losses_dev = []
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        dset = TensorDataset(X, y)
        data_train_loader = DataLoader(dset, batch_size=batch_size, shuffle=False)

        torch.manual_seed(seed)
        model = ProbingModule()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        evaluator = Evaluator(model)
        n_steps = 0

        for epoch in range(10):
            loss_seed = []
            for _, (state, target) in enumerate(data_train_loader):
                optimizer.zero_grad()
                tag_score = model(state)
                loss = criterion(tag_score, target)
                loss_seed.append(loss.item())
                loss.backward()
                optimizer.step()

                if n_steps in evaluator.steps:
                    step_acc = evaluator.step_accuracy(X_val, y_val)
                    train_step_acc = evaluator.train_step_accuracy(X, y)
                    if verbose:
                        print("n_steps: {0}, accuracy: {1} ".format(n_steps, step_acc))
                n_steps += 64

            with torch.no_grad():
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
            dset_dev = TensorDataset(X_val, y_val)
            data_dev_loader = DataLoader(dset_dev, batch_size=1, shuffle=False)

            for _, (state, target) in enumerate(data_dev_loader):
                tag_score = model(state)
                loss = criterion(tag_score, target)
                loss_seed_dev.append(loss.item())
            losses_dev.append(loss_seed_dev)

    mean_sd = MeanSD(evaluators)
    if verbose:
        mean_sd.print_all()
    else:
        mean_sd.print_best()
    return evaluators, losses, losses_dev
