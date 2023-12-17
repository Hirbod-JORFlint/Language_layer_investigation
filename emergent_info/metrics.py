import torch, math


class Evaluator:
    """Perform and save evaluation of a model at given steps"""

    def __init__(self, model):
        self.steps = [0, 64, 128, 256, 512, 1024, 2048, 4096]  # steps for early evaluation
        self.accs = []
        self.step_accs = []
        self.train_known_ex = []
        self.known_ex = []  # save correctly tagged test examples to calculate avg.# training steps
        self.model = model

    # calc accuracy; save it in self.accs
    def accuracy(self, X, y):
        correct = 0
        total = 0
        for state, target in zip(X, y):
            outputs = self.model(state)
            # supporting both torch and tensorflow by using .data
            #max_value = torch.max(outputs.data)
            # tensorflow max
            max_value = tf.math.reduce_max(outputs.data)
            pred = (outputs.data == max_value).nonzero()
            total += 1
            try:
                correct += (pred == target).sum().item()
            except: pass
        self.accs.append(100 * correct / total)
        return 100 * correct / total

    # calc accuracy; save it in self.step_accs
    def step_accuracy(self, X, y):
        correct = 0
        total = 0
        knowns_lst = []
        for state, target in zip(X, y):
            outputs = self.model(state)
            max_value = torch.max(outputs.data)
            max_value = tf.math.reduce_max(outputs.data)
            pred = (outputs.data == max_value).nonzero()
            total += 1
            try:
                correct += (pred == target).sum().item()
                if pred == target:
                    knowns_lst.append((state, target))
            except: pass
        self.known_ex.append(knowns_lst)
        self.step_accs.append(100 * correct / total)
        return 100 * correct / total

    # calc accuracy; save it in self.step_accs
    def train_step_accuracy(self, X, y):
        correct = 0
        total = 0
        knowns_lst = []
        for state, target in zip(X, y):
            outputs = self.model(state)
            max_value = torch.max(outputs.data)
            max_value = tf.math.reduce_max(outputs.data)
            pred = (outputs.data == max_value).nonzero()
            total += 1
            try:
                correct += (pred == target).sum().item()
                if pred == target:
                    knowns_lst.append((state, target))
            except: pass
        self.train_known_ex.append(knowns_lst)
        return 100 * correct / total

    # calc accuracy; don't save it
    def accuracy_notrack(self, X, y):
        correct = 0
        total = 0
        for state, target in zip(X, y):
            outputs = self.model(state)
            max_value = torch.max(outputs.data)
            pred = (outputs.data == max_value).nonzero()
            total += 1
            try:
                correct += (pred == target).sum().item()
            except: pass
        return 100 * correct / total


class MeanSD:
    """Calculate mean and standart deviation for a list of objects of the class Evaluator"""

    def __init__(self, evaluators):
        self.evaluators = evaluators

    # calculate mean of a list
    @staticmethod
    def mean(acc_list):
        return sum(acc_list) / len(acc_list)

    # calculate standard deviation of the mean of a list
    @staticmethod
    def std_deviation(acc_list):
        deviations = [math.pow(MeanSD.mean(acc_list) - r, 2) for r in acc_list]
        std_dev = math.sqrt((1 / (len(acc_list) - 1)) * sum(deviations))  # with Bessel's correction
        return std_dev

    # print mean and sd for accuracies of the best epoch
    def print_best(self):
        acc_list = [max(e.accs) for e in self.evaluators]
        acc_mean = MeanSD.mean(acc_list)
        std_dev = MeanSD.std_deviation(acc_list)
        print('Best epoch:    Mean: {0}    Standard Deviation: {1}'.format(acc_mean, std_dev))

    # print mean and sd for all intermediate accuracies used for WS
    def print_accuracies(self):
        for i in range(1, 10):
            acc_list = [e.accs[i] for e in self.evaluators]
            acc_mean = MeanSD.mean(acc_list)
            std_dev = MeanSD.std_deviation(acc_list)
            print('Epoch: {0}    Mean: {1}    Standard Deviation: {2}'.format(i, acc_mean,
                                                                             std_dev))

    # print mean and sd for all metrics
    def print_all(self):
        self.print_best()
        self.print_accuracies()
