# The script containing classes used in other scripts

import math, random, re, itertools
import numpy as np
from scipy.optimize import minimize
from sklearn import svm
from aa_constants import AA_GROUPS_

class PeptideFeature:
    def __init__(self, regex, lambda_):
        self.regex = regex
        self.lambda_ = lambda_

class EnrichedDimer:
    def __init__(self, group0, group1, linker_len, occur_prob = 1.0):
        self.group0 = group0
        self.group1 = group1
        self.linker_len = linker_len
        self.occur_prob = occur_prob
        self.regex = '[{}]{}[{}]'.format(AA_GROUPS_[group0], '.' * linker_len, AA_GROUPS_[group1])
        self.cregex = re.compile(self.regex)

class DecisionTree:
    def __init__(self):
        self.nodes = []
        self.ranges = {}
        self.start_value = None

class TreeNode:
    def __init__(self, current, min_, max_, less = None, noless = None):
        self.current = current
        self.min_ = min_
        self.max_ = max_
        self.less = less
        self.noless = noless

class TreeNodeGenerator:
    """Class to generate nodes of a decision tree"""
    # This class operates like a functor
    def __init__(self, tree, random_letter_generator):
        self.tree = tree
        self.random_letter_generator = random_letter_generator
    def generate_node(self, node):
        """Recursive method to generate or complete a ranges decision tree"""
        # A node shows that if a value is less than that corresponding to node.current and node.min is the miniaml possible node, node.less should be compared with as the next
        at_lower_border = node.current == node.min_ + 1
        at_upper_border = node.current == node.max_
        if node.less is None:
            node.less = self.random_letter_generator.letter_list[node.min_] if at_lower_border else node.min_ + math.ceil((node.current - node.min_) / 2)
        if node.noless is None:
            node.noless = self.random_letter_generator.letter_list[node.current] if at_upper_border else node.current + math.ceil((node.max_ - node.current) / 2)
        self.tree.nodes.append(node)
        if at_upper_border:
            if at_lower_border:
                return
            self.generate_node(TreeNode(current = node.less, min_ = node.min_, max_ = node.current - 1))
            return
        elif at_lower_border:
            self.generate_node(TreeNode(current = node.noless, min_ = node.current, max_ = node.max_))
        else:
            self.generate_node(TreeNode(current = node.less, min_ = node.min_, max_ = node.current - 1))
            self.generate_node(TreeNode(current = node.noless, min_ = node.current, max_ = node.max_))
    def get_tree(self):
        return self.tree

class RandomLetterGenerator:
    """Class to train the algorithm and generate random letters"""
    def __init__(self, letter_freq):
        self.letter_freq = letter_freq
        self.normalize_letter_freq()
        self.generate_letter_ranges()
        self.tree = self.generate_initial_decision_tree() if len(letter_freq) > 1 else None
        random.seed()
    def normalize_letter_freq(self):
        """Method to ensure that the letter frequencies are summed to 1"""
        freq_sum = sum(self.letter_freq.values())
        if freq_sum <= 0.0:
            raise ValueError("Illegal values of letter frequencies")
        for letter in self.letter_freq:
            self.letter_freq[letter] /= freq_sum
    def generate_letter_ranges(self):
        """Method to generate ranges in the interval [0,1) that corresponds to each letter"""
        self.letter_list = sorted(list(self.letter_freq))
        self.letter_ranges_lower = []
        freq_sum = 0.0
        for letter in self.letter_list:
            freq_sum_prev = freq_sum
            freq_sum += self.letter_freq[letter]
            if freq_sum != freq_sum_prev: #Ignore letters with zero frequencies
                self.letter_ranges_lower.append(freq_sum_prev)
    def generate_tree_ranges_from_nodes(self, tree):
        """Method to generate the decison rnages in the interval [0,1) from the node information"""
        for node in tree.nodes:
            range_lower = self.letter_ranges_lower[node.current]
            less_try = self.letter_ranges_lower[node.less] if type(node.less) == int else node.less
            noless_try = self.letter_ranges_lower[node.noless] if type(node.noless) == int else node.noless
            tree.ranges[range_lower] = (less_try, noless_try)
        tree.start_value = self.letter_ranges_lower[tree.nodes[0].current]
    def generate_initial_decision_tree(self):
        """Method to generate the initial tree with the order of division of the interval [0,1) into sub-intervals based on the letter frequencies"""
        tree = DecisionTree()
        tree_node_generator = TreeNodeGenerator(tree, self)
        max_ = len(self.letter_freq) - 1
        if max_ == 0:
            raise RuntimeError('There must be at least 2 letters for the random generator to choose from')
        tree_node_generator.generate_node(TreeNode(current = math.ceil(max_ / 2), min_ = 0, max_ = max_)) # As tree is passed to the object by pointer, it will be updated
        self.generate_tree_ranges_from_nodes(tree)
        return tree
    def generate_letter(self):
        """Method to generate a random letter"""
        if self.tree is None:
            return list(self.letter_freq.keys())[0]
        random_value = random.random()
        value = self.tree.start_value
        while True:
            if random_value < value:
                next_value = self.tree.ranges[value][0]
            else:
                next_value = self.tree.ranges[value][1]
            if type(next_value) == str:
                return next_value
            value = next_value

class RandomPeptideGenerator:
    """Factory to generate random peprides with certain patterns"""
    def __init__(self, aa_freq, length, enriched_dimers):
        self.aa_freq = aa_freq
        self.length = length
        self.enriched_dimers = enriched_dimers
        self.letter_generator = RandomLetterGenerator(self.aa_freq)
        self.group_letter_generators = self.generate_group_letter_generators()
    def generate_group_letter_generators(self):
        """Method to generate the dictionary of letter generators for each amino acid group"""
        group_letter_generators = {}
        for group_name, aa_list in AA_GROUPS_.items():
            group_aa_freq = {}
            for aa in aa_list:
                group_aa_freq[aa] = self.aa_freq[aa]
            group_letter_generators[group_name] = RandomLetterGenerator(group_aa_freq)
        return group_letter_generators
    def generate_random_peptide(self):
        """Method ro generate a random peptide with given background amino acid frequencies"""
        peptide = []
        for i in range(self.length):
            peptide.append(self.letter_generator.generate_letter())
        return peptide
    def generate_random_peptide_with_dimers(self):
        """Method ro generate a random peptide with given background amino acid frequencies and containing enriched dimers"""
        peptide = self.generate_random_peptide()
        random.shuffle(self.enriched_dimers)
        blocked_positions = set()
        for dimer in self.enriched_dimers:
            if dimer.occur_prob < random.random():
                continue
            start_positions = set(range(self.length - dimer.linker_len - 1)) - blocked_positions
            for position in blocked_positions:
                start_positions.discard(position - dimer.linker_len - 1)
            if not start_positions:
                raise RuntimeError('Impossible to generate a peptide with given dimers')
            start_position = random.choice(tuple(start_positions))
            end_position = start_position + dimer.linker_len + 1
            blocked_positions.add(start_position)
            blocked_positions.add(end_position)
            peptide[start_position] = self.group_letter_generators[dimer.group0].generate_letter()
            peptide[end_position] = self.group_letter_generators[dimer.group1].generate_letter()
        return peptide
    def generate_random_peptides(self, n, with_dimers = True):
        """Method to generate a dataset of random peptides"""
        random_peptides = []
        for i in range(n):
            random_peptides.append(''.join(self.generate_random_peptide_with_dimers() if with_dimers else self.generate_random_peptide()))
        return random_peptides

class DistanceWeightCalculator:
    """Class to calculate distance between and weights of the data samples"""
    def calc_matrix(dataset, subject = 'distances', unit_space = False):
        """Method to calculate the matrix of either pairwise euclidian distances in the feauture space between the input data points (samples) of correlations"""
        n_samples = len(dataset)
        matrix = np.zeros((n_samples, n_samples))
        unit_diag_len = np.sqrt(n_samples)
        if subject == 'cosines':
            norms = np.zeros(n_samples)
            for i in range(n_samples):
                norms[i] = np.linalg.norm(dataset[i])
        for i, j in itertools.product(range(n_samples), range(n_samples)):
            if j > i:
                continue
            if subject == 'distances':
                matrix[i][j] = np.sqrt(np.sum(np.power(dataset[i] -  dataset[j], 2)))
            elif subject == 'correlations_from_distances':
                distance = np.sqrt(np.sum(np.power(dataset[i] -  dataset[j], 2)))
                matrix[i][j] = 1 - ((distance / unit_diag_len) if unit_space else np.tanh(distance / unit_diag_len))
            elif subject == 'dot_products':
                matrix[i][j] = np.dot(dataset[i], dataset[j])
            elif subject == 'cosines':
                dot_product = np.dot(dataset[i], dataset[j])
                matrix[i][j] = dot_product / (norms[i] * norms[j]) if dot_product != 0.0 else 1.0
            else:
                raise RuntimeError('Invalid subject')
        upper_indices = np.triu_indices(n_samples)
        matrix[upper_indices] = matrix.T[upper_indices]
        return matrix
    def distances_to_correlations(distances, unit_space = False):
        """Mehtod to convert a matrix or pairwise distances into the matrix of pairwise correlations. In a unit space, all the underlying features have values in the range [0,1]"""
        coef = np.sqrt(distances.shape[0])
        correlations = 1 - ((distances / coef) if unit_space else np.tanh(distances / coef))
        return correlations
    def weights_from_correlations(correlations):
        """Method to derive weights for the samples on the basis of their correlations"""
        def func_to_minimize(weights):
            return np.abs(np.dot(np.dot(weights, correlations.T), weights))
        n_samples = correlations.shape[0]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0, 1) for i in range(n_samples)]
        weights_init = np.zeros(n_samples) + 1/ n_samples
        result = minimize(func_to_minimize, weights_init, method = 'SLSQP', constraints = constraints, bounds = bounds)
        return result.x

class SVC_w:
    """Class to wrap sklearn.svm.SVC for the usage within pipelines with weights"""
    def __init__(self, *args, **kwargs):
        self.svc = svm.SVC(*args, **kwargs)
    def _calc_weights(X, y):
        """Method to calculate the sample weights"""
        weights = np.zeros(y.shape[0])
        levels = np.unique(y)
        for label_value in levels:
            sample_indices = y == label_value
            correlation_matrix = DistanceWeightCalculator.calc_matrix(X[sample_indices], subject = 'cosines')
            weights[sample_indices] = DistanceWeightCalculator.weights_from_correlations(correlation_matrix) * np.sum(sample_indices)
        return weights
    def fit_transform(self, X, y):
        weights = SVC_w._calc_weights(X, y)
        self.svc.fit_transform(X, y, sample_weight = weights)
    def decision_function(self, *args, **kwargs):
        return self.svc.decision_function(*args, **kwargs)
    def density(self):
        self.svc.density()
        return self
    def get_params(self, *args, **kwargs):
        return self.svc.get_params(*args, **kwargs)
    def fit(self, X, y):
        weights = SVC_w._calc_weights(X, y)
        self.svc.fit(X, y, sample_weight = weights)
        return self
    def predict(self, *args, **kwargs):
        return self.svc.predict(*args, **kwargs)
    def score(self, X, y):
        #weights = SVC_w._calc_weights(X, y)
        return self.svc.score(X, y)#, sample_weight = weights)
    def set_params(self, **params):
        self.svc.set_params(**params)
        return self
    def sparsity(self):
        self.svc.sparsity()
        return self