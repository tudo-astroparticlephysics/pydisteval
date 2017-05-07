import logging

import numpy as np
import copy

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor


logger = logging.getLogger('disteval.discretization.TreeBinningSklearn')


def __sample_uniform__(y, sample_weight=None):
    """Function used to sample a uniform distribution from a binned y.

    Parameters
    ----------
    y : numpy.intarray, shape=(n_samples)
        Array of the true classification labels.

    y : numpy.floatarray, shape=(n_samples)
        Event weights.

    Returns
    -------
    mask: list of bools
        A boolean mask for y. True for events that should be kept.
    """
    logger.info('Sampling uniform distributed class label!')
    freq = np.bincount(y, weights=sample_weight)
    mask = freq > 0
    if sample_weight is None:
        freq = freq.astype(float)
    freq /= np.min(freq[mask])
    rnd = np.random.uniform(size=len(y)) * freq[y]
    return rnd <= 1.


def get_family(tree):
    """Function to get the relations between nodes in a sklean tree.

    Parameters
    ----------
    tree : sklearn.tree._tree.Tree
        Tree form which the relations should be extracted

    Returns
    -------
    parents : array of ints, shape=(n_nodes)
        List of the parents of each node. -1 indicates no parents and
        should only appear for the first node.

    grand_parents : array of ints, shape=(n_nodes)
        List of the parents of each node. -1 indicates no parents and
        should only appear for the first node.

    siblings : array of ints, shape=(n_nodes)
        List of the parents of each node. -1 indicates no parents and
        should only appear for the first node.
    """
    logger.debug('Creating array for tree relations.')

    def walk_path(tree, idx, last_idx, node_list):
        node_list.append([idx, last_idx])
        l_child = tree.children_left[idx]
        if l_child != -1:
            node_list.extend(walk_path(tree, l_child, idx, node_list))
        r_child = tree.children_right[idx]
        if r_child != -1:
            node_list.extend(walk_path(tree, r_child, idx, node_list))
        return node_list

    node_list = walk_path(tree, 0, -1, [])
    parents = np.ones_like(tree.children_left, dtype=int) * -1
    grand_parents = np.ones_like(tree.children_left, dtype=int) * -1
    siblings = np.ones_like(tree.children_left, dtype=int) * -1
    for own_idx, parent_idx in node_list:
        parents[own_idx] = parent_idx
        grand_parents[own_idx] = parents[parent_idx]
    for parent_idx in parents:
        l_child_idx = tree.children_left[parent_idx]
        r_child_idx = tree.children_left[parent_idx]
        siblings[l_child_idx] = r_child_idx
        siblings[r_child_idx] = l_child_idx
    return parents, grand_parents, siblings


def delete_node(tree, node_idx):
    tree.children_right[node_idx] = -4
    tree.children_left[node_idx] = -4
    tree.feature[node_idx] = -4
    tree.threshold[node_idx] = -4


def remove_node(tree, node_idx):
    parents, grand_parents, siblings = get_family(tree)

    parent_idx = parents[node_idx]
    sibling_idx = siblings[node_idx]

    tree.children_right[parent_idx] = tree.children_right[sibling_idx]
    tree.children_left[parent_idx] = tree.children_left[sibling_idx]
    tree.feature[parent_idx] = tree.feature[sibling_idx]
    tree.threshold[parent_idx] = tree.threshold[sibling_idx]

    delete_node(tree, sibling_idx)
    return parent_idx, sibling_idx


class TreeBinningSklearn(object):
    def __init__(self,
                 regression=False,
                 max_features=None,
                 min_samples_split=2,
                 max_depth=None,
                 min_samples_leaf=1,
                 max_leaf_nodes=None,
                 boosted=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 ensemble_select='best',
                 random_state=None):

        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state
        self.regression = regression
        if regression:
            logger.info('Initialized TreeBiningSklearn with a '
                         'regression tree.')
            self.tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features=max_features,
                random_state=random_state)
            if boosted in ['linear', 'square', 'exponential']:
                logger.info('Activated AdaBoost!')
                self.boosted = AdaBoostRegressor(
                    base_estimator=self.tree,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    algorithm=boosted,
                    random_state=random_state)

            elif boosted is not None:
                raise ValueError(
                    '\'boosted\' should be None for no boosting '
                    'or either \'linear\', \'square\', \'exponential\' for a '
                    'boosted regression.')
            else:
                self.boosted = None
        else:
            logger.info('Initialized TreeBiningSklearn with a '
                         'classification tree.')
            self.tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features=max_features,
                random_state=random_state)
            if boosted in ['SAMME', 'SAMME.R']:
                logger.info('Activated AdaBoost!')
                self.boosted = AdaBoostClassifier(
                    base_estimator=self.tree,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    algorithm=boosted,
                    random_state=random_state)
            elif boosted is not None:
                raise ValueError(
                    '\'boosted\' should be None for no boosting '
                    'or either \'SAMME\' or \'SAMME.R\' for a '
                    'boosted classification.')
            else:
                self.boosted = None
        self.ensemble_select = ensemble_select.lower()
        self.leaf_idx_mapping = None
        self.n_bins = None

    def fit(self,
            X,
            y,
            sample_weight=None,
            uniform=True):
        logger.info('Start to fit the model!')
        if self.regression and uniform:
            logger.warn('Uniform smapling is only supported for classifcation')
        elif uniform:
            mask = __sample_uniform__(y, sample_weight=sample_weight)
            y = y[mask]
            X = X[mask]
        if self.boosted is not None:
            if self.ensemble_select.lower() not in ['best', 'last']:
                raise ValueError(
                    '\'ensemble_select\' must be \'best\' or \'last\'!')
            self.boosted.fit(X=X,
                             y=y,
                             sample_weight=sample_weight)
            if self.ensemble_select == 'best':
                tree_idx = np.argmax(self.boosted.estimator_weights_)
                logger.info('{} has the highest estimator weight.'.format(
                    tree_idx))
            elif self.ensemble_select == 'last':
                tree_idx = -1
                logger.info('Last tree selected!')
            self.tree = self.boosted.estimators_[tree_idx]
        else:
            self.tree.fit(X=X,
                          y=y,
                          sample_weight=sample_weight)

        self.generate_leaf_mapping()


    def generate_leaf_mapping(self):
        logger.debug('Mapping for leafs is created.')
        self.leaf_idx_mapping = {}
        is_leaf = np.where(self.tree.tree_.feature == -2)[0]
        counter = 0
        for is_leaf_i in is_leaf:
            self.leaf_idx_mapping[is_leaf_i] = counter
            counter += 1
        self.n_bins = len(self.leaf_idx_mapping)

    def predict(self, X):
        return self.tree.predict(X)

    def digitize(self, X):
        leafyfied = self.tree.apply(X)
        digitized = np.array([self.leaf_idx_mapping[val_i]
                              for val_i in leafyfied])
        return digitized

    def decision_path(self, X, column_names=None):
        indicator = self.tree.decision_path(X)
        return indicator

    def copy(self):
        clone = TreeBinningSklearn(random_state=self.random_state)
        clone.tree = copy.deepcopy(self.tree)
        return clone

    def prune(self, X, threshold):
        logger.info('Started to prune leafs with less than {} events.'.format(
            threshold))
        tree = self.tree.tree_

        def find_withered_leaf():
            leafyfied = self.tree.apply(X)
            occureance = np.bincount(leafyfied, minlength=tree.node_count)

            is_leaf = np.where(self.tree.children_right == -1)[0]
            is_below = occureance < threshold
            is_leaf_below = np.logical_and(is_leaf, is_below)
            if any(is_leaf_below):
                is_leaf_below_idx = np.where(is_leaf_below)
                idx_min_leaf = np.argmin(occureance[is_leaf_below_idx])
                return is_leaf_below_idx[idx_min_leaf]
            else:
                return None
        n_bins_before_pruning = self.n_bins
        while True:
            idx = find_withered_leaf()
            if idx is None:
                break
            else:
                remove_node(tree, idx)
                self.generate_leaf_mapping()
        n_bins_after_pruning = self.n_bins
        logger.info('Number of leafs reduced from {} to {}.'.format(
            n_bins_before_pruning, n_bins_after_pruning))
