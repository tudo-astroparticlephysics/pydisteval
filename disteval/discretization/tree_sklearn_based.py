import logging

import numpy as np
import copy

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor


logger = logging.getLogger('TreeBinningSklearn')


def __sample_uniform__(y, sample_weight=None):
    freq = np.bincount(y, weights=sample_weight)
    mask = freq > 0
    if sample_weight is None:
        freq = freq.astype(float)
    freq /= np.min(freq[mask])
    rnd = np.random.uniform(size=len(y)) * freq[y]
    return rnd <= 1.


def get_parents(tree):
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
    parents = np.zeros_like(tree.children_left, dtype=int)
    for own_idx, parent_idx in node_list:
        parents[own_idx] = parent_idx
    return parents


def remove_node(tree, node_idx):
    tree.children_right[node_idx] = -4
    tree.children_left[node_idx] = -4
    tree.feature[node_idx] = -4
    tree.threshold[node_idx] = -4


def set_to_leaf(tree, node_idx):
    l_child = tree.children_left[node_idx]
    r_child = tree.children_left[node_idx]
    if l_child == -1 and r_child == -1:
        logger.warn('{} is already a leaf!')
    elif l_child == -1 and r_child != -1:
        raise RuntimeError('Broken Tree! l_child is -1 while r_child is != -1')
    elif l_child != -1 and r_child == -1:
        raise RuntimeError('Broken Tree! r_child is -1 while l_child is != -1')
    else:
        remove_node(tree, l_child)
        remove_node(tree, r_child)

        tree.children_right[node_idx] = -1
        tree.children_left[node_idx] = -1
        tree.feature[node_idx] = -2
        tree.threshold[node_idx] = -2


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
            self.tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features=max_features,
                random_state=random_state)
            if boosted in ['linear', 'square', 'exponential']:
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
            self.tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_leaf_nodes=max_leaf_nodes,
                max_features=max_features,
                random_state=random_state)
            if boosted in ['SAMME', 'SAMME.R']:
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

            elif self.ensemble_select == 'last':
                tree_idx = -1
            self.tree = self.boosted.estimators_[tree_idx]
        else:
            self.tree.fit(X=X,
                          y=y,
                          sample_weight=sample_weight)
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
        tree = self.tree.tree_

        leafyfied = self.tree.apply(X)
        occureance = np.bincount(leafyfied, minlength=tree.node_count)

        parents = get_parents(tree)

        is_leaf = np.where(self.tree.children_right == -1)[0]
        is_below = occureance < threshold
        is_leaf_below = np.logical_and(is_leaf, is_below)
        while any(is_leaf_below):
            idx = np.where(is_leaf_below)[0][-1]
            parent_idx = parents[idx]
            set_to_leaf(tree, parent_idx)

            l_child = tree.children_left[parent_idx]
            r_child = tree.children_right[parent_idx]

            is_leaf[l_child] = False
            is_leaf[r_child] = False
            is_leaf[parent_idx] = True

            occureance[parent_idx] += occureance[idx]
            occureance[parent_idx] += occureance[idx]
            is_below = occureance < threshold

            is_leaf_below = np.logical_and(is_leaf, is_below)
