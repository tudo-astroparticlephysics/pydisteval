#!/usr/bin/env python
# -*- coding: utf-8 -*-

class ClassifierCharacteristics(object):
    """Class to define and compare Characteristics of classifier.
    The core of the Class is the dict ops containing keys whether
    attributes or functions are required/forbidden. Keys like
    'callable:fit' are True if the classifier has a callable function
    'fit'. Keys like 'has:feature_importance' are True if the classifier
    has an atribute 'feature_importance'.
    True in the dict means function/attribute is needed or present.
    False means function/attribute is forbidden or not present.
    None in the dict means is ignore in the evaluation

    Parameters
    ----------
    clf: None or object
        If None the dict is initiated with None for all keys.
        If clf is provided the dict is contains only True and False
        depending on the clf characteristics

    Attributes
    ----------
    opts : dict
        Dictionary containing all the needed/desired characteristics.

    clf : object
        If a clf is provided, a pointer to the classifier is stored.
        To check characteristics later on."""
    def __init__(self, clf=None):
        self.opts = {
            'callable:fit': None,
            'callable:predict': None,
            'callable:predict_proba': None,
            'callable:decision_function': None,
            'has:feature_importance': None}
        if clf is not None:
            self.clf = clf
            for key in self.opts.keys():
                self.opts[key] = self.__evalute_clf__(key)

    def __evalute_clf__(self, key):
        """Check if the classifier provides the attribute/funtions
        asked for with the key. Keys must start with either "callable:"
        or "has:".
        "callable:<name>"  would check for a funtions with the name <name>.
        "has:<name>"  would check for a attribute with the name <name>.
        Parameters
        ----------
        key: str
            If None the dict is initiated with None for all keys.
            If clf is provided the dict is contains only True and False
            depending on the clf characteristics

        Returns
        ----------
        present : bool
            Boolean whether the asked for characteristic is present"""
        if key.startswith('callable:'):
            desired_callable = key.replace('callable:', '')
            if hasattr(self.clf, desired_callable):
                if callable(getattr(self.clf, desired_callable)):
                    return True
        elif key.startswith('has:'):
            desired_attribute = key.replace('has:', '')
            if hasattr(self.clf, desired_attribute):
                return True
        else:
            print(key)
            raise ValueError('Opts keys have to start with eiter callable:'
                             ' for functions or has: for attributes')
        return False

    def fulfilling(self, second_instance, two_sided=False):
        """Check if the classifier provides the attribute/funtions
        asked for with the key. Keys must start with either "callable:"
        or "has:".
        "callable:<name>"  would check for a funtions with the name <name>.
        "has:<name>"  would check for a attribute with the name <name>.
        Parameters
        ----------
        second_instance: ClassifierCharacteristics
            Second instance of a ClassifierCharacteristics which defines
            the needed characteristics.

        two_sided: boolean, optional (default=False)
            If False only the characteristics asked for in the second
            instance has to be fulfilled. If two_sided is True. Both
            instances has to be the same (equivalent to __eq__)

        Returns
        ----------
        present : bool
            Boolean whether the asked for characteristic is present"""
        if two_sided:
            check_keys_1 = set([k for k, v in self.opts.items()
                                if v is not None])
            check_keys_2 = set([k for k, v in second_instance.opts.items()
                                if v is not None])
            check_keys = check_keys_1.intersection(check_keys_2)
        else:
            check_keys = [k for k, v in second_instance.opts.items()
                          if v is not None]
        for key in check_keys:
            if key not in self.opts.keys():
                if hasattr(self, 'clf'):
                    value = self.__evalute_clf__(key)
                    self.opts[key] = value
                else:
                    raise KeyError('%s not set for the comparison partner')
            if key not in second_instance.opts.keys():
                if hasattr(second_instance, 'clf'):
                    value = second_instance.__evalute_clf__(key)
                    second_instance.opts[key] = value
                else:
                    raise KeyError('%s not set for the comparison partner')
            if self.opts[key] != second_instance.opts[key]:
                att = key.replace('callable:', '')
                att = att.replace('has:', '')
                if self.opts[key]:
                    msg = 'Provided classifier has %s' % att
                else:
                    msg = 'Provided classifier is missing %s' % att
                raise AttributeError(msg)
        return True

    def __eq__(self, second_instance):
        return self.fulfilling(second_instance, two_sided=True)
