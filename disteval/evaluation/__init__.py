# -*- coding:utf-8 -*-
"""
Collection of methods to evaluate the results of disteval functions
"""
from .feature_importance_test import feature_importance_mad
from .feature_importance_test import feature_importance_mad_majority
from .roc_curve_equivalence_test import roc_curve_equivalence_ks_test

__all__ = ['feature_importance_mad',
           'feature_importance_mad_majority',
           'roc_curve_equivalence_ks_test']
