import logging

import numpy as np

from .base_classes import CalcPart, PlotPart


class CalcBinning(CalcPart):
    name = 'CalcBinning'
    level = 0
    def __init__(self, n_bins, check_all=True):
        self.n_bins = n_bins
        self.check_all = check_all
        self.n_components = 0

    def execute(self, result_tray, component):
        super(CalcPart, self).execute(result_tray, component)
        if not hasattr(result_tray, 'binning'):
            min_x = np.min(component.X)
            max_x = np.max(component.X)
            binning = np.linspace(min_x, max_x, self.n_bins + 1)
            result_tray.add(binning, 'binning')
        elif self.check_all:
            current_min_x = result_tray.binning[0]
            current_max_x = result_tray.binning[-1]
            min_x = min(current_min_x, np.min(component.X))
            max_x = max(current_max_x, np.max(component.X))
            binning = np.linspace(min_x, max_x, self.n_bins + 1)
            result_tray.add(binning, 'binning')
        self.n_components += 1
        result_tray.add(self.n_components, 'n_components')
        return result_tray

    def reset(self):
        super(CalcPart, self).reset()
        self.n_components = 0


class CalcHistogram(CalcPart):
    name = 'CalcHistogram'
    level = 1

    def execute(self, result_tray, component):
        super(CalcPart, self).execute(result_tray, component)
        binning = result_tray.binning

        weights = component.weights
        X = component.X
        idx = component.idx

        n_bins = len(binning) + 1
        if not hasattr(result_tray, 'sum_w'):
            sum_w = np.zeros((n_bins, result_tray.n_components))
            sum_w_squared = np.zeros_like(sum_w)
        else:
            sum_w = result_tray.sum_w
            sum_w_squared = result_tray.sum_w_squared

        digitized = np.digitize(X, bins=binning)
        sum_w[:, idx] = np.bincount(digitized,
                                    weights=weights,
                                    minlength=n_bins)
        if weights is not None:
            sum_w_squared[:, idx] = np.bincount(digitized,
                                                weights=weights**2,
                                                minlength=n_bins)
        else:
            sum_w_squared[:, idx] = sum_w[:, idx]
        result_tray.add(sum_w, 'sum_w')
        result_tray.add(sum_w_squared, 'sum_w_squared')
        return result_tray


class CalcAggarwalHistoErrors(CalcPart):
    name = 'CalcAggarwalHistoErrors'
    def __init__(self, alphas):
        self.alphas = alphas
        self.ref_idx = None

    def execute(self, result_tray, component):
        super(CalcPart, self).execute(result_tray, component)
        raise NotImplementedError

    def reset(self):
        super(CalcPart, self).reset()
        self.ref_idx = None


class CalcClassicHistoErrors(CalcPart):
    name = 'CalcClassicHistoErrors'
    def __init__(self):
        self.ref_idx = None

    def execute(self, result_tray, component):
        super(CalcPart, self).execute(result_tray, component)
        raise NotImplementedError

    def reset(self):
        super(CalcPart, self).reset()
        self.ref_idx = None


class PlotHistAggerwal(PlotPart):
    name = 'PlotHistAggerwal'
    rows = 5
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        super(PlotPart, self).execute(result_tray, component)
        raise NotImplementedError


class PlotHistClassic(PlotPart):
    name = 'PlotHistClassic'
    rows = 5
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        super(PlotPart, self).execute(result_tray, component)
        raise NotImplementedError


class PlotRatioAggerwal(PlotPart):
    name = 'PlotRatioAggerwal'
    rows = 1
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        super(PlotPart, self).execute(result_tray, component)
        raise NotImplementedError


class PlotRatioClassic(PlotPart):
    name = 'PlotRatioClassic'
    rows = 1
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        super(PlotPart, self).execute(result_tray, component)
        raise NotImplementedError

