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
        self.n_components = 0

class CalcHistogram(CalcPart):
    name = 'CalcHistogram'
    level = 1
    def execute(self, result_tray, component):
        binning = result_tray.binning
        if not hasattr(result_tray, 'histo'):
            n_components = result_tray.n_components
            histo = np.zeros((len(binning) - 1, n_components))
        else:
            histo = result_tray.histo
        histo[:, component.idx] = np.histogram(component.X,
                                               weights=component.weights,
                                               bins=binning)[0]
        result_tray.add(histo, 'histo')
        return result_tray

class CalcAggarwalHistoErrors(CalcPart):
    name = 'CalcAggarwalHistoErrors'
    def __init__(self, alphas):
        self.alphas = alphas
        self.ref_idx = None

    def execute(self, result_tray, component):
        raise NotImplementedError

    def reset(self):
        self.ref_idx = None

class CalcClassicHistoErrors(CalcPart):
    name = 'CalcClassicHistoErrors'
    def __init__(self):
        self.ref_idx = None

    def execute(self, result_tray, component):
        raise NotImplementedError

    def reset(self):
        self.ref_idx = None


class PlotHistAggerwal(PlotPart):
    name = 'PlotHistAggerwal'
    rows = 5
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        raise NotImplementedError


class PlotHistClassic(PlotPart):
    name = 'PlotHistClassic'
    rows = 5
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        raise NotImplementedError


class PlotRatioAggerwal(PlotPart):
    name = 'PlotRatioAggerwal'
    rows = 1
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        raise NotImplementedError


class PlotRatioClassic(PlotPart):
    name = 'PlotRatioClassic'
    rows = 1
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        raise NotImplementedError

