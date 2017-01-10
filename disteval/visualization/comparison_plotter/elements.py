from .base_classes import Element
from . import parts


class AggarwalHisto(Element):
    name = 'AggarwalHisto'

    def __init__(self, n_bins=50, log_y=True, alphas=[0.68, 0.9, 0.95]):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcAggarwalHistoErrors(alphas))
        self.plot_components.append(parts.PlotHistAggerwal(log_y=log_y,
                                                           alphas=alphas))


class ClassicHisto(Element):
    name = 'ClassicHisto'

    def __init__(self, n_bins=50, log_y=True):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcClassicHistoErrors())
        self.plot_components.append(parts.PlotHistClassic(log_y=log_y))


class AggarwalRatio(Element):
    name = 'AggarwalRatio'
    def __init__(self, n_bins=50, alphas=[0.68, 0.9, 0.95]):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcAggarwalHistoErrors(alphas))
        self.plot_components.append(parts.PlotRatioAggerwal())


class ClassicRatio(Element):
    name = 'ClassicRatio'

    def __init__(self, n_bins=50):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcClassicHistoErrors())
        self.plot_components.append(parts.PlotRatioClassic())
