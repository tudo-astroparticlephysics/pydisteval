from .base_classes import Element
from . import parts


class AggarwalHisto(Element):
    name = 'AggarwalHisto'

    def __init__(self,
                 n_bins=50,
                 log_y=True,
                 alphas=[0.68, 0.9, 0.95]):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcAggarwalHistoErrors(alphas))
        self.plot_components.append(parts.PlotHistAggerwal(log_y=log_y,
                                                           alphas=alphas))


class ClassicHisto(Element):
    name = 'ClassicHisto'

    def __init__(self,
                 n_bins=50,
                 log_y=True,
                 normalize=False,
                 bands=False,
                 band_borders=False,
                 band_brighten=False,
                 band_alpha=0.0):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcClassicHistoErrors())

        plot_hist = parts.PlotHistClassic(log_y=log_y,
                                          normalize=normalize,
                                          bands=bands,
                                          band_borders=band_borders,
                                          band_brighten=band_brighten,
                                          band_alpha=band_alpha)
        self.plot_components.append(plot_hist)


class AggarwalRatio(Element):
    name = 'AggarwalRatio'
    def __init__(self, n_bins=50, alphas=[0.68, 0.9, 0.95]):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcAggarwalHistoErrors(alphas))
        self.plot_components.append(parts.PlotRatioAggerwal())


class ClassicRatio(Element):
    name = 'ClassicRatio'

    def __init__(self,
                 n_bins=50,
                 bands=False,
                 band_borders=False,
                 band_brighten=False,
                 band_alpha=0.0):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcClassicHistoErrors())
        plot_ratio = parts.PlotRatioClassic(bands=bands,
                                            band_borders=band_borders,
                                            band_brighten=band_brighten,
                                            band_alpha=band_alpha)
        self.plot_components.append(plot_ratio)
