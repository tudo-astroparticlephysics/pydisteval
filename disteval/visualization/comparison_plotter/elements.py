from .base_classes import Element
from . import parts


class AggarwalHisto(Element):
    name = 'AggarwalHisto'

    def __init__(self,
                 n_bins=50,
                 log_y=True,
                 alpha=[0.68, 0.9, 0.95],
                 bands=False,
                 band_borders=True,
                 band_brighten=True,
                 band_alpha=0.5):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcAggarwalHistoErrors(alpha))
        plot_hist = parts.PlotHistAggerwal(log_y=log_y,
                                           bands=bands,
                                           band_borders=band_borders,
                                           band_brighten=band_brighten,
                                           band_alpha=band_alpha)
        self.plot_components.append(plot_hist)


class ClassicHisto(Element):
    name = 'ClassicHisto'
    def __init__(self,
                 n_bins=50,
                 log_y=True,
                 bands=False,
                 band_borders=True,
                 band_brighten=True,
                 band_alpha=0.5):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcClassicHistoErrors())
        plot_hist = parts.PlotHistClassic(log_y=log_y,
                                          bands=bands,
                                          band_borders=band_borders,
                                          band_brighten=band_brighten,
                                          band_alpha=band_alpha)
        self.plot_components.append(plot_hist)


class AggarwalRatio(Element):
    name = 'AggarwalRatio'
    def __init__(self, n_bins=50, alpha=[0.68, 0.9, 0.95], zoomed=True):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcAggarwalHistoErrors(alpha))
        self.plot_components.append(parts.PlotRatioAggerwal(zoomed=zoomed))


class ClassicRatio(Element):
    name = 'ClassicRatio'

    def __init__(self,
                 n_bins=50,
                 bands=False,
                 band_borders=True,
                 band_brighten=True,
                 band_alpha=0.5,
                 y_lims=None,
                 y_label=r'$\frac{\mathregular{Test - Ref}}{\sigma}$'):
        self.calc_components.append(parts.CalcBinning(n_bins=n_bins))
        self.calc_components.append(parts.CalcHistogram())
        self.calc_components.append(parts.CalcClassicHistoErrors())
        plot_ratio = parts.PlotRatioClassic(bands=bands,
                                            band_borders=band_borders,
                                            band_brighten=band_brighten,
                                            band_alpha=band_alpha,
                                            y_lims=y_lims,
                                            y_label=y_label)
        self.plot_components.append(plot_ratio)

class Normalization(Element):
    name = 'Normalization'
    def __init__(self, normalize=None):
        if normalize is None:
            normalize = False
        if isinstance(normalize, bool):
            if normalize:
                normalize = 'sum_w'
            if isinstance(normalize, str):
                normalize = normalize.lower()
                if not normalize in ['test_livetime', 'livetime', 'sum_w']:
                    raise AttributeError('Possible values for \'normalize\': '
                                         '[\'test_livetime\', \'livetime\', '
                                         '\'sum_w\', True, False]!')
        self.calc_components.append(parts.CalcNormalization(normalize))
