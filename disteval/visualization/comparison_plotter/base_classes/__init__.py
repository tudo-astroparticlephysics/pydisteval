class Part:
    name = 'BasePart'
    level = 1
    def __init__(self):
        raise NotImplementedError

    def calc(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def __lt__(self, other):
        return self.level < other.level

    def __eq__(self, other):
        return self.name == other.name


class CalcPart(Part):
    def plot(self, *args, **kwargs):
        raise RuntimeError('Trying to use a \'CalcPart\' for plotting')


class PlotPart(Part):
    def calc(self, *args, **kwargs):
        raise RuntimeError('Trying to use a \'CalcPart\' for calculation')


from ..parts import CalcBinning


class Element:
    name = 'DefaultElement'
    n_bins = 50
    plot_components = []
    calc_components = [CalcBinning(n_bins)]
    def __init__(self):
        pass

    def register(self, comparator):
        for calc_comp_i in Element.calc_components:
            comparator.register(calc_comp_i)
        for plot_comp_i in Element.plot_components:
            comparator.register(plot_comp_i)

    def __eq__(self, other):
        return self.name == other.name


