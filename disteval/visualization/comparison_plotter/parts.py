from .base_classes import CalcPart, PlotPart

class CalcBinning(CalcPart):
    name = 'CalcBinning'
    level = 0
    def __init__(self, n_bins):
        pass

    def calc(self, df):
        raise NotImplementedError


class CalcHistAggerwal(CalcPart):
    name = 'CalcHistAggerwal'
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError

class CalcHistClassic(CalcPart):
    name = 'CalcHistClassic'
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError

class CalcRatioAggerwal(CalcPart):
    name = 'CalcRatioAggerwal'
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError

class CalcRatioClassic(CalcPart):
    name = 'CalcRatioClassic'
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError

class PlotHistAggerwal(PlotPart):
    name = 'PlotHistAggerwal'
    def __init__(self):
        pass

    def plot(self, ax, calc_obj):
        raise NotImplementedError


class PlotHistClassic(PlotPart):
    name = 'PlotHistClassic'
    def __init__(self):
        pass

    def plot(self, ax, calc_obj):
        raise NotImplementedError



class PlotRatioAggerwal(PlotPart):
    name = 'PlotRatioAggerwal'
    def __init__(self):
        pass

    def plot(self, ax, calc_obj):
        raise NotImplementedError


class PlotRatioClassic(PlotPart):
    name = 'PlotRatioClassic'
    def __init__(self):
        pass

    def plot(self, ax, calc_obj):
        raise NotImplementedError

