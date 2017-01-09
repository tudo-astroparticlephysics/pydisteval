class Part:
    def __init__(self):
        raise NotImplementedError

    def calc(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError


class CalcPart(Part):
    def plot(self, *args, **kwargs):
        raise RuntimeError('Trying to use a \'CalcPart\' for plotting')


class PlotPart(Part):
    def calc(self, *args, **kwargs):
        raise RuntimeError('Trying to use a \'CalcPart\' for calculation')


class CalcBinning(CalcPart):
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError


class CalcHistAggerwal(CalcPart):
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError

class CalcHistClassic(CalcPart):
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError

class CalcRatioAggerwal(CalcPart):
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError

class CalcRatioClassic(CalcPart):
    def __init__(self):
        pass

    def calc(self, df):
        raise NotImplementedError

class PlotHistAggerwal(PlotPart):
    def __init__(self):
        pass

    def plot(self, ax, calc_obj):
        raise NotImplementedError


class PlotHistClassic(PlotPart):
    def __init__(self):
        pass

    def plot(self, ax, calc_obj):
        raise NotImplementedError



class PlotRatioAggerwal(PlotPart):
    def __init__(self):
        pass

    def plot(self, ax, calc_obj):
        raise NotImplementedError



class PlotRatioClassic(PlotPart):
    def __init__(self):
        pass

    def plot(self, ax, calc_obj):
        raise NotImplementedError

