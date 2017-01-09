from .base_classes import Element

class AggarwalRatio(Element):
    name = 'AggarwalRatio'
    def __init__(self,):
        pass
    def register(self, comparator):
        super(AggarwalRatio, self).__init__(comparator)


    def run(self):
        raise NotImplementedError


class AggarwalHisto(Element):
    name = 'AggarwalHisto'
    def __init__(self):
        pass

    def register(self, comparator):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class ClassicHisto(Element):
    name = 'ClassicHisto'
    def __init__(self):
        pass

    def register(self, comparator):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class ClassicRatio(Element):
    name = 'ClassicRatio'
    def __init__(self):
        pass

    def register(self, comparator):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
