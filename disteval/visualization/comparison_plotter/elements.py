class Element:
    def __init__(self):
        raise NotImplementedError

    def register(self, comparator):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class AggarwalRatio(Element):
    def __init__(self):
        pass

    def register(self, comparator):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class AggarwalHisto(Element):
    def __init__(self):
        pass

    def register(self, comparator):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class ClassicHisto(Element):
    def __init__(self):
        pass

    def register(self, comparator):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class ClassicRatio(Element):
    def __init__(self):
        pass

    def register(self, comparator):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError
