from matplotlib import pyplot as plt


ORDER = ['test', 'ref', 'ref_part', 'test_part']
COLOR_CYCLE = [(31, 119, 180),
               (255, 127, 14),
               (44, 160, 44),
               (214, 39, 40),
               (148, 103, 189),
               (140, 86, 75),
               (227, 119, 194),
               (127, 127, 127),
               (188, 189, 34),
               (23, 190, 207)]
COLOR_CYCLE = [(r / 255., g / 255., b / 255) for r, g, b in COLOR_CYCLE]

CMAP_CYCLE = ['viridis_r',
              'plasma_r',
              'magma_r',
              'inferno_r']


def get_cmap_name():
    get_cmap_name.pointer += 1
    if get_cmap_name.pointer >= len(CMAP_CYCLE):
        get_cmap_name.pointer = 0
    return CMAP_CYCLE[get_cmap_name.pointer]


get_cmap_name.pointer = -1


def get_color():
    get_color.pointer += 1
    if get_color.pointer >= len(COLOR_CYCLE):
        get_color.pointer = 0
    return COLOR_CYCLE[get_color.pointer]


get_color.pointer = -1


class Component:
    def __init__(self, idx, label, c_type, X, livetime=1,
                 weights=None, color=None, cmap=None):
        assert c_type in ['ref', 'ref_part', 'test', 'test_part'], \
            'Invalid c_type! Possible: [ref, ref_part, test, test_part]!'
        self.idx = idx
        self.label = label
        self.c_type = c_type
        self.X = X
        self.livetime = livetime
        self.weights = weights
        if color is None:
            color = get_color()
        self.color = color
        if cmap is None and c_type == 'ref':
            self.cmap = plt.get_cmap(get_cmap_name())
        elif c_type != 'ref':
            self.cmap = None
        else:
            self.cmap = cmap

    def __lt__(self, other):
        return ORDER.index(self.c_type) < ORDER.index(other.c_type)
