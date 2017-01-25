from matplotlib import pyplot as plt
from matplotlib.colors import Colormap


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


class ColorPalette:
    def __init__(self, color_cycle=None, cmap_cycle=None):
        if color_cycle is None:
            self.color_cycle = COLOR_CYCLE
        elif isinstance(color_cycle, list) or \
                isinstance(color_cycle, tuple):
            self.color_cycle = color_cycle
        self.color_pointer = 0

        if cmap_cycle is None:
            self.cmap_cycle = CMAP_CYCLE
        elif isinstance(cmap_cycle, list) or \
                isinstance(cmap_cycle, tuple):
            self.cmap_cycle = cmap_cycle
        self.cmap_pointer = 0

    def get_cmap(self):
        if self.cmap_pointer >= len(self.cmap_cycle):
            self.cmap_pointer = 0
        else:
            cmap = self.cmap_cycle[self.cmap_pointer]
            self.cmap_pointer += 1
        return cmap

    def get_color(self):
        if self.color_pointer >= len(self.color_cycle):
            self.color_pointer = 0
        else:
            cmap = self.color_cycle[self.color_pointer]
            self.color_pointer += 1
        return cmap

    def reset(self):
        self.color_pointer = 0
        self.cmap_pointer = 0


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
        self.color = color
        if c_type not in ['ref', 'test']:
            self.cmap = None
        elif isinstance(cmap, str):
            self.cmap = plt.get_cmap(cmap)
        elif isinstance(cmap, Colormap):
            self.cmap = cmap
        else:
            raise ValueError('cmap must be a colormap from matplotlib or '
                             'an instance of type matplotlib.colors.ColorMap')

    def __lt__(self, other):
        return ORDER.index(self.c_type) < ORDER.index(other.c_type)
