from . import elements


class ComparisonPlotter:
    def __init__(self):
        pass

    def add_plot_element(self, element):
        """Function to add a plot element.

        Parameters
        ----------
        element : disteval.visualization.comparison_plotter.Element
            Element
        """
        raise NotImplementedError

    def plot(self, fig=None):
        """Function to add a data component.

        Parameters
        ----------
        fig : matplotlib.Figure, optional, (default=None)
            Label of the Component

        outpath : str, optional, (default=None)
            Input Dataframe

        Returns
        -------
        fig : matplotlib.Figure
            Figure with the result plot.

        axes : list of matplotlib.axis
            List containing all axes of the result plot.

        """
        raise NotImplementedError

    def __calc__(self, ref_df, test_df, test_parts=None, ref_parts=None):
        """Function to add a data component.

        Parameters
        ----------
        fig : matplotlib.Figure, optional, (default=None)
            Label of the Component

        outpath : str, optional, (default=None)
            Input Dataframe

        Returns
        -------
        fig : matplotlib.Figure
            Figure with the result plot.

        axes : list of matplotlib.axis
            List containing all axes of the result plot.

        """
        raise NotImplementedError
