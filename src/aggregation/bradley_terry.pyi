__all__ = [
    'BradleyTerry',
]
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from toloka.client.base_aggregator import BaseAggregator

class BradleyTerry(BaseAggregator):
    """Bradley-Terry, the classic algorithm for aggregating pairwise comparisons.

    David R. Hunter. 2004.
    MM algorithms for generalized Bradley-Terry models
    Ann. Statist., Vol. 32, 1 (2004): 384–406.

    Bradley, R. A. and Terry, M. E. 1952.
    Rank analysis of incomplete block designs. I. The method of paired comparisons.
    Biometrika, Vol. 39 (1952): 324–345.
    Attributes:
        result_ (Series): 'Labels' scores
            A pandas.Series index by labels and holding corresponding label's scores
    """

    def __init__(self, n_iter: int) -> None:
        """Method generated by attrs for class BradleyTerry.
        """
        ...

    def fit(self, data: DataFrame) -> 'BradleyTerry':
        """Args:
            data (DataFrame): Performers' pairwise comparison results
                A pandas.DataFrame containing `performer`, `left`, `right`, and `label` columns'.
                For each row `label` must be equal to either `left` or `right`.

        Returns:
            BradleyTerry: self
        """
        ...

    def fit_predict(self, data: DataFrame) -> Series:
        """Args:
            data (DataFrame): Performers' pairwise comparison results
                A pandas.DataFrame containing `performer`, `left`, `right`, and `label` columns'.
                For each row `label` must be equal to either `left` or `right`.

        Returns:
            Series: 'Labels' scores
                A pandas.Series index by labels and holding corresponding label's scores
        """
        ...

    n_iter: int
    result_: Series
