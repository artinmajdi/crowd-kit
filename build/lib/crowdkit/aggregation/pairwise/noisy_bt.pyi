__all__ = [
    'NoisyBradleyTerry',
]
import crowdkit.aggregation.base
import pandas.core.frame
import pandas.core.series


class NoisyBradleyTerry(crowdkit.aggregation.base.BasePairwiseAggregator):
    """A modification of Bradley-Terry with parameters for performers' skills and
    their biases.
    Attributes:
        scores_ (Series): 'Labels' scores
            A pandas.Series index by labels and holding corresponding label's scores
        skills_ (Series): Performers' skills
            A pandas.Series index by performers and holding corresponding performer's skill
        biases_ (Series): Predicted biases for each performer. Indicates the probability of a performer to choose the left item.
            A series of performers' biases indexed by performers
    """

    def fit(self, data: pandas.core.frame.DataFrame) -> 'NoisyBradleyTerry':
        """Args:
            data (DataFrame): Performers' pairwise comparison results
                A pandas.DataFrame containing `performer`, `left`, `right`, and `label` columns'.
                For each row `label` must be equal to either `left` column or `right` column.

        Returns:
            NoisyBradleyTerry: self
        """
        ...

    def fit_predict(self, data: pandas.core.frame.DataFrame) -> pandas.core.series.Series:
        """Args:
            data (DataFrame): Performers' pairwise comparison results
                A pandas.DataFrame containing `performer`, `left`, `right`, and `label` columns'.
                For each row `label` must be equal to either `left` column or `right` column.

        Returns:
            Series: 'Labels' scores
                A pandas.Series index by labels and holding corresponding label's scores
        """
        ...

    def __init__(
        self,
        n_iter: int = 100,
        random_state: int = 0
    ) -> None:
        """Method generated by attrs for class NoisyBradleyTerry.
        """
        ...

    scores_: pandas.core.series.Series
    n_iter: int
    random_state: int
    skills_: pandas.core.series.Series
    biases_: pandas.core.series.Series
