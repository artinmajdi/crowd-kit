__all__ = [
    'ZeroBasedSkill',
]
import crowdkit.aggregation.base
import pandas.core.frame
import pandas.core.series
import typing


class ZeroBasedSkill(crowdkit.aggregation.base.BaseClassificationAggregator):
    """The Zero-Based Skill aggregation model

    Performs weighted majority voting on tasks. After processing a pool of tasks,
    re-estimates performers' skills according to the correctness of their answers.
    Repeats this process until labels do not change or the number of iterations exceeds.

    It's necessary that all performers in a dataset that send to 'predict' existed in answers
    the dataset that was sent to 'fit'.
    """

    def fit(self, data: pandas.core.frame.DataFrame) -> 'ZeroBasedSkill':
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
        Returns:
            ZeroBasedSkill: self
        """
        ...

    def predict(self, data: pandas.core.frame.DataFrame) -> pandas.core.series.Series:
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
        Returns:
            Series: Tasks' labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def predict_proba(self, data: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
        Returns:
            DataFrame: Tasks' label probability distributions
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        ...

    def fit_predict(self, data: pandas.core.frame.DataFrame) -> pandas.core.series.Series:
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
        Returns:
            Series: Tasks' labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def fit_predict_proba(self, data: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
        Returns:
            DataFrame: Tasks' label probability distributions
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        ...

    def __init__(
        self,
        n_iter: int = 100,
        lr_init: float = ...,
        lr_steps_to_reduce: int = 20,
        lr_reduce_factor: float = ...,
        eps: float = ...
    ) -> None:
        """Method generated by attrs for class ZeroBasedSkill.
        """
        ...

    labels_: typing.Optional[pandas.core.series.Series]
    n_iter: int
    lr_init: float
    lr_steps_to_reduce: int
    lr_reduce_factor: float
    eps: float
    skills_: ...
    probas_: ...
