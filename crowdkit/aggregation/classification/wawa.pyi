__all__ = [
    'Wawa',
]
import crowdkit.aggregation.base
import pandas
import typing


class Wawa(crowdkit.aggregation.base.BaseClassificationAggregator):
    """Worker Agreement with Aggregate.

    This algorithm does three steps:
    1. Calculate the majority vote label
    2. Estimate workers' skills as a fraction of responses that are equal to the majority vote
    3. Calculate the weigthed majority vote based on skills from the previous step

    Examples:
        >>> from crowdkit.aggregation import Wawa
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> result = Wawa().fit_predict(df)
    Attributes:
        labels_ (typing.Optional[pandas.core.series.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's most likely true label.

        skills_ (typing.Optional[pandas.core.series.Series]): workers' skills.
            A pandas.Series index by workers and holding corresponding worker's skill
        probas_ (typing.Optional[pandas.core.frame.DataFrame]): Tasks' label probability distributions.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1
    """

    def fit(self, data: pandas.DataFrame) -> 'Wawa':
        """Fit the model.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            Wawa: self.
        """
        ...

    def predict(self, data: pandas.DataFrame) -> pandas.Series:
        """Infer the true labels when the model is fitted.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def predict_proba(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Return probability distributions on labels for each task when the model is fitted.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        ...

    def fit_predict(self, data: pandas.DataFrame) -> pandas.Series:
        """Fit the model and return aggregated results.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def fit_predict_proba(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Fit the model and return probability distributions on labels for each task.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        ...

    def __init__(self) -> None:
        """Method generated by attrs for class Wawa.
        """
        ...

    labels_: typing.Optional[pandas.Series]
    skills_: typing.Optional[pandas.Series]
    probas_: typing.Optional[pandas.DataFrame]
