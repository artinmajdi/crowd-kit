__all__ = [
    'ROVER',
]
import crowdkit.aggregation.base
import pandas.core.frame
import pandas.core.series
import typing


class ROVER(crowdkit.aggregation.base.BaseTextsAggregator):
    """Recognizer Output Voting Error Reduction (ROVER)

    J. G. Fiscus,
    "A post-processing system to yield reduced word error rates: Recognizer Output Voting Error Reduction (ROVER),"
    1997 IEEE Workshop on Automatic Speech Recognition and Understanding Proceedings, 1997, pp. 347-354.
    https://doi.org/10.1109/ASRU.1997.659110
    Attributes:
        texts_ (Series): Tasks' texts
            A pandas.Series indexed by `task` such that `result.loc[task, text]`
            is the task's text.
    """

    def fit(self, data: pandas.core.frame.DataFrame) -> 'ROVER':
        """Args:
            data (DataFrame): Performers' text outputs
                A pandas.DataFrame containing `task`, `performer` and `text` columns.
        Returns:
            ROVER: self
        """
        ...

    def fit_predict(self, data: pandas.core.frame.DataFrame) -> pandas.core.series.Series:
        """Args:
            data (DataFrame): Performers' text outputs
                A pandas.DataFrame containing `task`, `performer` and `text` columns.
        Returns:
            Series: Tasks' texts
                A pandas.Series indexed by `task` such that `result.loc[task, text]`
                is the task's text.
        """
        ...

    def __init__(
        self,
        tokenizer: typing.Callable[[str], typing.List[str]],
        detokenizer: typing.Callable[[typing.List[str]], str],
        silent: bool = True
    ) -> None:
        """Method generated by attrs for class ROVER.
        """
        ...

    texts_: pandas.core.series.Series
    tokenizer: typing.Callable[[str], typing.List[str]]
    detokenizer: typing.Callable[[typing.List[str]], str]
    silent: bool
