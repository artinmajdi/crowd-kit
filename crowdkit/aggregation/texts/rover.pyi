__all__ = [
    'ROVER',
]
import crowdkit.aggregation.base
import pandas
import typing


class ROVER(crowdkit.aggregation.base.BaseTextsAggregator):
    """Recognizer Output Voting Error Reduction (ROVER).

    This method uses dynamic programming to align sequences. Next, aligned sequences are used
    to construct the Word Transition Network (WTN):
    ![ROVER WTN scheme](http://tlk.s3.yandex.net/crowd-kit/docs/rover.png)
    Finally, the aggregated sequence is the result of majority voting on each edge of the WTN.

    J. G. Fiscus,
    "A post-processing system to yield reduced word error rates: Recognizer Output Voting Error Reduction (ROVER),"
    *1997 IEEE Workshop on Automatic Speech Recognition and Understanding Proceedings*, 1997, pp. 347-354.
    https://doi.org/10.1109/ASRU.1997.659110

    Args:
        tokenizer: A callable that takes a string and returns a list of tokens.
        detokenizer: A callable that takes a list of tokens and returns a string.
        silent: If false, show a progress bar.

    Examples:
        >>> from crowdkit.aggregation import load_dataset
        >>> from crowdkit.aggregation import ROVER
        >>> df, gt = load_dataset('crowdspeech-test-clean')
        >>> df['text'] = df['text].apply(lambda s: s.lower())
        >>> tokenizer = lambda s: s.split(' ')
        >>> detokenizer = lambda tokens: ' '.join(tokens)
        >>> result = ROVER(tokenizer, detokenizer).fit_predict(df)
    Attributes:
        texts_ (Series): Tasks' texts.
            A pandas.Series indexed by `task` such that `result.loc[task, text]`
            is the task's text.
    """

    def fit(self, data: pandas.DataFrame) -> 'ROVER':
        """Fits the model. The aggregated results are saved to the `texts_` attribute.
        Args:
            data (DataFrame): Workers' text outputs.
                A pandas.DataFrame containing `task`, `worker` and `text` columns.
        Returns:
            ROVER: self.
        """
        ...

    def fit_predict(self, data: pandas.DataFrame) -> pandas.Series:
        """Fit the model and return the aggregated texts.
        Args:
            data (DataFrame): Workers' text outputs.
                A pandas.DataFrame containing `task`, `worker` and `text` columns.
        Returns:
            Series: Tasks' texts.
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

    texts_: pandas.Series
    tokenizer: typing.Callable[[str], typing.List[str]]
    detokenizer: typing.Callable[[typing.List[str]], str]
    silent: bool
