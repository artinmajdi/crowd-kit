from pandas.core.frame import DataFrame
from pandas.core.series import Series
from typing import ClassVar, Tuple, Union, Optional

class BaseAggregator:
    """Base functions and fields for all aggregators
    Attributes:
        tasks_labels (typing.ClassVar[typing.Optional[pandas.core.frame.DataFrame]]): Estimated labels
            A pandas.DataFrame indexed by `task` with a single column `label` containing
            `tasks`'s most probable label for last fitted data, or None otherwise.

        probas (typing.ClassVar[typing.Optional[pandas.core.frame.DataFrame]]): Estimated label probabilities
            A frame indexed by `task` and a column for every label id found
            in `data` such that `result.loc[task, label]` is the probability of `task`'s
            true label to be equal to `label`.

        performers_skills (typing.ClassVar[typing.Optional[pandas.core.series.Series]]): Predicted skills for each performer
            A series of performers' skills indexed by performers"""

    tasks_labels: ClassVar[Optional[DataFrame]]
    probas: ClassVar[Optional[DataFrame]]
    performers_skills: ClassVar[Optional[Series]]

    def __init__(self) -> None:
        """Method generated by attrs for class BaseAggregator."""
        ...

    def _answers_base_checks(self, answers: DataFrame) -> None:
        """Checks basic 'answers' dataset requirements"""
        ...

    def _calc_performers_skills(self, answers: DataFrame, task_truth: DataFrame) -> Series:
        """Calculates skill for each performer

        Note:
            There can be only one * correct label *

        Args:
            answers (pandas.DataFrame): performers answers for tasks
                Should contain columns 'task', 'performer', 'label'
            task_truth (pandas.DataFrame): label regarding which to count the skill
                Should contain columns 'task', 'label'
                Could contain column 'weight'Returns:
            Series: Predicted skills for each performer
                A series of performers' skills indexed by performers"""
        ...

    def _calculate_probabilities(self, estimated_answers: DataFrame) -> DataFrame:
        """Calculate probabilities for each task for each label

        Note:
            All "score" must be positive.
            If the sum of scores for a task is zero, then all probabilities for this task will be NaN.

        Args:
            estimated_answers(pandas.DataFrame): Frame with "score" for each pair task-label.
                Should contain columns 'score', 'task', 'label'Returns:
            DataFrame: Estimated label probabilities
                A frame indexed by `task` and a column for every label id found
                in `data` such that `result.loc[task, label]` is the probability of `task`'s
                true label to be equal to `label`."""
        ...

    def _choose_labels(self, labels_probas: DataFrame) -> DataFrame:
        """Selection of the labels with the most probalitities
        Args:
            labels_probas (DataFrame): Estimated label probabilities
                A frame indexed by `task` and a column for every label id found
                in `data` such that `result.loc[task, label]` is the probability of `task`'s
                true label to be equal to `label`.

        Returns:
            DataFrame: Estimated labels
                A pandas.DataFrame indexed by `task` with a single column `label` containing
                `tasks`'s most probable label for last fitted data, or None otherwise."""
        ...

    @staticmethod
    def _max_probas_random_on_ties(x: Union[DataFrame, Series]) -> Tuple[str, float]:
        """Chooses max 'proba' value and return 'label' from same rows
        If several rows have same 'proba' - choose random"""
        ...
