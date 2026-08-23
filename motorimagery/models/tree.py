"""Depth-limited decision-tree classifier."""

from sklearn.tree import DecisionTreeClassifier


def build_depth4_tree(random_state: int = 2020) -> DecisionTreeClassifier:
    """Return the fixed depth-4 decision tree requested by the pipeline."""
    return DecisionTreeClassifier(
        max_depth=4,
        random_state=random_state,
    )

