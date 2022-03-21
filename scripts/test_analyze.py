import pandas as pd
from analyze import get_inconsistent_triplets


def test_get_inconsistent_triplets():

    consistent_examples = [
        [
            ["1", "abc", "jfk", "a", "a"],
            ["1", "abc", "xyz", "a", "a"],
            ["1", "jfk", "xyz", "a", "a"],
        ],
        [
            ["2", "abc", "jfk", "a", "a"],
            ["2", "abc", "xyz", "a", "a"],
            ["2", "jfk", "xyz", "equivalent", "equivalent"],
        ],
    ]

    inconsistent_examples = [
        # AB / AC / --
        [
            ["1", "abc", "jfk", "a", "a"],
            ["1", "abc", "xyz", "b", "b"],
            ["1", "jfk", "xyz", "a", "a"],
        ],
        [
            ["2", "abc", "jfk", "a", "a"],
            ["2", "abc", "xyz", "b", "b"],
            ["2", "xyz", "jfk", "b", "b"],
        ],
        # AB / CB / --
        [
            ["3", "abc", "jfk", "a", "a"],
            ["3", "xyz", "jfk", "b", "b"],
            ["3", "abc", "xyz", "b", "b"],
        ],
        [
            ["4", "abc", "jfk", "a", "a"],
            ["4", "xyz", "jfk", "b", "b"],
            ["4", "xyz", "abc", "a", "a"],
        ],
        # AB / BC / --
        [
            ["5", "abc", "jfk", "a", "a"],
            ["5", "jfk", "xyz", "a", "a"],
            ["5", "abc", "xyz", "b", "b"],
        ],
        [
            ["6", "abc", "jfk", "a", "a"],
            ["6", "jfk", "xyz", "a", "a"],
            ["6", "xyz", "abc", "a", "a"],
        ],
    ]

    for ex in consistent_examples:
        print(ex)
        df = pd.DataFrame(
            ex,
            columns=[
                "triplet_id",
                "a_id",
                "b_id",
                "higher_arousal",
                "higher_valence",
            ],
        )

        inconsistent_arousals, inconsistent_valences = get_inconsistent_triplets(df)

        assert len(inconsistent_arousals) == 0
        assert len(inconsistent_valences) == 0

    for ex in inconsistent_examples:
        print(ex)
        df = pd.DataFrame(
            ex,
            columns=[
                "triplet_id",
                "a_id",
                "b_id",
                "higher_arousal",
                "higher_valence",
            ],
        )

        inconsistent_arousals, inconsistent_valences = get_inconsistent_triplets(df)

        assert len(inconsistent_arousals) == 1
        assert len(inconsistent_valences) == 1


if __name__ == "__main__":
    test_get_inconsistent_triplets()
