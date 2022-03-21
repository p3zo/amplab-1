"""
Compare ground-truth valence/arousal annotations to predictions made by:
- Spotify's audio analysis API
- 6 pre-trained models
    - ded: DEAM dataset, Effnet-Discogs embeddings
    - dmm: DEAM dataset, MusiCNN-MSD embeddings
    - dva: DEAM dataset, VGGish-AudioSet embeddings
    - eed: EmoMusic dataset, Effnet-Discogs embeddings
    - emm: EmoMusic dataset, MusiCNN-MSD embeddings
    - eva: EmoMusic dataset, VGGish-AudioSet embeddings
"""

import argparse
import glob
import json
import os

import pandas as pd
import yaml
from constants import DATA_DIR, OUTPUT_DIR

ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
TRIPLETS_PATH = os.path.join(DATA_DIR, "triplets.all.chunk.001.pairs")
GROUND_TRUTH_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "ground-truth-annotations.csv")
SPOTIFY_ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations-spotifyapi.001")
SPOTIFY_PREDICTIONS_PATH = os.path.join(DATA_DIR, "spotify-predictions.csv")
ESSENTIA_PREDICTIONS_PATH = os.path.join(DATA_DIR, "essentia-models-predictions.csv")
PAIR_AGREEMENT_STATS_PATH = os.path.join(OUTPUT_DIR, "pairwise-agreements-by-model.csv")
CONSISTENT_AROUSAL_PAIR_AGREEMENT_STATS_PATH = os.path.join(
    OUTPUT_DIR, "pairwise-agreements-by-model-consistent-arousal.csv"
)
CONSISTENT_VALENCE_PAIR_AGREEMENT_STATS_PATH = os.path.join(
    OUTPUT_DIR, "pairwise-agreements-by-model-consistent-valence.csv"
)
PAIR_AGREEMENTS_PATH = os.path.join(OUTPUT_DIR, "pairwise-agreements.csv")

# Valence/arousal predictions can have 6 significant figures
# Define a tolerance within which to call them equal
# TODO: justify this tolerance value
EQUIVALENCE_TOLERANCE = 0.05


def create_ground_truth_annotation_csv():
    """Parse the raw ground truth annotations into a single CSV"""
    annotation_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))

    annotations_df = pd.DataFrame()
    for af in annotation_files:
        pair_id = "-".join(os.path.splitext(os.path.basename(af))[0].split("-")[2:])
        a_df = pd.read_json(af, orient="index").T
        a_df.index = [f"{pair_id}"]
        annotations_df = pd.concat([annotations_df, a_df])

    with open(TRIPLETS_PATH, "r") as jld:
        triplets = pd.DataFrame([json.loads(line) for line in jld.readlines()])

    triplets = triplets.set_index("pairid", drop=True)
    triplets = triplets.drop(["a_file", "b_file", "a_volume", "b_volume"], axis=1)

    joined = triplets.join(annotations_df)
    joined.to_csv(GROUND_TRUTH_ANNOTATIONS_PATH)
    print(f"Wrote parsed ground truth annotations to {GROUND_TRUTH_ANNOTATIONS_PATH}")


def load_ground_truth_annotations():
    """Load annotations parsed by `create_ground_truth_annotation_csv()`"""
    df = pd.read_csv(GROUND_TRUTH_ANNOTATIONS_PATH)

    df["triplet_id"] = df["pairid"].apply(lambda x: x[:32])

    df = df.set_index("pairid", drop=True)

    # Drop triplets that have any pair with a `not_selected` annotation
    # These are non-music tracks, like speech or noise
    triplets_no_arousal = df[df["higher_arousal"] == "not_selected"]["triplet_id"]
    triplets_no_valence = df[df["higher_valence"] == "not_selected"]["triplet_id"]

    triplet_ids_to_drop = set(triplets_no_arousal.values).union(
        triplets_no_valence.values
    )

    # Hard-code a triplet ID which contains a pair of duplicates (same song, different Spotify IDs)
    triplet_ids_to_drop.add("1b02f828d27049dfbae3caed2011ed46")

    df = df[~df["triplet_id"].isin(triplet_ids_to_drop)]

    return df


def create_spotify_annotation_csv():
    """Parse the raw spotify annotations into a single CSV"""
    spotify_annotation_files = glob.glob(
        os.path.join(SPOTIFY_ANNOTATIONS_DIR, "*/*.yaml")
    )

    df = pd.DataFrame()

    for sa_path in spotify_annotation_files:
        print(f"Loading {sa_path}")

        with open(sa_path, "r") as file:
            sf = yaml.safe_load(file)

        df_part = pd.DataFrame(
            [
                sf["track"]["id"],
                sf["track"]["preview_url"],
                sf["audio_features"]["valence"],
                sf["audio_features"]["energy"],
            ],
            index=["trackid", "preview_url", "valence", "arousal"],
        ).T

        df = pd.concat([df, df_part])
        df.to_csv(SPOTIFY_PREDICTIONS_PATH, index=False)
        print(f"Wrote parsed spotify annotations to {SPOTIFY_PREDICTIONS_PATH}")


def load_spotify_predictions():
    """Load spotify predictions parsed by `create_spotify_annotation_csv()`"""
    df = pd.read_csv(SPOTIFY_PREDICTIONS_PATH)
    return df.set_index("trackid", drop=True)


def load_essentia_model_predictions():
    """Load essentia-models predictions"""
    df = pd.read_csv(ESSENTIA_PREDICTIONS_PATH)
    df = df.set_index("trackid", drop=True)

    ded_df = df[df["model"] == "ded"]
    dmm_df = df[df["model"] == "dmm"]
    dva_df = df[df["model"] == "dva"]
    eed_df = df[df["model"] == "eed"]
    emm_df = df[df["model"] == "emm"]
    eva_df = df[df["model"] == "eva"]

    return [ded_df, dmm_df, dva_df, eed_df, emm_df, eva_df]


def get_higher(a, b, measure):
    """Takes two pandas.Series and compares them in a specific `measure`"""
    higher = "a"
    if abs(a[measure] - b[measure]) < EQUIVALENCE_TOLERANCE:
        higher = "equivalent"
    elif b[measure] > a[measure]:
        higher = "b"
    return higher


def get_inconsistent_measure(
    a_id, b_id, higher, order, triplet_id, inconsistent_triplets
):
    def add_to_sublist(el, ref_el, arr):
        ref_ix = get_sublist_index(ref_el, arr)
        arr[ref_ix].append(el)
        return arr

    def add_after_sublist(el, ref_el, arr):
        ref_ix = get_sublist_index(ref_el, arr)
        if ref_ix == len(arr) - 1:
            arr.append([el])
        else:
            arr[ref_ix + 1].append(el)
        return arr

    def add_before_sublist(el, ref_el, arr):
        ref_ix = get_sublist_index(ref_el, arr)
        if ref_ix == 0:
            arr.insert(0, [el])
        else:
            arr[ref_ix - 1].append(el)
        return arr

    def get_sublist_index(el, arr):
        for ix, sublist in enumerate(arr):
            if el in sublist:
                return ix
        return -1

    if not order:
        order.append([a_id])

    if get_sublist_index(a_id, order) == -1:
        if higher == "a":
            order = add_after_sublist(a_id, b_id, order)
        elif higher == "b":
            order = add_before_sublist(a_id, b_id, order)
        elif higher == "equivalent":
            order = add_to_sublist(a_id, b_id, order)

    if get_sublist_index(b_id, order) > -1:
        a_ix = get_sublist_index(a_id, order)
        b_ix = get_sublist_index(b_id, order)

        if higher == "b" and a_ix > b_ix:
            inconsistent_triplets.append(triplet_id)
        elif higher == "a" and a_ix < b_ix:
            inconsistent_triplets.append(triplet_id)

    else:
        if higher == "a":
            order = add_before_sublist(b_id, a_id, order)
        elif higher == "b":
            order = add_after_sublist(b_id, a_id, order)
        elif higher == "equivalent":
            order = add_to_sublist(b_id, a_id, order)

    return order, inconsistent_triplets


def get_inconsistent_triplets(df):
    """Find inconsistent triplets.

    Returns cases where (A > B, B > C, A < C) or (A < B, B < C, A > C)
    for either valence or arousal.
    """
    inconsistent_arousals = []
    inconsistent_valences = []

    seen_triplet_ids = []

    for gt_ix, gt_row in df.iterrows():
        triplet_id = gt_row["triplet_id"]

        if triplet_id in seen_triplet_ids:
            continue

        seen_triplet_ids.append(triplet_id)

        triplet_df = df[df["triplet_id"] == triplet_id]
        assert len(triplet_df) == 3

        arousal_order = []
        valence_order = []

        for t_ix, t_row in triplet_df.iterrows():
            a_id = t_row["a_id"]
            b_id = t_row["b_id"]
            ha = t_row["higher_arousal"]
            hv = t_row["higher_valence"]

            arousal_order, inconsistent_arousals = get_inconsistent_measure(
                a_id, b_id, ha, arousal_order, triplet_id, inconsistent_arousals
            )

            valence_order, inconsistent_valences = get_inconsistent_measure(
                a_id, b_id, hv, valence_order, triplet_id, inconsistent_valences
            )

    return inconsistent_arousals, inconsistent_valences


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reparse_annotations",
        action="store_true",
        help="Parse the raw ground-truth & spotify annotations into CSVs.",
    )
    args = parser.parse_args()

    if args.reparse_annotations:
        create_ground_truth_annotation_csv()
        create_spotify_annotation_csv()

    ground_truth_df = load_ground_truth_annotations()

    spotify_df = load_spotify_predictions()
    spotify_df = spotify_df.drop("preview_url", axis=1)
    spotify_df["model"] = "spotify"

    ded_df, dmm_df, dva_df, eed_df, emm_df, eva_df = load_essentia_model_predictions()

    all_model_df = pd.concat(
        [spotify_df, ded_df, dmm_df, dva_df, eed_df, emm_df, eva_df]
    )

    # Compare pairwise agreement between each model & ground truth
    agreement_df = pd.DataFrame()
    for gt_ix, gt_row in ground_truth_df.iterrows():
        arousal_agreements = {model: 0 for model in all_model_df.model.unique()}
        valence_agreements = arousal_agreements.copy()

        all_a = all_model_df.loc[gt_row["a_id"]]
        all_b = all_model_df.loc[gt_row["b_id"]]

        for ix, r in all_a.iterrows():
            model = r["model"]

            a = all_a[all_a["model"] == model].iloc[0]
            b = all_b[all_b["model"] == model].iloc[0]

            higher_arousal = get_higher(a, b, "arousal")
            higher_valence = get_higher(a, b, "valence")

            if higher_arousal == gt_row["higher_arousal"]:
                arousal_agreements[model] = 1

            if higher_valence == gt_row["higher_valence"]:
                valence_agreements[model] = 1

        aa = pd.Series(arousal_agreements)
        aa.index = [f"{i}_agreement_arousal" for i in aa.index]

        va = pd.Series(valence_agreements)
        va.index = [f"{i}_agreement_valence" for i in va.index]

        agreement_by_model = pd.DataFrame(aa.append(va)).T
        agreement_by_model.index = [gt_ix]

        agreement_df = pd.concat([agreement_df, agreement_by_model])

    n_pairs = len(agreement_df)

    agreement = (agreement_df.sum() / n_pairs).apply(lambda x: round(x, 2))
    agreement.to_csv(PAIR_AGREEMENT_STATS_PATH, header=["pct_agreement"])

    print(f"{n_pairs} pairs")
    print(agreement)
    print(f"Wrote pairwise agreement stats to {PAIR_AGREEMENT_STATS_PATH}")

    pair_agreements = ground_truth_df.join(agreement_df)
    pair_agreements.to_csv(PAIR_AGREEMENTS_PATH)
    print(f"Wrote pairwise agreements to {PAIR_AGREEMENTS_PATH}")

    print("Finding inconsistencies...")
    inconsistent_arousals, inconsistent_valences = get_inconsistent_triplets(
        ground_truth_df
    )
    print(f"  Found {len(inconsistent_arousals)} inconsitent triplet(s) in arousal")
    print(f"  Found {len(inconsistent_valences)} inconsitent triplet(s) in valence")

    # Remove inconsitent triplets and recalculate agreement stats
    agreement_df["triplet_id"] = [i[:32] for i in agreement_df.index]

    agreement_df_consistent_arousal = agreement_df[
        ~agreement_df["triplet_id"].isin(inconsistent_arousals)
    ]
    agreement_df_consistent_arousal = agreement_df_consistent_arousal.drop(
        "triplet_id", axis=1
    )

    n_pairs_consistent_arousal = len(agreement_df_consistent_arousal)
    agreement_consistent_arousal = (
        agreement_df_consistent_arousal.sum() / n_pairs_consistent_arousal
    ).apply(lambda x: round(x, 2))
    agreement_consistent_arousal.to_csv(
        CONSISTENT_AROUSAL_PAIR_AGREEMENT_STATS_PATH, header=["pct_agreement"]
    )
    print(f"{n_pairs_consistent_arousal} pairs consistent in arousal")
    print(agreement_consistent_arousal)
    print(
        f"Wrote pairwise agreement (consistent arousal) stats to {CONSISTENT_AROUSAL_PAIR_AGREEMENT_STATS_PATH}"
    )

    agreement_df_consistent_valence = agreement_df[
        ~agreement_df["triplet_id"].isin(inconsistent_valences)
    ]
    agreement_df_consistent_valence = agreement_df_consistent_valence.drop(
        "triplet_id", axis=1
    )

    n_pairs_consistent_valence = len(agreement_df_consistent_valence)
    agreement_consistent_valence = (
        agreement_df_consistent_valence.sum() / n_pairs_consistent_valence
    ).apply(lambda x: round(x, 2))
    agreement_consistent_valence.to_csv(
        CONSISTENT_VALENCE_PAIR_AGREEMENT_STATS_PATH, header=["pct_agreement"]
    )
    print(f"{n_pairs_consistent_valence} pairs consistent in valence")
    print(agreement_consistent_valence)
    print(
        f"Wrote pairwise agreement (consistent valence) stats to {CONSISTENT_VALENCE_PAIR_AGREEMENT_STATS_PATH}"
    )
