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
AGREEMENT_STATS_PATH = os.path.join(OUTPUT_DIR, "agreements-by-model.csv")
PAIR_AGREEMENTS_PATH = os.path.join(OUTPUT_DIR, "pair-agreements.csv")

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
    return df.set_index("pairid", drop=True)


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

    # Compare agreement between each model & ground truth
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
    agreement.to_csv(AGREEMENT_STATS_PATH, header=["pct_agreement"])

    print(f"For {n_pairs} pairs")
    print(agreement)
    print(f"Wrote agreement stats to {AGREEMENT_STATS_PATH}")

    pair_agreements = ground_truth_df.join(agreement_df)
    pair_agreements.to_csv(PAIR_AGREEMENTS_PATH)
    print(f"Wrote pair agreements to {PAIR_AGREEMENTS_PATH}")
