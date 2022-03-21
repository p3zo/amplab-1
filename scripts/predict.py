"""
Predict arousal/valence of audio using six pre-trained, transfer-learning models.

Models were trained on two datasets (DEAM, EmoMusic) using three types of deep
embeddings available in Essentia (MusiCNN-MSD, VGGish-AudioSet, Effnet-Discogs).
"""

import glob
import json
import os
import time

import essentia
import pandas as pd
from constants import AUDIO_DIR, DATA_DIR, MODELS_DIR
from essentia.standard import (
    MonoLoader,
    TensorflowPredict,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredictMusiCNN,
    TensorflowPredictVGGish,
)

# disable warnings from essentia-tensorflow
essentia.log.warningActive = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

EFFNET_DISCOGS_MODEL_PATH = os.path.join(MODELS_DIR, "effnet-discogs-1.pb")
MSD_MUSICNN_MODEL_PATH = os.path.join(MODELS_DIR, "msd-musicnn-1.pb")
AUDIOSET_VGGISH_MODEL_PATH = os.path.join(MODELS_DIR, "audioset-vggish-3.pb")
ESSENTIA_PREDICTIONS_PATH = os.path.join(DATA_DIR, "essentia-models-predictions.csv")


def load_effnet_discogs_embeddings_model():
    # For configuring MusiCNN and VGGish embeddings, see:
    # https://essentia.upf.edu/tutorial_tensorflow_auto-tagging_classification_embeddings.html

    # Parameters for the Effnet-Discogs embeddings model. It requires patch size
    # and patch hop size different from MusiCNN.

    # The patch_size defines the number of melscpectrogram frames the model needs to return an
    # embedding vector. Effnet model needs 128 frames and each frame is extracted with a 256-samples
    # hopsize. That's why the receptive field of Effnet is 2-seconds, 128 * 256/ 16000 ≈ 2 seconds,
    # and it differs of MusiCNN which utilizes 187 frames (3-seconds).
    patch_size = 128

    # The patch_hop_size defines the number of feature frames in between successive embedding.
    # It defines the time interval analysed at each embedding vector and it can vary for each model.
    # Effnet applies 64 patches, close to 1-seconds (64 * 256 / 16000),
    # whereas MusiCNN applies 1.5-seconds (93 * 256/ 16000 ≈ 1.5)
    patch_hop_size = patch_size // 2

    # Although Effnet-Discogs was published recently in
    # https://essentia.upf.edu/models/music-style-classification/discogs-effnet/
    # deam-effnet-discogs-1 model was trained with a previous version.
    # NOTE: The embedding size and the layer names differ among versions.

    input_layer = "melspectrogram"
    output_layer = "onnx_tf_prefix_BatchNormalization_496/add_1"

    # Instantiate the embeddings model
    return TensorflowPredictEffnetDiscogs(
        graphFilename=EFFNET_DISCOGS_MODEL_PATH,
        input=input_layer,
        output=output_layer,
        patchSize=patch_size,
        patchHopSize=patch_hop_size,
    )


def load_msd_musicnn_embeddings_model():
    # For configuring MusiCNN and VGGish embeddings, see:
    # https://essentia.upf.edu/tutorial_tensorflow_auto-tagging_classification_embeddings.html

    # Parameters for the MusiCNN-MSD embeddings model. It requires patch size
    # and patch hop size different from Effnet.

    # The patch_size defines the number of melscpectrogram frames the model needs
    # to return an embedding vector. MusiCNN model needs 187 frames and
    # each frame is extracted with a 256-samples hopsize.
    # That's why the receptive field of MusiCNN is 3-seconds, 187 * 256/ 16000 ≈ 3 seconds,
    # and it differs of Effnet which utilizes 128 frames (2-seconds).
    patch_size = 187

    # The patch_hop_size defines the number of feature frames in between successive embedding.
    # It defines the time interval analysed at each embedding vector and it can vary for each model.
    # MusiCNN applies 93 patches, close to 1.5-seconds (93 * 256/ 16000 ≈ 1.5).
    # whereas Effnet applies 1-seconds (64 * 256 / 16000 ≈ 1),
    patch_hop_size = patch_size // 2

    input_layer = "model/Placeholder"
    output_layer = "model/dense/BiasAdd"

    # Instantiate the embeddings model
    return TensorflowPredictMusiCNN(
        graphFilename=MSD_MUSICNN_MODEL_PATH,
        input=input_layer,
        output=output_layer,
        patchHopSize=patch_hop_size,
    )


def load_audioset_vggish_embeddings_model():
    # Parameters for the VGGish embeddings model. It works in time domain,
    # it doesn't needs to specify patch_size and patch_hop_size, only output_layer name.
    output_layer = "model/vggish/embeddings"

    # Instantiate the embeddings model
    return TensorflowPredictVGGish(
        graphFilename=AUDIOSET_VGGISH_MODEL_PATH,
        output=output_layer,
    )


def load_model_input_output_layers(model_name):
    """Load model metadata and parse input/output layers"""

    av_model_config_path = os.path.join(MODELS_DIR, model_name, f"{model_name}.json")
    metadata = json.load(open(av_model_config_path, "r"))

    input_layer = metadata["schema"]["inputs"][0]["name"]
    output_layer = metadata["schema"]["outputs"][0]["name"]

    return input_layer, output_layer


def load_av_model(model_name):
    # First we need to configure the input and output layers for this model
    input_layer, output_layer = load_model_input_output_layers(model_name)

    # Instantiate the arousal-valence model
    model = TensorflowPredict(
        graphFilename=os.path.join(MODELS_DIR, model_name, f"{model_name}.pb"),
        inputs=[input_layer],
        outputs=[output_layer],
    )

    return model, input_layer, output_layer


def predict(embeddings, av_model, av_input_layer, av_output_layer):
    """Runs inference with the provided model.

    Returns a prediction in the form of [valence, arousal]
    """
    # Run inference

    # TensorflowPredict works with an input Essentia Pool. Typically you won't
    # use the TensorFlowPredict algorithm directly with many of our models as we
    # provide wrappers (for example, TensorflowPredictMusiCNN). However this is
    # not the case for the new arousal-valence models, so we need to manually
    # prepare the input features

    feature = embeddings.reshape(-1, 1, 1, embeddings.shape[1])

    pool = essentia.Pool()
    pool.set(av_input_layer, feature)

    predictions = av_model(pool)[av_output_layer].squeeze()

    # We estimate the average of the predictions to get an arousal-valence
    # representation for the entire song
    prediction = predictions.mean(axis=0)  # [valence, arousal]

    print(f"prediction: {prediction}")
    return prediction


if __name__ == "__main__":
    # Load embeddings models
    effnet_discogs_embeddings_model = load_effnet_discogs_embeddings_model()
    msd_musicnn_embeddings_model = load_msd_musicnn_embeddings_model()
    audioset_vggish_embeddings_model = load_audioset_vggish_embeddings_model()

    # 1 ded: DEAM dataset, Effnet-Discogs embeddings
    ded_av_model, ded_input_layer, ded_output_layer = load_av_model(
        "deam-effnet-discogs-1"
    )

    # 2 dmm: DEAM dataset, MusiCNN-MSD embeddings
    dmm_model, dmm_input_layer, dmm_output_layer = load_av_model("deam-musicnn-msd-1")

    # 3 dva: DEAM dataset, VGGish-AudioSet embeddings
    dva_model, dva_input_layer, dva_output_layer = load_av_model(
        "deam-vggish-audioset-1"
    )

    # 4 eed: EmoMusic dataset, Effnet-Discogs embeddings
    eed_av_model, eed_input_layer, eed_output_layer = load_av_model(
        "emomusic-effnet-discogs-1"
    )

    # 5 emm: EmoMusic dataset, MusiCNN-MSD embeddings
    emm_model, emm_input_layer, emm_output_layer = load_av_model(
        "emomusic-musicnn-msd-1"
    )

    # 6 eva: EmoMusic dataset, VGGish-AudioSet embeddings
    eva_model, eva_input_layer, eva_output_layer = load_av_model(
        "emomusic-vggish-audioset-1"
    )

    audio_paths = glob.glob(os.path.join(AUDIO_DIR, "*/*.mp3"))

    # Pick up where we left off
    predictions_df = pd.read_csv(ESSENTIA_PREDICTIONS_PATH)

    tracks = [os.path.splitext(os.path.basename(x))[0] for x in audio_paths]
    tracks_to_predict = set(tracks) - set(predictions_df.trackid)
    audio_paths_to_predict = [
        os.path.join(AUDIO_DIR, track_id[:2], f"{track_id}.mp3")
        for track_id in list(tracks_to_predict)
    ]
    print(f"{len(audio_paths_to_predict)} remaining files")

    for audio_path in audio_paths_to_predict:
        # Load audio at a 16 kHz sample rate for compatibility with embeddings models
        print(f"Loading {audio_path}...")
        audio = MonoLoader(filename=audio_path, sampleRate=16000)()

        # Compute embeddings
        t0 = time.time()
        print("  Computing effnet_discogs_embeddings...")
        effnet_discogs_embeddings = effnet_discogs_embeddings_model(audio)
        print("  Computing msd_musicnn_embeddings...")
        msd_musicnn_embeddings = msd_musicnn_embeddings_model(audio)
        print("  Computing audioset_vggish_embeddings...")
        audioset_vggish_embeddings = audioset_vggish_embeddings_model(audio)
        print(f"  Embeddings computed in {round(time.time() - t0, 2)} seconds")

        # Run inference with each model
        print("  Running inference with ded model...")
        ded_prediction = predict(
            effnet_discogs_embeddings,
            ded_av_model,
            ded_input_layer,
            ded_output_layer,
        )

        print("  Running inference with dmm model...")
        dmm_prediction = predict(
            msd_musicnn_embeddings,
            dmm_model,
            dmm_input_layer,
            dmm_output_layer,
        )

        print("  Running inference with dva model...")
        dva_prediction = predict(
            audioset_vggish_embeddings,
            dva_model,
            dva_input_layer,
            dva_output_layer,
        )

        print("  Running inference with eed model...")
        eed_prediction = predict(
            effnet_discogs_embeddings,
            eed_av_model,
            eed_input_layer,
            eed_output_layer,
        )

        print("  Running inference with emm model...")
        emm_prediction = predict(
            msd_musicnn_embeddings,
            emm_model,
            emm_input_layer,
            emm_output_layer,
        )

        print("  Running inference with eva model...")
        eva_prediction = predict(
            audioset_vggish_embeddings,
            eva_model,
            eva_input_layer,
            eva_output_layer,
        )

        pred_df = pd.DataFrame(
            [
                ded_prediction,
                dmm_prediction,
                dva_prediction,
                eed_prediction,
                emm_prediction,
                eva_prediction,
            ],
            columns=["valence", "arousal"],
        )

        pred_df["model"] = ["ded", "dmm", "dva", "eed", "emm", "eva"]
        pred_df["trackid"] = os.path.splitext(os.path.basename(audio_path))[0]

        predictions_df = pd.concat([predictions_df, pred_df])

        predictions_df.to_csv(ESSENTIA_PREDICTIONS_PATH, index=False)
        print(f"  Wrote predictions to {ESSENTIA_PREDICTIONS_PATH}")
