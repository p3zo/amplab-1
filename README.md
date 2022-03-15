## Usage

Install [Docker Compose](https://docs.docker.com/compose/install).

Download `essentia-models` from [Google Drive](https://drive.google.com/drive/folders/1LSiIrqFJfInbeJlWFMWdVH0A0uJfQNb0) and unzip it in the base directory of this repo.

#### Setup

Start the container and get a shell inside it

```bash
docker-compose up -d && docker-compose exec app bash
```

#### Scripts

##### To compare model predictions to ground truth

```bash
python scripts/analyze.py
```

The output will be written to `output/agreements-by-model.csv`.

##### To load essentia models and make predictions

```bash
python scripts/predict.py
```

The output will be written to `data/essentia-models-predictions.csv`.

##### To parse raw ground-truth & spotify annotations

From [Google Drive](https://drive.google.com/drive/folders/1LSiIrqFJfInbeJlWFMWdVH0A0uJfQNb0):

-   Download `audio_chunks/audio.001` and unzip it in the base directory of this repo
-   Download `annotations-spotifyapi_chunks/annotations-spotifyapi.001` and unzip it into `data`

```bash
python scripts/analyze.py --reparse_annotations
```

The output will be written to `data/ground-truth-annotations.csv` and `data/spotify-annotations.csv`.
