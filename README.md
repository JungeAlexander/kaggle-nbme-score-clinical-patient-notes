<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: NBME - Score Clinical Patient Notes

TODO.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `preprocess` | Convert the data to spaCy's binary format |
| `debug` | Debug the data |
| `train` | Train a custom NER model |
| `evaluate` | Evaluate the custom model and export metrics |
| `package` | Package the trained model so it can be installed |
| `serve` | Serve the models via a FastAPI REST API using the given host and port |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `preprocess` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`input/nbme-score-clinical-patient-notes/train_split.json.gz`](input/nbme-score-clinical-patient-notes/train_split.json.gz) | Local |  |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

# Legacy

# Kaggle challenge: NBME - Score Clinical Patient Notes

https://www.kaggle.com/c/nbme-score-clinical-patient-notes

## Initi

```
pyenv virtualenv 3.7.12 20220211-nbme-score-clinical-patient-notes
```

```
pyenv activate 20220211-nbme-score-clinical-patient-notes
```

```
poetry install --dev
```

```
python -m ipykernel install --user --name 20220211-nbme-score-clinical-patient-notes
```

## Running 

```
[1]  !pip install colabcode
```

```
[2] from colabcode import ColabCode
```

```
[3] ColabCode(port=10000, password=PASSWORD, authtoken=AUTHTOKEN) 
```
