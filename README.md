# Codebase: Unsupervised Improvement for Audio-Text Cross-Modal Representations

This is the codebase for our paper [**Unsupervised Improvement for Audio-Text Cross-Modal Representations**](https://arxiv.org/abs/2305.01864). This repository contains code for training the teacher and student CLAP models as well as curating the Improvement-Set from unpaired audio and text data. It is implemented primarily with [SpeechBrain](https://speechbrain.github.io/).


## Environments
The python scripts are based upon Python 3.9. You may install the dependencies with
```bash
pip install -r requirements.txt
```

## Running the code

### CLAP Training
We provide the script to train the CLAP model as an alternative reproduction of the CLAP implemented by Microsoft. The same script can be used to train the student models, given proper specification of the arguments.


```bash
python microsoft_clap_train.py hparams/insper/microsoft_clap_train.yaml \
--data_annot_folder=$DATA_ANNOT_FOLDER \
--train_csv_path=$TRAIN_CSV_PATH \
--curation_csv=$CURATION_CSV \
--output_base=$OUTPUT_BASE \
--pann_ckpt_path=$PANN_CKPT_PATH \
--clap_ckpt_path=$CLAP_CKPT_PATH \
--ssl_checkpoints_dir=$SSL_CHECKPOINTS_DIR \
--ssl_checkpointer=$SSL_CHECKPOINTER \
--resume_interrupt=False \
--prev_ckpt_dir=$PREV_CKPT_DIR \
--prev_checkpointer=$PREV_CHECKPOINTER \
--curation_prob=0 \
--subset_pct=1 \
--number_of_epochs=20 \
--batch_size=128 \
--base_lr=0.0001 \
--final_lr=0.00000001 \
--experiment_name=$EXPERIMENT_NAME \
--auto_mix_prec \
--data_parallel_backend
```

Notice that if using the default values for each flag, you may simply **delete** this flag. This is not equivalent to **leaving the flag empty**, which sets a `null` value to the flag. Details of the arguments are as follows,
- `data_annot_folder`: this should be the root data directory containing the audio files and their associated caption. Used in `dataset/prepare_insper` to generate the training csv path
- `train_csv_path`: path to the csv file containing annotated audio-caption pairs; ignored for teacher training; used for DS, ADS training to specify the replay dataset
- `data_curation_folder`: path to the csv file containing curated audio-caption pairs; used for student training
- `output_base`: root of the output directory, which contains all the checkpoints and training artifacts
- `pann_ckpt_path`: path to the [CNN14 checkpoint provided by the PANN authors](https://zenodo.org/record/3987831); leave empty unless for teacher training from scratch
- `clap_ckpt_path`: path to the [CLAP weights provided by Microsoft](https://zenodo.org/record/7312125#.Y22vecvMIQ9); leave empty unless intending to load the Microsoft's pretrained weights for subsequent training
- `ssl_checkpoints_dir`: path to the teacher checkpoint directory (path should end with a `save` directory); leave empty otherwise
- `ssl_checkpointer`: delete this flag when `ssl_checkpoints_dir` is non-empty; otherwise, set this flag to `null` or empty
- `resume_interrupt`: whether to continue a previously interrupted training or to start a new training
- `prev_ckpt_dir`: path to the speechbrain checkpointer if training is interrupted (resumed training would overwrite this directory); leave it empty if `resume_interrupt` is set to `False`
- `prev_checkpointer`: delete this flag when `prev_ckpt_dir` is non-empty; set this flag to `null` or empty otherwise
- `curation_prob`: probability of loading the curation data for student training (vs replay); set to 0 when no curation data is provided
- `subset_pct`: percentage of the full annotated dataset to be used for training; set to 1 when using the full dataset
- `auto_mix_prec`: applies half precision training; to disable, delete this flag
- `data_parallel_backend`: multi-gpu training; to disable, delete this flag


For student training, we include a variant of CLAP using soft labels that are generated automatically on-the-fly:

```bash
python softclap_train.py hparams/insper/softclap_train.yaml \
--data_annot_folder=$DATA_ANNOT_FOLDER \
--train_csv_path=$TRAIN_CSV_PATH \
--curation_csv=$CURATION_CSV \
--data_zs_folder=$DATA_ZS_FOLDER \
--output_base=$OUTPUT_BASE \
--pann_ckpt_path=$PANN_CKPT_PATH \
--clap_ckpt_path=$CLAP_CKPT_PATH \
--ssl_checkpoints_dir=$SSL_CHECKPOINTS_DIR \
--ssl_checkpointer=$SSL_CHECKPOINTER \
--teacher_checkpoints_dir=$TEACHER_CHECKPOINTS_DIR \
--teacher_checkpointer=$TEACHER_CHECKPOINTER \
--resume_interrupt=False \
--prev_ckpt_dir=$PREV_CKPT_DIR \
--prev_checkpointer=$PREV_CHECKPOINTER \
--curation_prob=0 \
--subset_pct=1 \
--number_of_epochs=20 \
--batch_size=128 \
--base_lr=0.0001 \
--final_lr=0.00000001 \
--softclap_beta=0.3 \
--experiment_name=$EXPERIMENT_NAME \
--auto_mix_prec \
--data_parallel_backend
```
And the additional arguments (compared with regular CLAP training) are explained below:

- `data_zs_folder`: directory of the downstream dataset (i.e, ESC-50, UrbanSound8K, TUT17)
- `teacher_checkpoints_dir`: speechbrain checkpoint of the teacher model; if using third-party checkpoints, leave this flag empty
- `teacher_checkpointer`: delete this flag when the `teacher_checkpoints_dir` is non-empty; if using third-party checkpoints, set this flag to empty (null)
- `softclap_beta`: the coefficients adjusting the weights between soft and hard labels

### Zero-shot Evaluation
To perform zero-shot classificatio on downstream tasks, run

```bash
python microsoft_zeroshot.py hparams/$DATASET/microsoft_zeroshot.yaml \
--data_folder=$DATA_FOLDER \
--clap_ckpt_path=$CLAP_CKPT_PATH \
--ssl_checkpoints_dir=$SSL_CHECKPOINTS_DIR \
--output_base=$OUTPUT_BASE \
--experiment_name=$EXPERIMENT_NAME
```
where `$DATASET` is one of `esc, tut, us8k`, and `data_folder` should be the root of the corresponding downstream dataset.

### Unsupervised Data Curation

We also provide scripts to perform data curation with unpaired audio dataset (AudioSet) and text corpora (the set of audio captions used for teacher training). To perform the domain-unspecific (DU) generation, run the following commands to generate a csv file containing curated audio-text pairs (which is used for the `$CURATION_CSV` argument for the student training):


```bash
# this command generate audio and text embeddings (npy)
python microsoft_curation_toemb.py hparams/curation/microsoft_audioset_insper_toemb.yaml \
--output_folder=$OUTPUT_DIR \
--clap_ckpt_path=$CLAP_CKPT_PATH \
--ssl_checkpoints_dir=$SSL_CHECKPOINTS_DIR \
--audio_pathfile=$AUDIO_PATHFILE \
--text_pathfile=$TEXT_PATHFILE \

# this command generate audio-caption pairs from the embeddings
# computed in previous step based on a similarity threshold
python retrieve_from_emb_thresh.py hparams/curation/audioset_insper_duv2_retr.yaml \
--emb_dir=$EMB_DIR \
--audio_bs=$AUDIO_BS \
--sim_threshold=$SIM_THRESHOLD
 ```

- `audio_pathfile`: a text file containing a list of absolute paths of the audio recordings to be used for curation
- `text_pathfile`: a csv file that contains the teacher training set of audio and caption pairs (if using our pipeline for training, an example file is the `$SSL_CHECKPOINTS_DIR/train_raw.csv`)
- `emb_dir`: directory to look for the embedding files for audio and text; it should be equivalent to `$OUTPUT_DIR/save` for the embedding generation command
- `sim_threshold`: similarity threshold for filtering domain-specific captions


In addition, we also provide scripts to perform domain-specific (DS) curation and the augmented domain-specific curation (ADS). For DS curation, the following commands generates a csv file containing audio-text pairs from the original training set that are relevant to the downstream task (also dumps the corresponding text embeddings):


```bash
# this command compares the similarity between the (prompted) labels of the
# downstream dataset and the training captions, and keeps the captions that are
# relevant to the downstream task

python retrieve_ds_t2t.py hparams/curation/$DATASET_insper_t2t_retr.yaml \
--target_csv_path=$TARGET_CSV_PATH \
--clap_ckpt_path=$CLAP_CKPT_PATH \
--ssl_checkpoints_dir=$SSL_CHECKPOINTS_DIR \
--emb_dir=$EMB_DIR \
--sim_threshold=$SIM_THRESHOLD
```
where `$DATASET` is one of `esc, tut, us8k`, `target_csv_path` is equivalent to the `text_pathfile`, and `emb_dir` is the output directory.

To perform ADS, after generating the DS dataset, run the following to generate the augmented set in the form of csv:

```bash
# this command takes a list of audio files and a csv of the subset of training
# audio-caption pairs and performs curation on these two sets
python retrieve_from_emb_thresh.py hparams/curation/$DATASET_insper_audioset_dudsv3_retr.yaml \
--emb_dir=$EMB_DIR \
--aud_ds_path=$AUD_DS_PATH \
--sim_threshold=$SIM_THRESHOLD
```
where `aud_ds_path` is equivalent to `audio_pathfile` for DU curation. Make sure that `emb_dir` contains the list of audio embeddings (from DU generation) and text embeddings `.npy` files before running the command.
