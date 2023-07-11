import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import logging
import datetime
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
#  from dataset.data_pipelines import dataio_prep
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import wandb
from confusion_matrix_fig import create_cm_fig
from dataset.cl_pipeline import (
    prepare_task_csv_from_subset,
    prepare_task_csv_from_replay,
    prepare_concat_csv,
    class_balanced_dataio_prep,
)
from schedulers import SimSiamCosineScheduler
from cl_table_tools import compute_cl_statistics

import pdb


class ZSClassifier(sb.core.Brain):
    """
        Brain class for classifier with supervised training
    """
    def compute_forward(self, batch, stage):
        self.modules.clap.audio_encoder.eval()
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        # audio embeddings
        aud_shared, _ = self.modules.clap.audio_encoder(wavs)
        #  aud_shared = aud_shared / (torch.norm(aud_shared, dim=-1, keepdim=True) + 1e-8)
        return aud_shared, lens

    def prepare_aud_features(self, wavs, lens, stage):
        #  with torch.cuda.amp.autocast(enabled=False):
        feats = self.modules.compute_features(wavs)  # [B, T, D]
        if self.hparams.amp_to_db:
            Amp2db = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=80
            )  # try "magnitude" Vs "power"? db= 80, 50...
            feats = Amp2db(feats)
        # Normalization
        if self.hparams.normalize:
            feats = self.modules.mean_var_norm(feats, lens)

        return feats

    def prepare_txt_features(self, text):
        '''
            args:
                text: a list of strings

            output:
                txt_inp: dict, with input_ids, token_type_ids, and attention_mask
        '''
        txt_inp = self.hparams.txt_tokenizer(
            text,
            max_length=self.hparams.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return txt_inp

    def compute_similarity(self, emb1, emb2):
        logit_scale = self.modules.clap.logit_scale.exp()
        e1_norm = emb1 / (torch.norm(emb1, dim=-1, keepdim=True) + 1e-8)
        e2_norm = emb2 / (torch.norm(emb2, dim=-1, keepdim=True) + 1e-8)
        #  sim = self.modules.logit_scale(e1_norm @ e2_norm.T)
        sim = logit_scale * e1_norm @ e2_norm.T
        return sim

    def compute_objectives(self, predictions, batch, stage):
        aud_emb, lens = predictions
        predictions = self.compute_similarity(aud_emb, self.txt_emb).unsqueeze(1)  # [bs, 1, C]
        targets, _ = batch.class_string_encoded # [bs, 1]
        targets = F.one_hot(targets, predictions.shape[-1]).float()  # [bs, 1, C]
        loss = self.hparams.compute_cost(predictions, targets, lens)
        #  # Confusion matrices
        if stage != sb.Stage.TRAIN:
            y_true = targets.cpu().detach().numpy().argmax(-1).squeeze(-1)
            y_pred = predictions.cpu().detach().numpy().argmax(-1).squeeze(-1)
        if stage == sb.Stage.TEST:
            confusion_matix = confusion_matrix(
                y_true,
                y_pred,
                labels=sorted(self.hparams.label_encoder.ind2lab.keys()),
            )
            self.test_confusion_matrix += confusion_matix
        self.acc_metric.append(predictions, targets.argmax(-1), lens)
        return loss

    def register_txt_emb(self, text):
        self.modules.clap.caption_encoder.eval()
        with torch.no_grad():
            txt_encoding = self.prepare_txt_features(text).to(self.device)
            self.txt_emb = self.modules.clap.caption_encoder(txt_encoding)

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        self.acc_metric = self.hparams.acc_metric()
        # Confusion matrices
        if stage == sb.Stage.VALID:
            self.valid_confusion_matrix = np.zeros(
                shape=(self.hparams.n_classes, self.hparams.n_classes),
                dtype=int,
            )
        if stage == sb.Stage.TEST:
            self.test_confusion_matrix = np.zeros(
                shape=(self.hparams.n_classes, self.hparams.n_classes),
                dtype=int,
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        if stage == sb.Stage.TRAIN:
            self.train_stats = {
                'loss': stage_loss,
                'acc': self.acc_metric.summarize(),
            }
        elif stage == sb.Stage.VALID:
            valid_stats = {
                'loss': stage_loss,
                'acc': self.acc_metric.summarize(),
            }
        else:
            self.test_stats = {
                'loss': stage_loss,
                'acc': self.acc_metric.summarize(),
            }

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_scheduler(epoch)
            if not hasattr(self.hparams.lr_scheduler, "on_batch_end"):
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            # The train_logger writes a summary to stdout and to the logfile.
            # wandb logger
            if self.hparams.use_wandb:
                cm_fig = create_cm_fig(
                    self.valid_confusion_matrix,
                    display_labels=list(
                        self.hparams.label_encoder.ind2lab.values()
                    ),
                )
                valid_stats.update({
                    'confusion': wandb.Image(cm_fig),
                })
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "datapoints_seen": self.hparams.datapoint_counter.current,
                },
                train_stats=self.train_stats,
                valid_stats=valid_stats,
            )
            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta={'acc': valid_stats['acc']}, max_keys=["acc"]
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            # Per class accuracy from Test confusion matrix
            per_class_acc_arr = np.diag(self.test_confusion_matrix) / np.sum(
                self.test_confusion_matrix, axis=1
            )
            per_class_acc_arr_str = "\n" + "\n".join(
                "{:}: {:.3f}".format(class_id, class_acc)
                for class_id, class_acc in enumerate(per_class_acc_arr)
            )
            # wandb logger
            if self.hparams.use_wandb:
                cm_fig = create_cm_fig(
                    self.test_confusion_matrix,
                    display_labels=list(
                        self.hparams.label_encoder.ind2lab.values()
                    ),
                )
                self.test_stats.update({
                    'confusion': wandb.Image(cm_fig),
                })
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "Epoch loaded": self.hparams.epoch_counter.current,
                    },
                    test_stats=self.test_stats
                )
            else:
                self.hparams.train_logger.log_stats(
                    {
                        "Epoch loaded": self.hparams.epoch_counter.current,
                        "\n Per Class Accuracy": per_class_acc_arr_str,
                        "\n Confusion Matrix": "\n{:}\n".format(
                            self.test_confusion_matrix
                        ),
                    },
                    test_stats=self.test_stats,
                )
            self.test_stats.update({'cmat': self.test_confusion_matrix})


def dataio_prep(hparams, csv_path, label_encoder):
    "Creates the datasets and their data processing pipelines."

    config_sample_rate = hparams["sample_rate"]
    # TODO  use SB implementation but need to make sure it give the same results as PyTorch
    # resampler = sb.processing.speech_augmentation.Resample(orig_freq=latest_file_sr, new_freq=config_sample_rate)
    hparams["resampler"] = torchaudio.transforms.Resample(
        new_freq=config_sample_rate
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        sig, read_sr = torchaudio.load(wav_path)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed compared to last file
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)
        if hparams['duration'] * config_sample_rate >= sig.shape[0]:
            repeat_factor = int(np.ceil((hparams['duration']*config_sample_rate)/sig.shape[0]))
            sig = sig.repeat(repeat_factor)
            sig = sig[0:int(hparams['duration']*config_sample_rate)]
        else:
            start_index = random.randrange(
                sig.shape[0] - int(hparams['duration'] * config_sample_rate)
            )
            sig = sig[start_index:start_index + int(hparams['duration']*config_sample_rate)]
        #  # scaling
        #  max_amp = torch.abs(sig).max().item()
        #  #  assert max_amp > 0
        #  scaling = 1 / max_amp * 0.9
        #  sig = scaling * sig

        return sig

    # 3. Define label pipeline:
    @sb.utils.data_pipeline.takes("class_name")
    @sb.utils.data_pipeline.provides("class_name", "class_string_encoded")
    def label_pipeline(class_name):
        yield class_name
        class_string_encoded = label_encoder.encode_label_torch(class_name)
        yield class_string_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    ds = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_path,
        dynamic_items=[audio_pipeline, label_pipeline],
        output_keys=["id", "sig", "class_string_encoded"]
    )

    return ds


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # setting up experiment stamp
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d+%H-%M-%S')
    if run_opts['debug']:
        time_stamp = 'debug_' + time_stamp
    stamp_override = 'time_stamp: {}'.format(time_stamp)
    overrides = stamp_override + '\n' + overrides if len(overrides) > 0 else stamp_override

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger_fn']()

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.load_or_create(hparams['label_encoder_path'])
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    # set new checkpointer
    hparams['checkpointer'] = sb.utils.checkpoints.Checkpointer(
        hparams['save_folder'],
        recoverables=hparams['recoverables']
    )
    print('==> Resetting scheduler, counter and linear classifier at {}'.format(hparams['checkpointer'].checkpoints_dir))

    # load weights from pretrained audio embedder
    if hparams['clap_ckpt_path'] is not None:
         model_state_dict = torch.load(hparams['clap_ckpt_path'], map_location=torch.device('cpu'))['model']
         hparams['clap'].load_state_dict(model_state_dict)
         print('==> Loading pretrained CLAP checkpoint from {}'.format(hparams['clap_ckpt_path']))
    #  print("==> Recovering embedder checkpointer at {}".format(ssl_checkpointer.checkpoints_dir))
    if hparams['ssl_checkpointer'] is not None:
        if hparams['use_maxacc_ep']:
            hparams['ssl_checkpointer'].recover_if_possible(max_key='acc')
        elif hparams['use_minloss_ep']:
            hparams['ssl_checkpointer'].recover_if_possible(min_key='loss')
        else:
            #  hparams['ssl_checkpointer'].recover_if_possible()
            chosen_ckpts = hparams['ssl_checkpointer'].find_checkpoints()
            chosen_ckpt = chosen_ckpts[0]
            hparams['ssl_checkpointer'].load_checkpoint(chosen_ckpt)

    # data
    run_on_main(
        hparams['prepare_split_csv_fn']
    )
    #  if hparams['linclf_train_type'] == 'full':
    #      train_data = dataio_prep(
    #          hparams,
    #          os.path.join(hparams['save_folder'], 'train_raw.csv'),
    #          label_encoder,
    #      )
    #      valid_data = dataio_prep(
    #          hparams,
    #          os.path.join(hparams['save_folder'], 'valid_raw.csv'),
    #          label_encoder,
    #      )
    #  elif hparams['linclf_train_type'] == 'subset':
    #      raise NotImplementedError("subset linear probing not implemented tet")

    test_data = dataio_prep(
        hparams,
        os.path.join(hparams['save_folder'], 'test_raw.csv'),
        label_encoder,
    )

    brain = ZSClassifier(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    text_input = [hparams['text_prompt'] + e for e in class_labels]
    brain.register_txt_emb(text_input)

    #  brain.fit(
    #      epoch_counter=brain.hparams.epoch_counter,
    #      train_set=train_data,
    #      valid_set=valid_data,
    #      train_loader_kwargs=hparams["train_dataloader_opts"],
    #      valid_loader_kwargs=hparams["valid_dataloader_opts"],
    #  )
    #
    brain.evaluate(
        test_set=test_data,
        max_key="acc",
        progressbar=True,
        test_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    torch.save(
        brain.test_stats,
        os.path.join(
            hparams['checkpointer'].checkpoints_dir,
            'test_stats.pt'
        )
    )
