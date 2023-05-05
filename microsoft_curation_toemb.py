import os
import sys
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


class MSCLAPCurator(sb.core.Brain):
    """
        Brain class for classifier with supervised training
    """
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        # audio embeddings
        aud_shared, _ = self.modules.clap.audio_encoder(wavs)
        return aud_shared, lens

    def compute_audio_forward(self, wavs, stage):
        # audio embeddings
        aud_shared, _ = self.modules.clap.audio_encoder(wavs)
        return aud_shared

    def compute_text_forward(self, txt_encoding, stage):
        txt_shared = self.modules.clap.caption_encoder(txt_encoding)
        return txt_shared

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
        e1_norm = emb1 / (torch.norm(emb1, dim=-1, keepdim=True) + 1e-8)
        e2_norm = emb2 / (torch.norm(emb2, dim=-1, keepdim=True) + 1e-8)
        #  sim = self.modules.logit_scale(e1_norm @ e2_norm.T)
        sim = e1_norm @ e2_norm.T
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
        with torch.no_grad():
            txt_encoding = self.prepare_txt_features(text).to(self.device)
            embedding_output = self.modules.txt_emb_model(
                input_ids=txt_encoding['input_ids'],
                token_type_ids=txt_encoding['token_type_ids'],
            )
            bert_outputs = self.modules.mm_bert_model(
                embedding_output=embedding_output,
                attention_mask=txt_encoding['attention_mask'],
            )
            sequence_output = bert_outputs[0]
            self.txt_emb = self.modules.txt_cls_projector(sequence_output[:, 0])

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
            test_stats = {
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
                test_stats.update({
                    'confusion': wandb.Image(cm_fig),
                })
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "Epoch loaded": self.hparams.epoch_counter.current,
                    },
                    test_stats=test_stats
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
                    test_stats=test_stats,
                )

    def compute_audio_embeddings_in_batch(
        self, audio_loader,
    ):
        res = []
        self.modules.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(audio_loader)):
                wavs = batch['audio'].to(self.device)
                audio_embeddings = self.compute_audio_forward(wavs, sb.Stage.TEST)
                audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
                res.append(audio_embeddings.detach().half().cpu().numpy())
                #  if i == 10:
                #      break
        res = np.concatenate(res, axis=0)
        np.save(self.hparams.aud_svpth, res)

    def compute_text_embeddings_in_batch(
        self, text_loader,
    ):
        text_res = []
        self.modules.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(text_loader)):
                text = batch['text']
                txt_encoding = self.prepare_txt_features(text).to(self.device)
                text_embeddings = self.compute_text_forward(txt_encoding, sb.Stage.TEST)
                text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)
                text_res.append(text_embeddings.detach().half().cpu().numpy())
                #  if i == 10:
                #      break
        text_res = np.concatenate(text_res, axis=0)
        np.save(self.hparams.txt_svpth, text_res)


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


    if hparams['clap_ckpt_path'] is not None:
         model_state_dict = torch.load(hparams['clap_ckpt_path'], map_location=torch.device('cpu'))['model']
         hparams['clap'].load_state_dict(model_state_dict)
         print('==> Loading pretrained CLAP checkpoint from {}'.format(hparams['clap_ckpt_path']))

    # set new checkpointer
    hparams['checkpointer'] = sb.utils.checkpoints.Checkpointer(
        hparams['save_folder'],
        recoverables=hparams['recoverables']
    )
    print('==> Resetting checkpointer at {}'.format(hparams['checkpointer'].checkpoints_dir))

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

    brain = MSCLAPCurator(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # data
    if hparams['audio_dataset'] is not None:
        audio_loader = torch.utils.data.DataLoader(
            hparams['audio_dataset'], **hparams['audio_dataloader_opts'],
        )
        brain.compute_audio_embeddings_in_batch(audio_loader)
    if hparams['text_dataset'] is not None:
        text_loader = torch.utils.data.DataLoader(
            hparams['text_dataset'], **hparams['text_dataloader_opts'],
        )
        brain.compute_text_embeddings_in_batch(text_loader)
