import os
import sys
import shutil
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import logging
import datetime
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from transformers import BatchEncoding
from tqdm import tqdm
import wandb
from schedulers import SimSiamCosineScheduler
import pdb


class CLAP(sb.core.Brain):
    """
        Brain class for multimodal CLAP framework
    """
    def compute_forward(self, batch, stage):
        return_dic = {}
        batch = batch.to(self.device)
        wavs, lens = batch.sig1
        # for validation with zero-shot loader, obtain text embeddings from label set
        if stage == sb.Stage.VALID and self.hparams.use_zs_loader:
            input_ids = self.zs_label_encoding.input_ids
            token_type_ids = self.zs_label_encoding.token_type_ids
            attention_mask = self.zs_label_encoding.attention_mask
        else:
            input_ids = batch.input_ids.data
            token_type_ids = batch.token_type_ids.data
            attention_mask = batch.attention_mask.data
        if stage == sb.Stage.TRAIN:
            self.modules.teacher_clap.eval()
            with torch.no_grad():
                txt_teacher, aud_teacher = self.modules.teacher_clap(wavs, input_ids, token_type_ids, attention_mask, compute_sim=False)
                return_dic.update({
                    'txt_teacher': txt_teacher,
                    'aud_teacher': aud_teacher,
                })
        if self.hparams.global_contrastive:
            txt_shared, aud_shared = self.modules.clap(wavs, input_ids, token_type_ids, attention_mask, compute_sim=False)
            return_dic.update({
                'aud_emb': aud_shared,
                'txt_emb': txt_shared,
                'lens': lens,
            })
        else:
            txt_shared, aud_shared, similarity = self.modules.clap(wavs, input_ids, token_type_ids, attention_mask, compute_sim=True)
            return_dic.update({
                'aud_emb': aud_shared,
                'txt_emb': txt_shared,
                'similarity': similarity,
                'lens': lens,
            })
        return return_dic

    def prepare_aud_features(self, wavs, lens, stage):
        #  with torch.cuda.amp.autocast(enabled=False):
        feats = self.modules.compute_features(wavs)  # [B, T, D]
        feats = self.hparams.spec_domain_aug(feats, lens)
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
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        sim = logit_scale * emb1 @ emb2.T
        # clamp the scaled logit to avoid instability
        sim = torch.clamp(sim, max=100)
        return sim

    def compute_similarity_with_scale(self, emb1, emb2, logit_scale):
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
        sim = logit_scale * emb1 @ emb2.T
        # clamp the scaled logit to avoid instability
        sim = torch.clamp(sim, max=100)
        return sim

    def compute_objectives(self, predictions, batch, stage):
        #  z1, z2, lens = predictions
        z1 = predictions['aud_emb']
        z2 = predictions['txt_emb']
        lens = predictions['lens']
        if stage == sb.Stage.VALID and self.hparams.use_zs_loader:
            logit_scale = self.modules.clap.module.logit_scale.exp() \
                    if self.data_parallel_backend \
                    else self.modules.clap.logit_scale.exp()
            logits = self.compute_similarity_with_scale(z1, z2, logit_scale)  # [bs, C]
            targets, _ = batch.class_string_encoded # [bs, 1]
            targets = targets.squeeze(-1)  # [bs,]
            loss = self.hparams.compute_zs_cost(logits, targets)
            #  y_true = targets.cpu().detach().numpy()
            #  y_pred = logits.cpu().detach().numpy().argmax(-1).squeeze(-1)
            self.acc_metric.append(logits.unsqueeze(1), targets.unsqueeze(-1), lens)
        else:
            z1_teacher = predictions.get('aud_teacher', None)
            z2_teacher = predictions.get('txt_teacher', None)
            if self.hparams.global_contrastive:
                logit_scale = self.modules.clap.module.logit_scale.exp() \
                        if self.data_parallel_backend \
                        else self.modules.clap.logit_scale.exp()
                similarity = self.compute_similarity_with_scale(z1, z2, logit_scale)
                loss = self.hparams.compute_cross_sim_cost(similarity, z1_teacher, z2_teacher)
            else:
                similarity = predictions['similarity']
                n_gpus = similarity.shape[0] // similarity.shape[-1]
                loss = 0
                for i in range(n_gpus):
                    sim = similarity[i*similarity.shape[-1]:(i+1)*similarity.shape[-1]]
                    if z1_tc is not None and z2_tc is not None:
                        z1_tc = z1_teacher[i*similarity.shape[-1]*(i+1)*similarity.shape[-1]]
                        z2_tc = z2_teacher[i*similarity.shape[-1]*(i+1)*similarity.shape[-1]]
                    else:
                        z1_tc = None
                        z2_tc = None
                    loss += self.hparams.compute_cross_sim_cost(sim, z1_tc, z2_tc)

        return loss

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.
        The default implementation depends on a few methods being defined
        with a particular behavior:
        * ``compute_forward()``
        * ``compute_objectives()``
        Also depends on having optimizers passed at initialization.
        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        Returns
        -------
        detached loss
        """
        # adding a second dataloader
        curation_prob = torch.rand(1).item()
        if curation_prob < self.hparams.curation_prob:
            try:
                batch = next(self.hparams.train_curation_loader)
            except StopIteration:
                del self.hparams.train_curation_loader
                new_loader = sb.dataio.dataloader.make_dataloader(
                    self.hparams.train_curation_data, **(self.hparams.train_dataloader_opts),
                )
                self.hparams.train_curation_loader = iter(new_loader)
                batch = next(self.hparams.train_curation_loader)
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.check_gradients(loss):
                self.scaler.step(self.optimizer)
                # wandb logger: update datapoints info
                self.hparams.datapoint_counter.update(batch.sig1.data.shape[0])
            self.scaler.update()
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            loss.backward()
            if self.check_gradients(loss):
                self.optimizer.step()
                # wandb logger: update datapoints info
                self.hparams.datapoint_counter.update(batch.sig1.data.shape[0])
            self.optimizer.zero_grad()

        # scheduler update
        if hasattr(self.hparams.lr_scheduler, "on_batch_end"):
            self.hparams.lr_scheduler.on_batch_end(self.optimizer)

        # wandb logger
        if self.hparams.use_wandb:
            #  self.train_loss_buffer.append(loss.item())
            loss_dict = {}
            if len(loss_dict) > 1:
                loss_dict['loss'] = loss
            for loss_nm, loss_val in loss_dict.items():
                if loss_nm not in self.train_loss_buffer:
                    self.train_loss_buffer[loss_nm] = []
                self.train_loss_buffer[loss_nm].append(loss_val.item())
            if self.step % self.hparams.train_log_frequency == 0 and self.step > 1:
                self.hparams.train_logger.log_stats(
                    stats_meta={"datapoints_seen": self.hparams.datapoint_counter.current},
                    #  train_stats={'buffer-loss': np.mean(self.train_loss_buffer)},
                    train_stats = {'buffer-{}'.format(loss_nm): np.mean(loss_list) for loss_nm, loss_list in self.train_loss_buffer.items()},
                )
                #  self.train_loss_buffer = []
                self.train_loss_buffer = {}

        return loss.detach().cpu()

    def check_gradients(self, loss):
        """Check if gradients are finite and not too large.

        Automatically clips large gradients.

        Arguments
        ---------
        loss : tensor
            The loss tensor after ``backward()`` has been called but
            before the optimizers ``step()``.

        Returns
        -------
        bool
            Whether or not the optimizer step should be carried out.
        """
        if not torch.isfinite(loss):
            self.nonfinite_count += 1

            # Print helpful debug info
            logger.warn(f"Loss is {loss}.")
            for p in self.modules.parameters():
                if not torch.isfinite(p).all():
                    logger.warn("Parameter is not finite: " + str(p))

            # Check if patience is exhausted
            if self.nonfinite_count > self.nonfinite_patience:
                raise ValueError(
                    "Loss is not finite and patience is exhausted. "
                    "To debug, wrap `fit()` with "
                    "autograd's `detect_anomaly()`, e.g.\n\nwith "
                    "torch.autograd.detect_anomaly():\n\tbrain.fit(...)"
                )
            else:
                logger.warn("Patience not yet exhausted, ignoring this batch.")
                return False

        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(
            (p for p in self.modules.parameters()), self.max_grad_norm
        )

        return True

    def on_fit_start(self):
        super().on_fit_start()
        # wandb logger
        #  self.train_loss_buffer = []
        self.train_loss_buffer = {}
        self.train_stats = {}

    def init_optimizers(self):
        if self.opt_class is not None:
            predictor_prefix = (
                'module.predictor',
                'predictor',
                'module.aud_emb_predictor',
                'aud_emb_predictor',
            )
            optim_params = [{
                'name': 'base',
                'params': [param for name, param in self.modules.named_parameters() if not name.startswith(predictor_prefix)],
                'fix_lr': False,
            }, {
                'name': 'predictor',
                'params': [param for name, param in self.modules.named_parameters() if name.startswith(predictor_prefix)],
                'fix_lr': True,
            }]
            self.optimizer = self.opt_class(optim_params)

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

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
        pass
        if hparams['use_zs_loader'] and stage == sb.Stage.VALID:
            self.acc_metric = self.hparams.acc_metric()
            self.modules.eval()
            with torch.no_grad():
                class_labels = list(self.hparams.zs_label_encoder.ind2lab.values())
                text_input = [self.hparams.zs_text_prompt + e for e in class_labels]
                self.zs_label_encoding = self.prepare_txt_features(text_input).to(self.device)


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
            }
        elif stage == sb.Stage.VALID:
            valid_stats = {
                'loss': stage_loss,
            }
            if self.hparams.use_zs_loader:
                valid_stats.update({
                    'acc': self.acc_metric.summarize(),
                })

        # Perform end-of-iteration things, like annealing, logging, etc.
        if self.hparams.valid_pct > 0 or self.hparams.use_zs_loader:
            log_stage = sb.Stage.VALID
            if stage == sb.Stage.VALID:
                log_stats = valid_stats
        else:
            log_stage = sb.Stage.TRAIN
            log_stats = self.train_stats
        if stage == log_stage:
            if isinstance(self.hparams.lr_scheduler, SimSiamCosineScheduler):
                old_lr, new_lr = self.hparams.lr_scheduler(epoch)
            elif isinstance(self.hparams.lr_scheduler, sb.nnet.schedulers.ReduceLROnPlateau):
                old_lr, new_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "datapoints_seen": self.hparams.datapoint_counter.current,
                },
                train_stats=self.train_stats,
                valid_stats=valid_stats if log_stage==sb.Stage.VALID else None,
            )
            # Save the current checkpoint and delete previous checkpoints,
            if epoch % self.hparams.num_epoch_save == 0:
                if self.hparams.use_zs_loader:
                    self.checkpointer.save_and_keep_only(
                        meta={'acc': log_stats['acc']}, num_to_keep=self.hparams.num_ckpt_keep, max_keys=['acc']
                    )
                else:
                    self.checkpointer.save_and_keep_only(
                        meta={'loss': log_stats['loss']}, num_to_keep=self.hparams.num_ckpt_keep, min_keys=['loss']
                    )


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

    # set up teacher model
    if hparams['pann_ckpt_path'] is not None:
        aud_emb_ckpt = torch.load(hparams['pann_ckpt_path'])
        hparams['teacher_clap'].audio_encoder.base.load_state_dict(aud_emb_ckpt['model'])
        print('==> Loading pretrained teacher checkpoint from {}'.format(hparams['pann_ckpt_path']))
    if hparams['clap_ckpt_path'] is not None:
        model_state_dict = torch.load(hparams['clap_ckpt_path'], map_location=torch.device('cpu'))['model']
        hparams['teacher_clap'].load_state_dict(model_state_dict)
        print('==> Loading pretrained teacher checkpoint from {}'.format(hparams['clap_ckpt_path']))
    if hparams['teacher_checkpointer'] is not None:
        if hparams['use_maxacc_ep']:
            hparams['teacher_checkpointer'].recover_if_possible(max_key='acc')
        elif hparams['use_minloss_ep']:
            hparams['teacher_checkpointer'].recover_if_possible(min_key='loss')
        else:
            chosen_ckpts = hparams['teacher_checkpointer'].find_checkpoints()
            chosen_ckpt = chosen_ckpts[0]
            hparams['teacher_checkpointer'].load_checkpoint(chosen_ckpt)
        print('==> Loading pretrained teacher checkpoint from {}'.format(hparams['teacher_checkpointer'].checkpoints_dir))

    # initialize student model and training configs
    if not hparams['resume_interrupt']:
        # reset epoch counter and lr scheduler
        # weights should be restored already in on_evaluate_start()
        hparams['recoverables']['lr_scheduler'] = \
                hparams['lr_scheduler'] = hparams['lr_scheduler_fn']()
        hparams['recoverables']['epoch_counter'] = \
                hparams['epoch_counter'] = hparams['epoch_counter_fn']()
        # set new checkpointer
        hparams['checkpointer'] = sb.utils.checkpoints.Checkpointer(
            hparams['save_folder'],
            recoverables=hparams['recoverables']
        )
        print('==> Resetting scheduler and counter at {}'.format(hparams['checkpointer'].checkpoints_dir))
        # load weights from pretrained audio embedder
        if hparams['pann_ckpt_path'] is not None:
            aud_emb_ckpt = torch.load(hparams['pann_ckpt_path'])
            hparams['clap'].audio_encoder.base.load_state_dict(aud_emb_ckpt['model'])
            print('==> Loading pretrained PANN checkpoint from {}'.format(hparams['pann_ckpt_path']))
        if hparams['clap_ckpt_path'] is not None:
            model_state_dict = torch.load(hparams['clap_ckpt_path'], map_location=torch.device('cpu'))['model']
            hparams['clap'].load_state_dict(model_state_dict)
            print('==> Loading pretrained CLAP checkpoint from {}'.format(hparams['clap_ckpt_path']))
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
            print('==> Loading pretrained CLAP checkpoint from {}'.format(hparams['ssl_checkpointer'].checkpoints_dir))
    else:
        # reload everything from the interrupted checkpoint
        # set the checkpointer here, and on_fit_start() loads the content
        assert isinstance(hparams['prev_checkpointer'], sb.utils.checkpoints.Checkpointer)
        # initialize epoch counter and lr scheduler for restore
        hparams['recoverables']['lr_scheduler'] = \
                hparams['lr_scheduler'] = hparams['lr_scheduler_fn']()
        hparams['recoverables']['epoch_counter'] = \
                hparams['epoch_counter'] = hparams['epoch_counter_fn']()
        hparams['checkpointer'] = hparams['prev_checkpointer']
        hparams['checkpointer'].add_recoverables(hparams['recoverables'])
        hparams['save_folder'] = hparams['checkpointer'].checkpoints_dir
        print("==> Resuming from interrupted checkpointer at {}".format(hparams['checkpointer'].checkpoints_dir))


    # frozen encoder setup
    if hparams['freeze_txt_emb']:
        for p in hparams['clap'].caption_encoder.base.parameters():
            p.requires_grad = False
    if hparams['freeze_aud_emb']:
        for p in hparams['clap'].audio_encoder.base.parameters():
            p.requires_grad = False
    for p in hparams['teacher_clap'].parameters():
        p.requires_grad = False

    # labeled data
    if hparams['train_csv_path'] is not None:
        shutil.copy(hparams['train_csv_path'], os.path.join(hparams['save_folder'], 'train_raw.csv'))
    if not os.path.exists(os.path.join(hparams['save_folder'], 'train_raw.csv')):
        run_on_main(
            hparams['prepare_annot_csv_fn']
        )
    train_data = hparams['dataio_annot_prep_fn'](
        hparams,
        os.path.join(hparams['save_folder'], 'train_raw.csv'),
    )
    if hparams['valid_pct'] > 0:
        valid_data = hparams['dataio_annot_prep_fn'](
            hparams,
            os.path.join(hparams['save_folder'], 'valid_raw.csv'),
            is_train=False,
        )
    else:
        valid_data = None

    # optional zs data
    if hparams['use_zs_loader']:
        label_encoder = sb.dataio.encoder.CategoricalEncoder()
        label_encoder.load_or_create(hparams['zs_label_encoder_path'])
        hparams["zs_label_encoder"] = label_encoder
        run_on_main(
            hparams['prepare_zs_csv_fn']
        )
        valid_data = hparams['dataio_zs_prep_fn'](
            hparams,
            os.path.join(hparams['save_folder'], 'valid_raw.csv'),
            label_encoder,
        )

    # curation data
    if hparams['curation_prob'] > 0:
        hparams['train_curation_data'] = train_curation_data = hparams['dataio_curation_prep_fn'](
            hparams,
            hparams['curation_csv'],
        )
        train_curation_loader = sb.dataio.dataloader.make_dataloader(
            train_curation_data, **hparams['train_dataloader_opts'],
        )
        hparams['train_curation_loader'] = iter(train_curation_loader)

    # lr scheduler setups rely on task-wise dataloader
    if isinstance(hparams['lr_scheduler'], SimSiamCosineScheduler):
        steps_per_epoch = \
                int(np.ceil(len(train_data) / hparams['batch_size']))
        hparams['lr_scheduler_fn'].keywords['steps_per_epoch'] = \
                steps_per_epoch
        hparams['recoverables']['lr_scheduler'] = \
                hparams['lr_scheduler'] = hparams['lr_scheduler_fn']()
        hparams['checkpointer'].add_recoverables(hparams['recoverables'])
        print('==> Adjusting scheduler for {} steps at {}'.format(
            steps_per_epoch, hparams['checkpointer'].checkpoints_dir))

    brain = CLAP(
        modules=hparams['modules'],
        opt_class=hparams['opt_class'],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams['checkpointer'],
    )

    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams['valid_dataloader_opts'],
    )

