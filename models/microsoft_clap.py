import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from transformers import BatchEncoding
import pdb


def get_audio_encoder(name: str):
    if name == "Cnn14":
        return Cnn14
    else:
        raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)


    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)

        self.bn_att = nn.BatchNorm1d(n_out)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num, out_emb):

        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        # out_emb is 2048 for best Cnn14
        self.fc1 = nn.Linear(2048, out_emb, bias=True)
        self.fc_audioset = nn.Linear(out_emb, classes_num, bias=True)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)
        """

        with torch.cuda.amp.autocast(enabled=False):
            x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int) -> None:
        super().__init__()

        audio_encoder = get_audio_encoder(audioenc_name)

        self.base = audio_encoder(
            sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in)

        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output

class TextEncoder(nn.Module):
    def __init__(self, d_out: int, text_model: str, transformer_embed_dim: int) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(text_model)
        
        self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x):
        out = self.base(**x)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        return projected_vec

class CLAP(nn.Module):
    def __init__(self,
                # audio
                audioenc_name: str,
                sample_rate: int, 
                window_size: int, 
                hop_size: int, 
                mel_bins: int, 
                fmin: int, 
                fmax: int, 
                classes_num: int, 
                out_emb: int,
                # text
                text_model: str,
                transformer_embed_dim: int,
                # common
                d_proj: int,
                ):
        super().__init__()

        
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)

        self.caption_encoder = TextEncoder(
            d_proj, text_model, transformer_embed_dim
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio, input_ids, token_type_ids, attention_mask, compute_sim=True):
        text = BatchEncoding({
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
        })
        audio_embed, _ = self.audio_encoder(audio)
        caption_embed = self.caption_encoder(text)
        # normalize features
        audio_embed = audio_embed / audio_embed.norm(dim=1, keepdim=True)
        caption_embed = caption_embed / caption_embed.norm(dim=1, keepdim=True)
        if compute_sim:
            logit_scale = self.logit_scale.exp()
            similarity = logit_scale * audio_embed @ caption_embed.t()
            similarity = torch.clamp(similarity, max=100)

            return caption_embed, audio_embed, similarity
        else:
            return caption_embed, audio_embed
