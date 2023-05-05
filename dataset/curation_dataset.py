from torch.utils.data import Dataset
import os
import pandas as pd
import glob
import torchaudio
import torch
import numpy as np
import random
from tqdm import tqdm
import pdb


def load_curation_audio(file_path, target_sr, duration):
    random.seed(123)
    sig, read_sr = torchaudio.load(file_path)
    sig = torch.squeeze(sig)
    if len(sig.shape) > 1:
        sig = torch.mean(sig, dim=0)
    if read_sr != target_sr:
        resampler = torchaudio.transforms.Resample(read_sr, target_sr)
        sig = resampler(sig)
    audio_time_series = sig.reshape(-1)
    # duration selection
    if duration*target_sr >= audio_time_series.shape[0]:
        repeat_factor = int(np.ceil((duration*target_sr) /
                                    audio_time_series.shape[0]))
        # Repeat audio_time_series by repeat_factor to match audio_duration
        audio_time_series = audio_time_series.repeat(repeat_factor)
        # remove excess part of audio_time_series
        audio_time_series = audio_time_series[0:int(duration*target_sr)]
    else:
        # audio_time_series is longer than predefined audio duration,
        # so audio_time_series is trimmed
        start_index = random.randrange(
            audio_time_series.shape[0] - duration*target_sr)
        audio_time_series = audio_time_series[start_index:start_index +
                                              int(duration*target_sr)]
    return audio_time_series


class InsperAudioDataset(Dataset):
    def __init__(self, df_path, sr=44100, duration=5):
        self.df = pd.read_csv(df_path)
        self.target_sr = sr
        self.duration = duration

    def __getitem__(self, idx):
        file_path = self.df['wav_path'][idx]
        audio_time_series = load_curation_audio(file_path, self.target_sr, self.duration)
        return {
            'path': file_path,
            'audio': torch.FloatTensor(audio_time_series),
        }

    def __len__(self):
        return len(self.df)


class AudioSetDataset(Dataset):
    def __init__(self, pathfile, start=0, end=None, sr=44100, duration=5):
        super().__init__()
        self.audio_paths = []
        self.load_paths(pathfile)
        if end is not None:
            self.audio_paths = self.audio_paths[start:end]
        else:
            self.audio_paths = self.audio_paths[start:]
        self.target_sr = sr
        self.duration = duration

    def load_paths(self, pathfile):
        with open(pathfile, 'r') as ff:
            paths = ff.readlines()
        self.audio_paths = [p.strip() for p in paths]

    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        audio_time_series = load_curation_audio(file_path, self.target_sr, self.duration)
        return {
            'path': file_path,
            'audio': torch.FloatTensor(audio_time_series),
        }

    def __len__(self):
        return len(self.audio_paths)


class ESCDataset(Dataset):
    def __init__(self, root_dir, selected_folds, sr=44100, duration=5):
        self.audio_paths = []
        meta_df = pd.read_csv(os.path.join(root_dir, 'meta', 'esc50.csv'))
        for idx, row in tqdm(meta_df.iterrows()):
            fold = row['fold']
            fname = os.path.join(root_dir, 'audio', row['filename'])
            if os.path.exists(fname):
                if fold in selected_folds:
                    self.audio_paths.append(fname)
        print('Found {} recordings for ESC curation'.format(len(self.audio_paths)))
        self.target_sr = sr
        self.duration = duration

    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        audio_time_series = load_curation_audio(file_path, self.target_sr, self.duration)
        return {
            'path': file_path,
            'audio': torch.FloatTensor(audio_time_series),
        }

    def __len__(self):
        return len(self.audio_paths)


class UrbanSoundDataset(Dataset):
    def __init__(self, root_dir, selected_folds, sr=44100, duration=5):
        self.audio_paths = []
        meta_df = pd.read_csv(os.path.join(root_dir, 'metadata', 'UrbanSound8K.csv'))
        for idx, row in tqdm(meta_df.iterrows()):
            fold = row['fold']
            fname = os.path.join(root_dir, 'audio', 'fold{}'.format(fold), row['slice_file_name'])
            if os.path.exists(fname):
                if fold in selected_folds:
                    self.audio_paths.append(fname)
        print('Found {} recordings for UrbanSound curation'.format(len(self.audio_paths)))
        self.target_sr = sr
        self.duration = duration

    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        audio_time_series = load_curation_audio(file_path, self.target_sr, self.duration)
        return {
            'path': file_path,
            'audio': torch.FloatTensor(audio_time_series),
        }

    def __len__(self):
        return len(self.audio_paths)


class TAU19Dataset(Dataset):
    def __init__(self, root_dir, selected_folds, sr=44100, duration=5):
        self.audio_paths = []
        for fold in selected_folds:
            meta_csv = os.path.join(root_dir, 'evaluation_setup', 'fold1_{}.csv'.format(fold))
            meta_df = pd.read_csv(meta_csv, sep='\t')
            for idx, row in tqdm(meta_df.iterrows()):
                fname = os.path.join(root_dir, row['filename'])
                self.audio_paths.append(fname)
        print('Found {} recordings for TAU19 curation'.format(len(self.audio_paths)))
        self.target_sr = sr
        self.duration = duration

    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        audio_time_series = load_curation_audio(file_path, self.target_sr, self.duration)
        return {
            'path': file_path,
            'audio': torch.FloatTensor(audio_time_series),
        }

    def __len__(self):
        return len(self.audio_paths)


class TUT17Dataset(Dataset):
    def __init__(self, root_dir, selected_folds, sr=44100, duration=5):

        train_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-development', 'meta.txt')
        with open(train_path, 'r') as f_train:
            train_info = f_train.readlines()
        test_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-evaluation', 'meta.txt')
        with open(test_path, 'r') as f_test:
            test_info = f_test.readlines()

        self.audio_paths = []
        for split, meta_df in zip(['train', 'test'], [train_info, test_info]):
            for row in tqdm(meta_df):
                line = row.split('\t')
                split_dir = 'TUT-acoustic-scenes-2017-development' if split != 'test' \
                        else 'TUT-acoustic-scenes-2017-evaluation'
                fname = os.path.join(root_dir, split_dir, line[0])
                if os.path.exists(fname):
                    self.audio_paths.append(fname)

        print('Found {} recordings for TUT17 curation'.format(len(self.audio_paths)))
        self.target_sr = sr
        self.duration = duration

    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        audio_time_series = load_curation_audio(file_path, self.target_sr, self.duration)
        return {
            'path': file_path,
            'audio': torch.FloatTensor(audio_time_series),
        }

    def __len__(self):
        return len(self.audio_paths)


class InsperCaptionDataset(Dataset):
    def __init__(self, df_path):
        self.df = pd.read_csv(df_path)
        self.df[['caption']] = self.df[['caption']].fillna('')
        self.df['caption'] = self.df['caption'].apply(str)

    def __getitem__(self, idx):
        return {
            'idx': idx,
            'text': self.df['caption'][idx]
        }

    def __len__(self):
        return len(self.df)


class AudioSetEmbeddingDataset(Dataset):
    def __init__(self, pathfile, embfile):
        super().__init__()
        self.audio_paths = []
        self.load_paths(pathfile)
        self.tensor = torch.from_numpy(np.load(embfile))
        assert len(self.audio_paths) == self.tensor.shape[0]

    def load_paths(self, pathfile):
        with open(pathfile, 'r') as ff:
            paths = ff.readlines()
        #  store the absolute path for this version
        self.audio_paths = [p.strip() for p in paths]
        #  # store the basename to support root paths on differnt machines
        #  self.audio_paths = [os.path.basename(p.strip()) for p in paths]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        return {
            'path': self.audio_paths[idx],
            'emb': self.tensor[idx],
        }


class ESCEmbeddingDataset(Dataset):
    def __init__(self, root_dir, selected_folds, embfile):
        self.audio_paths = []
        meta_df = pd.read_csv(os.path.join(root_dir, 'meta', 'esc50.csv'))
        for idx, row in tqdm(meta_df.iterrows()):
            fold = row['fold']
            fname = os.path.join(root_dir, 'audio', row['filename'])
            if os.path.exists(fname):
                if fold in selected_folds:
                    self.audio_paths.append(fname)
                    #  # this stores the "basename"
                    #  self.audio_paths.append(os.path.join('audio', row['filename']))
        self.tensor = torch.from_numpy(np.load(embfile))
        assert len(self.audio_paths) == self.tensor.shape[0]

    def __getitem__(self, idx):
        return {
            'path': self.audio_paths[idx],
            'emb': self.tensor[idx],
        }

    def __len__(self):
        return len(self.audio_paths)


class UrbanSoundEmbeddingDataset(Dataset):
    def __init__(self, root_dir, selected_folds, embfile):
        self.audio_paths = []
        meta_df = pd.read_csv(os.path.join(root_dir, 'metadata', 'UrbanSound8K.csv'))
        for idx, row in tqdm(meta_df.iterrows()):
            fold = row['fold']
            fname = os.path.join(root_dir, 'audio', 'fold{}'.format(fold), row['slice_file_name'])
            if os.path.exists(fname):
                if fold in selected_folds:
                    self.audio_paths.append(fname)
                    #  # strips the root path
                    #  self.audio_paths.append(os.path.join('audio', 'fold{}'.format(fold), row['slice_file_name']))
        self.tensor = torch.from_numpy(np.load(embfile))
        assert len(self.audio_paths) == self.tensor.shape[0]

    def __getitem__(self, idx):
        return {
            'path': self.audio_paths[idx],
            'emb': self.tensor[idx],
        }

    def __len__(self):
        return len(self.audio_paths)


class TAU19EmbeddingDataset(Dataset):
    def __init__(self, root_dir, selected_folds, embfile):
        self.audio_paths = []
        for fold in selected_folds:
            meta_csv = os.path.join(root_dir, 'evaluation_setup', 'fold1_{}.csv'.format(fold))
            meta_df = pd.read_csv(meta_csv, sep='\t')
            for idx, row in tqdm(meta_df.iterrows()):
                fname = os.path.join(root_dir, row['filename'])
                self.audio_paths.append(fname)
        print('Found {} recordings for TAU19 curation'.format(len(self.audio_paths)))
        self.tensor = torch.from_numpy(np.load(embfile))
        assert len(self.audio_paths) == self.tensor.shape[0]

    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        return {
            'path': file_path,
            'emb': self.tensor[index],
        }

    def __len__(self):
        return len(self.audio_paths)


class TUT17EmbeddingDataset(Dataset):
    def __init__(self, root_dir, selected_folds, embfile):
        train_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-development', 'meta.txt')
        with open(train_path, 'r') as f_train:
            train_info = f_train.readlines()
        test_path = os.path.join(root_dir, 'TUT-acoustic-scenes-2017-evaluation', 'meta.txt')
        with open(test_path, 'r') as f_test:
            test_info = f_test.readlines()

        self.audio_paths = []
        for split, meta_df in zip(['train', 'test'], [train_info, test_info]):
            for row in tqdm(meta_df):
                line = row.split('\t')
                split_dir = 'TUT-acoustic-scenes-2017-development' if split != 'test' \
                        else 'TUT-acoustic-scenes-2017-evaluation'
                fname = os.path.join(root_dir, split_dir, line[0])
                if os.path.exists(fname):
                    self.audio_paths.append(fname)

        print('Found {} recordings for TUT17 curation'.format(len(self.audio_paths)))
        self.tensor = torch.from_numpy(np.load(embfile))
        assert len(self.audio_paths) == self.tensor.shape[0]

    def __getitem__(self, index):
        file_path = self.audio_paths[index]
        return {
            'path': file_path,
            'emb': self.tensor[index],
        }

    def __len__(self):
        return len(self.audio_paths)


def test_audio_dataset():
    ds = AudioSetDataset('/home/zhepei/workspace/microsoft_clap/CLAP/unsupervised_curation/good_files.txt')
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for batch in loader:
        pdb.set_trace()


def test_esc_dataset():
    ds = ESCDataset('/mnt/data2/Sound Sets/ESC-50', [2,3,4,5])
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for batch in loader:
        pdb.set_trace()


def test_tau19_dataset():
    ds = TAU19Dataset('/mnt/data2/Sound Sets/TAU-urban-acoustic-scenes-2019-development', ['train', 'test'])
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for batch in loader:
        pdb.set_trace()


def test_urbansound_dataset():
    ds = UrbanSoundDataset('/mnt/data2/Sound Sets/UrbanSound8K/UrbanSound8K', [2,3,4,5])
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for batch in loader:
        pdb.set_trace()
def test_caption_dataset():
    from torch.utils.data import DataLoader
    ds = InsperCaptionDataset('./train_raw.csv')
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    for batch in loader:
        pdb.set_trace()


if __name__ == '__main__':
    test_tau19_dataset()

