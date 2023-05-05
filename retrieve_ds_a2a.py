import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import sys
import pdb


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    audio_bs = 40000
    target_dataset = hparams['target_dataset']
    target_loader = DataLoader(target_dataset, batch_size=audio_bs, shuffle=False, num_workers=16)
    source_emb = torch.from_numpy(np.load(hparams['source_emb_path']))
    device = torch.device('cuda:0')
    source_emb = source_emb.to(device)

    target_out_files = []
    target_out_emb = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(target_loader)):
            target_emb = batch['emb'].to(device)
            similarity = source_emb @ target_emb.T
            # for each target text, find its most relevant source text
            max_sim = similarity.amax(dim=0)
            keep_idx = torch.nonzero(max_sim >= hparams['sim_threshold']).cpu().numpy()
            if len(keep_idx.flatten()) > 0:
                save_emb = target_emb[keep_idx.flatten()].half().cpu().numpy()
                target_out_emb.append(save_emb)
                save_paths = (np.array(batch['path'])[keep_idx.flatten().tolist()]).tolist()
                target_out_files.extend(save_paths)
            #  if i == 5:
            #      break

    # saving filtered embeddings
    np.save(hparams['output_emb_path'], np.concatenate(target_out_emb, axis=0))

    # saving filtered file paths (this should contain the full path)
    target_out_files = [os.path.join(hparams['target_root_dir'], e) for e in target_out_files]
    with open(hparams['output_file_path'], 'w') as fp:
        fp.write('\n'.join(target_out_files))
