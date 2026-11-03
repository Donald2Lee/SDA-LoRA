from pathlib import Path
import mne
import numpy as np
import scipy.io
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader


class SEEDProcessor:
    def __init__(self, 
                  raw_root,
                  processed_root,
                  window_sec=1,
                  sample_rate=200,
                 ):
        self.raw_root = Path(raw_root).resolve()
        self.processed_root = Path(processed_root).resolve()
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.window_sec = window_sec
        self.sample_rate = sample_rate

        self.LABRAM_SUPPORTED_CHANNELS = [
                    # 1-14: prefrontal region
                    "FP1", "FPZ", "FP2",
                    "AF9", "AF7", "AF5", "AF3", "AF1",
                    "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
                    # 15-25: frontal region
                    "F9", "F7", "F5", "F3", "F1",
                    "FZ", "F2", "F4", "F6", "F8", "F10",
                    # 26-36: frontotemporal and frontocentral regions
                    "FT9", "FT7",
                    "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
                    "FT8", "FT10",
                    # 37-47: temporal and central regions
                    "T9", "T7",
                    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
                    "T8", "T10",
                    # 48-58: temporoparietal and centroparietal regions
                    "TP9", "TP7",
                    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
                    "TP8", "TP10",
                    # 59-69: parietal region
                    "P9", "P7", "P5", "P3", "P1",
                    "PZ", "P2", "P4", "P6", "P8", "P10",
                    # 70-80: parieto-occipital region
                    "PO9", "PO7", "PO5", "PO3", "PO1",
                    "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
                    # 81-88: occipital and cerebellar regions
                    "O1", "OZ", "O2", "O9",
                    "CB1", "CB2", "IZ", "O10",
                    # 89-96: legacy temporal channels and reference electrodes
                    "T3", "T5", "T4", "T6",
                    "M1", "M2", "A1", "A2",
                    # 97-112: extended central region
                    "CFC1", "CFC2", "CFC3", "CFC4",
                    "CFC5", "CFC6", "CFC7", "CFC8",
                    "CCP1", "CCP2", "CCP3", "CCP4",
                    "CCP5", "CCP6", "CCP7", "CCP8",
                    # 113-120: extended temporal region
                    "T1", "T2",
                    "FTT9H", "TTP7H", "TPP9H",
                    "FTT10H", "TPP8H", "TPP10H",
                    # 121-128: supported bipolar derivations
                    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
                ]

    
    def save_to_pt(self):
        CH_NAMES = [
                    "Fp1", "Fpz", "Fp2",
                    "AF3", "AF4",
                    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
                    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
                    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
                    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
                    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
                    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
                    "POO7", "O1", "Oz", "O2", "POO8"
                    ] # CB1=POO7, CB2=POO8
        
        channel_aliases = {"POO7": "CB1", "POO8": "CB2"}

        normalised_ch_names = [channel_aliases.get(name.strip().upper(), name.strip().upper()) for name in CH_NAMES]
        
        # normalised_ch_names = [name.strip().upper() for name in CH_NAMES]
        input_chans = torch.tensor([0] + [self.LABRAM_SUPPORTED_CHANNELS.index(name) + 1 for name in normalised_ch_names], dtype=torch.long)
        
        montage = mne.channels.make_standard_montage("standard_1005")
        montage_pos = montage.get_positions()["ch_pos"]
        positions = np.array([montage_pos[ch_name] for ch_name in CH_NAMES], dtype=np.float32)

        output_path = self.processed_root / "SEED_LaBraM.pt"
        if output_path.exists():
            print(f"Processed file already exists: {output_path}")
            return
        
        subject, session, trial, data, label, position, ch_idx = [], [], [], [], [], [], []

        # low_freq, high_freq = 1, 47
        # wn = [low_freq * 2 / self.sample_rate, high_freq * 2 / self.sample_rate]
        # b, a = signal.cheby2(6, 60, wn, btype="bandpass")
        notch_b, notch_a = signal.iirnotch(w0=50.0, Q=30.0, fs=self.sample_rate)
        
        label_path = self.raw_root / "label.mat"
        labels = np.squeeze(scipy.io.loadmat(str(label_path))["label"])

        for sub in range(1, 16):
            sub_mat_files = sorted(self.raw_root.glob(f"{sub}_*.mat"), key=lambda p: p.stem.split("_")[1])
            for sess, mat_path in enumerate(sub_mat_files):
                sub_sess_data = scipy.io.loadmat(str(mat_path))
                print(f"Processing sub={sub}, session={sess + 1}")
                prefix = next(k.split("_eeg")[0] for k in sub_sess_data if "_eeg" in k)
                for trial_idx in range(1, 16):
                    sub_sess_trial_data = sub_sess_data[f"{prefix}_eeg{trial_idx}"]

                    # channel_mean = np.mean(sub_sess_trial_data, axis=1, keepdims=True)
                    # channel_std = np.std(sub_sess_trial_data, axis=1, keepdims=True)
                    # sub_sess_trial_data = (sub_sess_trial_data - channel_mean) / channel_std
                    sub_sess_trial_data = sub_sess_trial_data / 100 # LaBraM input scaling

                    sub_sess_trial_data = signal.filtfilt(notch_b, notch_a, sub_sess_trial_data, axis=1,).copy()

                    # sub_sess_trial_data = signal.filtfilt(b, a, sub_sess_trial_data, axis=1).copy()

                    one_sample_sqlength = int(self.sample_rate * self.window_sec)
                    sample_num_in_a_trial = sub_sess_trial_data.shape[1] // one_sample_sqlength
                    for tmp_num in range(sample_num_in_a_trial):
                        start = tmp_num * one_sample_sqlength
                        end = (tmp_num + 1) * one_sample_sqlength

                        subject.append(torch.tensor(sub, dtype=torch.long))
                        session.append(torch.tensor(sess + 1, dtype=torch.long))
                        trial.append(torch.tensor(trial_idx, dtype=torch.long))
                        data.append(torch.tensor(sub_sess_trial_data[:, start:end], dtype=torch.float32)) # channel * time_step
                        label.append(torch.tensor(labels[trial_idx - 1] + 1, dtype=torch.long)) # 0, 1, 2
                        position.append(torch.tensor(positions, dtype=torch.float32))
        
        data_dict = {
                    'subject': torch.stack(subject),
                    'session': torch.stack(session),
                    'trial': torch.stack(trial),
                    'data': torch.stack(data),
                    'label': torch.stack(label),
                    'position': torch.stack(position),
                    'input_chans': input_chans,
                }
        torch.save(data_dict, output_path)
        print(f"Saved processed SEED data to {output_path}")

    def load_dataset(self, config, target=1):
        session = config['data']['session']
        data_dict = torch.load(self.processed_root / "SEED_LaBraM.pt", weights_only=True)

        source_index = (data_dict['subject'] != target) & (data_dict['session'] == session)
        source_data_dict = {
            'subject': data_dict['subject'][source_index],
            'session': data_dict['session'][source_index],
            'trial': data_dict['trial'][source_index],
            'data': data_dict['data'][source_index],
            'label': data_dict['label'][source_index],
            'position': data_dict['position'][source_index],
            'input_chans': data_dict['input_chans'],
        }

        target_index = (data_dict['subject'] == target) & (data_dict['session'] == session)
        target_data_dict = {
            'subject': data_dict['subject'][target_index],
            'session': data_dict['session'][target_index],
            'trial': data_dict['trial'][target_index],
            'data': data_dict['data'][target_index],
            'label': data_dict['label'][target_index],
            'position': data_dict['position'][target_index],
            'input_chans': data_dict['input_chans'],
        }

        print(f"Source data size: {len(source_data_dict['data'])}")
        print(f"Target data size: {len(target_data_dict['data'])}")

        dataset_dict = {
            "source_dataset": ToDataset(source_data_dict, input_chans=source_data_dict["input_chans"]),
            "target_dataset": ToDataset(target_data_dict, input_chans=target_data_dict["input_chans"]),
        }

        return dataset_dict
    
    def load_dataset_for_visual(self, config):
        session = config['data']['session']
        data_dict = torch.load(self.processed_root / "SEED_LaBraM.pt", weights_only=True)

        # Retain all subjects in the selected session.
        visual_index = data_dict["session"] == session

        visual_data_dict = {
            "subject": data_dict["subject"][visual_index],
            "session": data_dict["session"][visual_index],
            "trial": data_dict["trial"][visual_index],
            "data": data_dict["data"][visual_index],
            "label": data_dict["label"][visual_index],
            "position": data_dict["position"][visual_index],
            # input_chans is shared by all samples and has no sample dimension.
            "input_chans": data_dict["input_chans"],
        }

        visual_dataset = ToDataset(visual_data_dict, input_chans=visual_data_dict["input_chans"])

        return visual_dataset


class ToDataset(Dataset):
    def __init__(self, data_dict, input_chans):
        self.subject = data_dict["subject"]
        self.session = data_dict["session"]
        self.trial = data_dict["trial"]
        self.data = data_dict["data"]
        self.label = data_dict["label"]
        self.position = data_dict["position"]

        # self.input_chans = torch.as_tensor(
        #     data_dict["input_chans"],
        #     dtype=torch.long,
        # )

        self.input_chans = torch.as_tensor(
            input_chans,
            dtype=torch.long,
        )

        n = len(self.data)
        assert len(self.subject) == n
        assert len(self.session) == n
        assert len(self.trial) == n
        assert len(self.label) == n
        assert len(self.position) == n

        # For 62 EEG channels, input_chans contains CLS plus 62 channel indices.
        assert self.input_chans.ndim == 1
        assert len(self.input_chans) == self.data.shape[1] + 1
        assert self.input_chans[0].item() == 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "subject": self.subject[idx],
            "session": self.session[idx],
            "trial": self.trial[idx],
            "data": self.data[idx],
            "label": self.label[idx],
            "position": self.position[idx],
        } 
            
if __name__ == '__main__':
    raw_root = "D:/about_code/AEmotionData/SEED/raw"
    processed_root = "D:/about_code/AEmotionData/SEED/processed"

    processor = SEEDProcessor(
        raw_root=raw_root,
        processed_root=processed_root,
    )
    processor.save_to_pt()


