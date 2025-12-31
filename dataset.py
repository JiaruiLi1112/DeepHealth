import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List


class HealthDataset(Dataset):
    """
    Dataset for health records.

    Args:
        data_prefix (str): Prefix for data files.
        covariate_list (List[str] | None): List of covariates to include.
    """

    def __init__(
            self,
            data_prefix: str,
            covariate_list: List[str] | None = None,
    ):
        basic_info = pd.read_csv(
            f"{data_prefix}_basic_info.csv", index_col='eid')
        tabular_data = pd.read_csv(f"{data_prefix}_table.csv", index_col='eid')
        event_data = np.load(f"{data_prefix}_event_data.npy")
        patient_events = defaultdict(list)
        vocab_size = 0
        for patient_id, time_in_days, event_code in event_data:
            patient_events[patient_id].append((time_in_days, event_code))
            if event_code > vocab_size:
                vocab_size = event_code
        self.n_disease = vocab_size - 1
        self.basic_info = basic_info.convert_dtypes()
        self.patient_ids = self.basic_info.index.tolist()
        self.patient_events = dict(patient_events)

        tabular_data = tabular_data.convert_dtypes()
        cont_cols = []
        cate_cols = []
        self.cate_dims = []
        if covariate_list is not None:
            tabular_data = tabular_data[covariate_list]
        for col in tabular_data.columns:
            if pd.api.types.is_float_dtype(tabular_data[col]):
                cont_cols.append(col)
            elif pd.api.types.is_integer_dtype(tabular_data[col]):
                series = tabular_data[col]
                unique_vals = series.dropna().unique()
                if len(unique_vals) > 11:
                    cont_cols.append(col)
                else:
                    cate_cols.append(col)
                    self.cate_dims.append(int(series.max()) + 1)

        self.cont_features = tabular_data[cont_cols].to_numpy(
            dtype=np.float32).copy()
        self.cate_features = tabular_data[cate_cols].to_numpy(
            dtype=np.int64).copy()
        self.n_cont = self.cont_features.shape[1]
        self.n_cate = self.cate_features.shape[1]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        records = sorted(self.patient_events.get(
            patient_id, []), key=lambda x: x[0])
        event_seq = [item[1] for item in records]
        time_seq = [item[0] for item in records]

        doa = self.basic_info.loc[patient_id, 'date_of_assessment']

        insert_pos = np.searchsorted(time_seq, doa)
        time_seq.insert(insert_pos, doa)
        # assuming 1 is the code for 'DOA' event
        event_seq.insert(insert_pos, 1)
        event_tensor = torch.tensor(event_seq, dtype=torch.long)
        time_tensor = torch.tensor(time_seq, dtype=torch.float)
        cont_tensor = torch.tensor(
            self.cont_features[idx, :], dtype=torch.float)
        cate_tensor = torch.tensor(
            self.cate_features[idx, :], dtype=torch.long)
        sex = self.basic_info.loc[patient_id, 'sex']

        return (event_tensor, time_tensor, cont_tensor, cate_tensor, sex)


def health_collate_fn(batch):
    event_seqs, time_seqs, cont_feats, cate_feats, sexes = zip(*batch)
    event_batch = pad_sequence(event_seqs, batch_first=True, padding_value=0)
    time_batch = pad_sequence(
        time_seqs, batch_first=True, padding_value=36525.0)
    cont_batch = torch.stack(cont_feats, dim=0)
    cont_batch = cont_batch.unsqueeze(1)  # (B, 1, n_cont)
    cate_batch = torch.stack(cate_feats, dim=0)
    cate_batch = cate_batch.unsqueeze(1)  # (B, 1, n_cate)
    sex_batch = torch.tensor(sexes, dtype=torch.long)
    return event_batch, time_batch, cont_batch, cate_batch, sex_batch
