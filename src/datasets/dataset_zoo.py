import math
import random
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, 
                 data_pth: str | list,
                 window: int | float,
                 sr: int | float=100,
                 norm_midi: bool=False,
                 weight: str | List[str] | bool=False,
                 is_valid: bool=False,) -> None:
        super().__init__()
        if isinstance(data_pth, str):
            self.data_dir = sorted(Path(data_pth).rglob('*.csv'))
        elif isinstance(data_pth, list):
            self.data_dir = data_pth
        
        self.weight = weight
        if isinstance(weight, str):
            self.weight_cols = [weight]
        elif hasattr(weight, "__iter__"):
            self.weight_cols = list(weight)
        else:
            self.weight_cols = []

        self.is_valid = is_valid
        self.norm_midi = norm_midi

        self.sr = sr
        self.window = window

        self.loaded_data = self._load_data()


    def _frequency_to_midi(self, 
                          frequency: float) -> float:
        return 69 + 12 * math.log2(frequency / 440)


    def _get_data(self, 
                  csv_file: Path) -> Tuple[torch.Tensor, float]:
        contour_df = pd.read_csv(csv_file, dtype={'frequency':np.float32})
        frequency = contour_df['frequency'].values
        if self.weight_cols:
            weight = contour_df[self.weight_cols].values
        else:
            weight = np.ones((frequency.shape[-1], 1))

        midi = [self._frequency_to_midi(freq) for freq in frequency]
        if self.norm_midi:
            tonic = Counter(np.round(midi)).most_common(1)[0][0]
        else:
            tonic = 0
        norm_midi = [(mi-float(tonic))/12 for mi in midi]

        norm_midi = torch.tensor(norm_midi, dtype=torch.float32)
        weight = torch.tensor(weight.T, dtype=torch.float32)

        return norm_midi, weight, tonic


    def _load_data(self) -> List[Tuple[torch.Tensor, float]]:
        loaded_data = []
        for csv_file in tqdm(self.data_dir, desc='Load Data'):
            norm_midi, weight, tonic = self._get_data(csv_file)
            loaded_data.append((norm_midi, weight, tonic))
        
        return loaded_data
    

    def __len__(self) -> int:
        return len(self.loaded_data)


    def __getitem__(self, idx:int):
        raise NotImplementedError("Subclasses must implement this method")




class FullPitchDataset(BaseDataset):
    def __init__(self, 
                 data_pth: str | list,
                 window: int,
                 sr: int | float=100,
                 norm_midi: bool=False,
                 weight: str | List[str] | bool=False,
                 is_valid: bool=False,
                 threshold: None | Dict=None,
                 threshold_q: None | Dict=None,
                 num_interpolate: int=2,) -> None:
        super().__init__(data_pth, window, sr, norm_midi, weight, is_valid)
        self.threshold = threshold if threshold is not None else {}
        self.threshold_q = threshold_q if threshold_q is not None else {}
        self.num_interpolate = num_interpolate
        self._validate_threshold_keys()


    def _validate_threshold_keys(self):
        t, tq = set(self.threshold), set(self.threshold_q)
        if t & tq:
            raise ValueError(f"key {t & tq} is in threshold, threshold_q")

        unknown = (t | tq) - set(self.weight_cols)
        if unknown:
            raise ValueError(f"threshold key {unknown} is not in self.weight")


    def _aggregate(self, 
                    w: torch.Tensor) -> torch.Tensor:
        if not self.weight_cols:
            is_filtered = torch.zeros(w.shape[-1], dtype=torch.bool)
            return is_filtered

        filtered = []
        for i, name in enumerate(self.weight_cols):
            if name in self.threshold:
                threshold = self.threshold[name]
            elif name in self.threshold_q:
                threshold = torch.quantile(w[i][w[i] > 0], self.threshold_q[name])
            else:
                continue
            filtered.append(w[i] < threshold)

        is_filtered = torch.stack(filtered, dim=0).any(dim=0) if filtered else torch.zeros(w.shape[-1], dtype=torch.bool)

        return is_filtered


    def aggregate_weight_filter(self, 
                                w:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")


    def get_segment_by_split(self, 
                             x: torch.Tensor, 
                             w: torch.Tensor, 
                             is_filtered: torch.Tensor) -> Tuple[torch.Tensor]:
        x_len = x.shape[-1]
        max_start = x_len - self.window

        if self.is_valid:
            n_samples = x_len//self.window
            x = torch.cat([x[:self.window*n_samples].reshape(n_samples,self.window), x[-self.window:].unsqueeze(0)], dim=0)
            w = torch.cat([w[...,:self.window*n_samples].reshape(-1,n_samples,self.window).permute(1,0,2), w[...,-self.window:].unsqueeze(0)], dim=0)
            is_filtered = torch.cat([is_filtered[:self.window*n_samples].reshape(n_samples,self.window), is_filtered[-self.window:].unsqueeze(0)], dim=0)
            n_samples += 1
        else:
            start = random.randint(0, max_start)
            n_samples = (x_len-start)//self.window
            x = x[start:start+self.window*n_samples].reshape(n_samples,self.window)
            w = w[...,start:start+self.window*n_samples].reshape(-1,n_samples,self.window).permute(1,0,2)
            is_filtered = is_filtered[start:start+self.window*n_samples].reshape(n_samples,self.window)
    
        return x, w, is_filtered


    def filter_receptive_field(self, 
                              is_filtered: torch.Tensor, 
                              return_ids: bool) -> torch.Tensor:
        interpolate_mask = ~(is_filtered.unfold(dimension=-1, step=1, size=self.num_interpolate+1).float().sum(dim=-1) >= self.num_interpolate + 1).sum(dim=-1).bool()
        start_end_mask = ~(is_filtered[...,0] | is_filtered[...,-1])
        
        final_mask = interpolate_mask & start_end_mask
        use_ids = torch.where(final_mask)[0]

        if return_ids:
            return final_mask, use_ids

        return final_mask


    def interpolate_filtered_point(self, 
                                   x: torch.Tensor, 
                                   is_filtered: torch.Tensor) -> torch.Tensor:
        n, w = x.shape
        pos = torch.arange(w).expand(n, w)

        left_src  = torch.where(is_filtered, torch.full_like(pos, -1), pos)
        left_idx  = torch.cummax(left_src, dim=-1).values

        right_src = torch.where(is_filtered, torch.full_like(pos, w), pos)
        right_idx = torch.flip(torch.cummin(torch.flip(right_src, [-1]), dim=-1).values, [-1])

        left_val  = torch.gather(x, -1, left_idx)
        right_val = torch.gather(x, -1, right_idx)

        span      = (right_idx - left_idx).clamp_min(1).to(x.dtype)
        frac      = (pos - left_idx).to(x.dtype) / span
        interp    = left_val + (right_val - left_val) * frac

        x_out = torch.where(is_filtered, interp, x)

        return x_out


    def _cache_filter(self) -> None:
        self.loaded_data = [(x, w, tonic, self.aggregate_weight_filter(w))
                            for x, w, tonic in self.loaded_data]


    def sample_receptive_field(self) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        sampled_x, sampled_w, sampled_tonic = [], [], []
        for x, w, tonic, is_filtered_full in self.loaded_data:
            if x.shape[-1] - self.window < 0:
                continue

            x, w, is_filtered = self.get_segment_by_split(x, w, is_filtered_full)
            final_mask = self.filter_receptive_field(is_filtered, return_ids=False)

            x = x[final_mask]
            w = w[final_mask]
            is_filtered = is_filtered[final_mask]
            tonic = np.repeat(tonic, x.shape[0])

            x = self.interpolate_filtered_point(x, is_filtered)

            sampled_x.append(x)
            sampled_w.append(w.numpy())
            sampled_tonic.append(tonic)
        
        sampled_data = (torch.from_numpy(np.concatenate(sampled_x, axis=0)), 
                        torch.from_numpy(np.concatenate(sampled_w, axis=0)), 
                        np.concatenate(sampled_tonic, axis=0))

        return sampled_data


    def resample_receptive_field(self) -> None:
        self.sampled_data = self.sample_receptive_field()


    def __len__(self) -> int:
        return len(self.sampled_data[0])


    def __getitem__(self, 
                    idx: int) -> Tuple[torch.Tensor]:
        x, w, meta = self.sampled_data[0][idx], self.sampled_data[1][idx], self.sampled_data[2][idx]
        return x.unsqueeze(0), w, meta




class FullPitchThresholdDataset(FullPitchDataset):
    def __init__(self, 
                 data_pth: str | list,
                 window: int,
                 sr: int | float=100,
                 norm_midi: bool=False,
                 weight: str | List[str] | bool=False,
                 is_valid: bool=False,
                 threshold: None | Dict=None,
                 threshold_q: None | Dict=None,
                 num_interpolate: int=2,) -> None:
        super().__init__(data_pth, window, sr, norm_midi, weight, is_valid, threshold, threshold_q, num_interpolate)
        self._cache_filter()
        self.sampled_data = self.sample_receptive_field()


    def aggregate_weight_filter(self,
                                w:torch.Tensor) -> torch.Tensor:
        return self._aggregate(w)




class NIAFixedClsDataset(BaseDataset):
    def __init__(self, 
                 data_pth: str,
                 data_info: pd.DataFrame, 
                 label_map: dict,
                 window: int=128,
                 sr: int | float=100,
                 norm_midi: bool=False,
                 weight: str | List[str] | bool=False,
                 **kwargs) -> None:
        self.data_dir = Path(data_pth)
        self.data_info = data_info
        self.label_map = label_map

        self.sr = sr
        self.window = window
        self.weight = weight
        self.norm_midi = norm_midi

        if isinstance(weight, str):
            self.weight_cols = [weight]
        elif hasattr(weight, "__iter__"):
            self.weight_cols = list(weight)
        else:
            self.weight_cols = []

        self.loaded_data = self._load_data()


    def _crop_segment(self,
                      onset: int,
                      offset: int) -> Tuple[int, int]:
        center = onset + (offset-onset)//2
        onset = center-self.window//2
        offset = onset+self.window

        return onset, offset


    def _expand_segment(self,
                        onset: int,
                        offset: int,
                        seg_df: pd.DataFrame,
                        total_len: int) -> Tuple[int, int]:
        need = self.window - (offset - onset)

        prev_rows = seg_df[seg_df['offset']*100 <= onset]
        next_rows = seg_df[seg_df['onset']*100 >= offset]
        prev_boundary = int(round(prev_rows.iloc[-1]['offset'], 2)*100) if not prev_rows.empty else 0
        next_boundary = int(round(next_rows.iloc[0]['onset'], 2)*100) if not next_rows.empty else total_len

        left_space = onset - prev_boundary
        right_space = next_boundary - offset

        if left_space + right_space >= need:
            if left_space >= right_space:
                l_pad = min(left_space, need)
                r_pad = need - l_pad
            else:
                r_pad = min(right_space, need)
                l_pad = need - r_pad

        else:
            extra = need - (left_space + right_space)
            l_extra = extra // 2
            r_extra = extra - l_extra
            l_pad = left_space + l_extra
            r_pad = right_space + r_extra

        onset = onset - l_pad
        offset = offset + r_pad

        if onset < 0:
            onset = 0
            offset = self.window
        elif offset > total_len:
            offset = total_len
            onset = offset-self.window

        return onset, offset


    def _load_data(self) -> List[Tuple[torch.Tensor, float]]:
        self.check_list = []
        loaded_data = []
        for fn in tqdm(self.data_info['filename'].unique().tolist(), desc='Load Data'):
            x, w, tonic = self._get_data((self.data_dir/fn).with_suffix('.csv'))
            seg_df = self.data_info[self.data_info['filename']==fn]
            
            for _, row in seg_df.iterrows():
                onset, offset, label = int(round(row['onset'],2)*100), int(round(row['offset'],2)*100), row['label']
                seg_len = offset - onset
                seg_info = []
                seg_info.extend([fn, label, onset, offset])

                if self.window < seg_len:
                    onset, offset = self._crop_segment(onset, offset)
                elif self.window > seg_len:
                    onset, offset = self._expand_segment(onset, offset, seg_df, x.shape[-1])

                seg = x[...,onset:offset]
                seg_w = w[...,onset:offset]

                seg_info.extend([onset, offset])
                self.check_list.append(seg_info)
                loaded_data.append((seg, seg_w, tonic, label))
        
        return loaded_data


    def __len__(self) -> int:
        return len(self.loaded_data)
    

    def __getitem__(self, 
                    idx: int) -> Tuple[torch.Tensor]:
        x, w, tonic, label = self.loaded_data[idx]
        y = torch.tensor(self.label_map[label], dtype=torch.long)

        return x.unsqueeze(0), w, tonic, y

class PansoriDataset(FullPitchDataset):
    def __init__(self,
                 data_pth: str,
                 data_info: pd.DataFrame,
                 label_type: str,
                 label_dict: None | dict=None,
                 window: int=128,
                 sr: int | float=100,
                 norm_midi: bool=False,
                 weight: str | List[str] | bool=False,
                 is_valid: bool=False,
                 threshold: None | Dict=None,
                 threshold_q: None | Dict=None,
                 num_interpolate: int=2,
                 num_offset: int=128,
                 **kwargs) -> None:
        self.data_dir = Path(data_pth)
        self.data_info = data_info

        self.sr = sr
        self.window = window
        self.weight = weight
        self.norm_midi = norm_midi
        self.is_valid = is_valid # 호환용, slide가 결정적이므로 미사용
        self.num_offset = num_offset

        if isinstance(weight, str):
            self.weight_cols = [weight]
        elif hasattr(weight, "__iter__"):
            self.weight_cols = list(weight)
        else:
            self.weight_cols = []

        self.threshold = threshold if threshold is not None else {}
        self.threshold_q = threshold_q if threshold_q is not None else {}
        self.num_interpolate = num_interpolate
        self._validate_threshold_keys()

        self.label_type = label_type
        if isinstance(label_dict, dict):
            self.label_map = label_dict
        else:
            self.label_map = {l: i for i, l in enumerate(sorted(l for l in data_info[label_type].unique() if l != 'Unknown'))}
        self.idx2label = {v:k for k,v in self.label_map.items()}

        self.loaded_data = self._load_data()


    def aggregate_weight_filter(self,
                                w: torch.Tensor) -> torch.Tensor:
        return self._aggregate(w)


    def _get_segments(self,
                      fn: str,
                      total_len: int) -> List[Tuple[int, int, int, str]]:
        # (idx, onset_frame, offset_frame, label) 목록 생성
        seg_df = self.data_info[self.data_info['filename']==fn]
        if self.label_type == 'label':
            seg_df = seg_df[seg_df[self.label_type].isin(self.label_map.keys())]
            return [(idx, int(round(row['onset'],2)*100), int(round(row['offset'],2)*100), row[self.label_type])
                    for idx, row in seg_df.iterrows()]

        label = seg_df[self.label_type].iloc[0]
        if label not in self.label_map:
            return []
        return [(0, 0, total_len, label)]


    def _roll_segment(self,
                      x: torch.Tensor,
                      w: torch.Tensor,
                      is_filtered: torch.Tensor,
                      onset: int,
                      offset: int) -> None | Tuple[torch.Tensor, ...]:
        x_seg, w_seg, f_seg = x[onset:offset], w[...,onset:offset], is_filtered[onset:offset]
        if x_seg.shape[-1] < self.window:
            return None

        x_r = x_seg.unfold(dimension=-1, size=self.window, step=self.num_offset)
        w_r = w_seg.unfold(dimension=-1, size=self.window, step=self.num_offset).permute(1,0,2)
        f_r = f_seg.unfold(dimension=-1, size=self.window, step=self.num_offset)
        onset_ids = torch.arange(x_r.shape[0])

        if (x_seg.shape[-1]-self.window) % self.num_offset != 0: # 잔여분은 끝에서 한 window 추가, onset_id=-1로 마킹
            x_r = torch.cat([x_r, x_seg[-self.window:].unsqueeze(0)])
            w_r = torch.cat([w_r, w_seg[...,-self.window:].unsqueeze(0)])
            f_r = torch.cat([f_r, f_seg[-self.window:].unsqueeze(0)])
            onset_ids = torch.cat([onset_ids, torch.tensor([-1])])

        return x_r, w_r, f_r, onset_ids


    def _filter_rolled(self,
                       x_r: torch.Tensor,
                       w_r: torch.Tensor,
                       f_r: torch.Tensor,
                       onset_ids: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        final_mask = self.filter_receptive_field(f_r, return_ids=False)
        x_r, w_r, f_r, onset_ids = x_r[final_mask], w_r[final_mask], f_r[final_mask], onset_ids[final_mask]
        x_r = self.interpolate_filtered_point(x_r, f_r)

        return x_r, w_r, onset_ids


    def _load_data(self):
        loaded_data = []
        for fn in tqdm(self.data_info['filename'].unique().tolist(), desc='Load Data'):
            x, w, tonic = self._get_data((self.data_dir/fn).with_suffix('.csv'))
            is_filtered = self.aggregate_weight_filter(w)

            for idx, onset, offset, label in self._get_segments(fn, x.shape[-1]):
                rolled = self._roll_segment(x, w, is_filtered, onset, offset)
                if rolled is None:
                    continue

                x_r, w_r, onset_ids = self._filter_rolled(*rolled)
                if x_r.shape[0] == 0:
                    continue

                loaded_data.extend((seg, seg_w, (tonic, fn, idx, onset, oid.item()), label)
                                   for seg, seg_w, oid in zip(x_r, w_r, onset_ids))

        return loaded_data


    def __len__(self) -> int:
        return len(self.loaded_data)


    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor]:
        x, w, meta, label = self.loaded_data[idx]
        y = torch.tensor(self.label_map[label], dtype=torch.long)

        return x.unsqueeze(0), w, meta, y
