import argparse
import copy
import datetime
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models.self_attn_model import NextPOIWindowSelfAttn
from st_prior_reranker import STPriorTables, build_st_prior_tables, rerank_topk


# ==========================
# Default configuration (can be overridden by CLI)
# ==========================
DATA_PATH = "data/processed/TKY_excluding_cold.pkl"
WINDOW_SIZE = 1
# Window sampling is fixed to recent (random / random_keep_last are deprecated parameters used in early exploration)
WINDOW_SAMPLING = "recent"

NUM_EPOCHS = 100
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 4096
LR = 3e-4
WEIGHT_DECAY = 5e-6

USE_CAT_EMBEDDING = False
USE_USER_EMBEDDING = True
USE_POSITIONAL_ENCODING = True
CLASSIFIER_TOKEN_POSITION = "first"

# ToD slot embedding (multi-scale time-of-day buckets)
USE_TOD_SLOT_EMBEDDING = True
TOD_SLOT_SCALES_DEFAULT = [6, 12, 24]
TOD_SLOT_SCALES: Optional[List[int]] = None
TOD_SLOT_EMB_DIM = 64

SELF_ATTN_D_MODEL = 128
SELF_ATTN_NUM_LAYERS = 2
SELF_ATTN_NUM_HEADS = 4
SELF_ATTN_DROPOUT = 0.1
SELF_ATTN_NORM_FIRST = True

POI_EMB_DIM = 128
CAT_EMB_DIM = 32
USER_EMB_DIM: Optional[int] = 256
EMBEDDING_DROPOUT = 0.4
OUTPUT_DROPOUT = 0.3

TOP_K_LIST = [1, 5, 10, 20]

TORCH_DTYPE = torch.float32
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

TEST_ON_EVERY_VAL_IMPROVEMENT = True

# Optional checkpoint IO (default: disabled)
SAVE_MODEL_DIR = None  # when set, save best checkpoint to this directory
LOAD_MODEL_PATH = None  # when set, load checkpoint before training
CHECKPOINT_PATH = None  # when set, save best checkpoint to this exact path

PAD_IDX = 0
MISSING_CAT_TOKEN = "<MISSING_CAT>"

SECONDS_PER_DAY = 24 * 60 * 60

# GeoCellEmbedding (enabled by default; can be disabled via CLI for ablation)
USE_GEO_CELL_EMBEDDING = True
GEO_CELL_SIZES_M_DEFAULT = [500, 1000, 2000]
GEO_CELL_SIZES_M: Optional[List[int]] = None
GEO_CELL_EMB_DIM = 64

# POI embedding contrastive learning (CL applies only to poi_embedding)
USE_POI_EMBEDDING_CONTRASTIVE_LEARNING = True
TOP_K_CANDIDATES_FOR_POI_EMBEDDING_CONTRASTIVE_LEARNING = 200
POI_EMBEDDING_CONTRASTIVE_LEARNING_FORCE_LABEL_INTO_CANDIDATES_STRATEGY = "replace_lowest"
POI_EMBEDDING_CONTRASTIVE_LEARNING_PROJ_DIM = 128
USE_MLP_FOR_CL_INSTEAD_OF_SIMPLE_PROJ = False
POI_EMBEDDING_CONTRASTIVE_LEARNING_TEMPERATURE = 0.07
POI_EMBEDDING_CONTRASTIVE_LEARNING_NORMALIZE_EMBEDDINGS = True
POI_EMBEDDING_CONTRASTIVE_LEARNING_LOSS_WEIGHT = 1.0
POI_ECL_NEGATIVE_SOURCE = "topk"  # choices: topk/random/mix

EARTH_RADIUS_M = 6_371_000.0
DEG2RAD = math.pi / 180.0

# ST-Prior ReRanker
USE_ST_PRIOR_RE_RANKER = True
ST_PRIOR_CANDIDATE_K = 200
ST_PRIOR_TIME_BINS = 24
ST_PRIOR_ALPHA = 1.0
ST_PRIOR_USER_BIN_MIN_COUNT = 5
ST_PRIOR_SIGMA_MIN_KM = 0.5
ST_PRIOR_TAU_KM = -1.0  # <=0 means "auto (train p75)"
ST_PRIOR_TAU_SAMPLE_CAP = 200_000
ST_PRIOR_LAMBDA_TIME = 0.3
ST_PRIOR_LAMBDA_USER = 0.5
ST_PRIOR_LAMBDA_DIST = 0.5


def _now_sydney_timestamp() -> str:
    try:
        from zoneinfo import ZoneInfo

        return datetime.datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y%m%d_%H%M%S")
    except Exception:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _metric_tag(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        s = f"{float(value):.{digits}f}"
    except Exception:
        return "NA"
    return s.replace(".", "p")


def _dataset_tag_from_path(data_path: str) -> str:
    base = os.path.basename(str(data_path))
    if base.endswith(".pkl"):
        base = base[:-4]
    return base.replace(" ", "_")


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    if isinstance(v, str):
        val = v.strip().lower()
        if val in {"true", "yes", "1"}:
            return True
        if val in {"false", "no", "0"}:
            return False
    raise argparse.ArgumentTypeError("Expected a boolean-like value (true/false/yes/no/1/0)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train window-input SelfAttn for next-POI prediction.")

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # window size
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None, help="Alias of --window_size (if set, overrides --window_size).")

    # model flags
    parser.add_argument("--use_cat_emb", type=_str2bool, default=None)
    parser.add_argument("--use_user_emb", type=_str2bool, default=None, help="Whether to use user_id embedding.")
    parser.add_argument("--use_positional_encoding", type=_str2bool, default=None)
    parser.add_argument("--use_tod_slot_embedding", type=_str2bool, default=None)
    parser.add_argument("--use_geo_cell_embedding", type=_str2bool, default=None)
    parser.add_argument(
        "--classifier_token_position",
        type=str,
        choices=["first", "last", "mean"],
        default=None,
        help="Which token position to use for classification: first (user token), last (last valid token), mean (avg of unmasked tokens).",
    )
    parser.add_argument(
        "--classifier_token_strategy",
        type=str,
        choices=["first", "last", "mean"],
        default=None,
        help="Deprecated alias of --classifier_token_position.",
    )
    parser.add_argument(
        "--output_token_strategy",
        type=str,
        choices=["first", "last", "mean"],
        default=None,
        help="Deprecated alias of --classifier_token_position.",
    )

    # dims
    parser.add_argument("--poi_emb_dim", type=int, default=None)
    parser.add_argument("--cat_emb_dim", type=int, default=None)
    parser.add_argument("--user_emb_dim", type=int, default=None, help="User embedding dim before projection (default: self_attn_d_model).")
    parser.add_argument("--tod_slot_emb_dim", type=int, default=None)
    parser.add_argument("--geo_cell_emb_dim", type=int, default=None)
    parser.add_argument(
        "--tod_slot_scales",
        type=int,
        nargs="+",
        default=None,
        help="Only used when --use_tod_slot_embedding is true. Example: --tod_slot_scales 6 24 48",
    )
    parser.add_argument(
        "--geo_cell_sizes_m",
        type=int,
        nargs="+",
        default=None,
        help="Only used when --use_geo_cell_embedding is true. Example: --geo_cell_sizes_m 500 2000",
    )
    parser.add_argument("--self_attn_d_model", type=int, default=None)
    parser.add_argument("--self_attn_num_layers", type=int, default=None)
    parser.add_argument("--self_attn_num_heads", type=int, default=None)
    parser.add_argument("--self_attn_dropout", type=float, default=None)
    parser.add_argument("--norm_first", type=_str2bool, default=None)
    parser.add_argument("--embedding_dropout", type=float, default=None)
    parser.add_argument("--output_dropout", type=float, default=None)

    # runtime
    parser.add_argument("--device", type=str, default=None, help="e.g. 0/1/2 for GPU index or 'cpu'/'cuda:0'")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default=None)
    parser.add_argument("--num_workers", type=int, default=0)

    # checkpoint
    parser.add_argument("--save_model_dir", type=str, default=None)
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--test_on_every_val_improvement", type=_str2bool, default=None)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "When set, save the best (by val MRR) checkpoint to this exact path "
            "(for downstream hyperparameter search scripts)."
        ),
    )

    # ST-Prior ReRanker
    parser.add_argument("--use_st_prior_re_ranker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--st_prior_candidate_k", type=int, default=None)
    parser.add_argument("--st_prior_time_bins", type=int, default=None)
    parser.add_argument("--st_prior_alpha", type=float, default=None)
    parser.add_argument("--st_prior_user_bin_min_count", type=int, default=None)
    parser.add_argument("--st_prior_sigma_min_km", type=float, default=None)
    parser.add_argument("--st_prior_tau_km", type=float, default=None, help="<=0 means auto (train p75).")
    parser.add_argument("--st_prior_tau_sample_cap", type=int, default=None)
    parser.add_argument("--st_prior_lambda_time", type=float, default=None)
    parser.add_argument("--st_prior_lambda_user", type=float, default=None)
    parser.add_argument("--st_prior_lambda_dist", type=float, default=None)

    # POI embedding contrastive learning
    parser.add_argument("--use_poi_embedding_contrastive_learning", type=_str2bool, default=None)
    parser.add_argument("--top_k_candidates_for_poi_embedding_contrastive_learning", type=int, default=None)
    parser.add_argument(
        "--poi_embedding_contrastive_learning_force_label_into_candidates_strategy",
        type=str,
        choices=["replace_lowest", "replace_highest"],
        default=None,
    )
    parser.add_argument("--poi_embedding_contrastive_learning_proj_dim", type=int, default=None)
    parser.add_argument("--use_mlp_for_cl_instead_of_simple_proj", type=_str2bool, default=None)
    parser.add_argument("--poi_embedding_contrastive_learning_temperature", type=float, default=None)
    parser.add_argument("--poi_embedding_contrastive_learning_normalize_embeddings", type=_str2bool, default=None)
    parser.add_argument("--poi_embedding_contrastive_learning_loss_weight", type=float, default=None)
    parser.add_argument("--poi_ecl_negative_source", type=str, choices=["topk", "random", "mix"], default=None)

    return parser.parse_args()


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if not device_arg:
        return DEVICE
    s = str(device_arg).strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("cuda:"):
        return torch.device(s)
    if s.isdigit():
        return torch.device(f"cuda:{int(s)}")
    # fallback
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sanitize_cat_id(cat_id) -> str:
    if cat_id is None:
        return MISSING_CAT_TOKEN
    s = str(cat_id).strip()
    if not s:
        return MISSING_CAT_TOKEN
    return s


def build_vocab(data: Dict) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Build vocab for user/poi/cat.
    """
    user_ids = set()
    poi_ids = set()
    cat_ids = {MISSING_CAT_TOKEN}

    for split_name in ("train", "val", "test"):
        split = data.get(split_name, None)
        if not split:
            continue
        user_ids.update(split.keys())
        for seq in split.values():
            for rec in seq:
                poi = rec.get("poi_id", None)
                if poi is not None:
                    poi_ids.add(str(poi))
                cat_ids.add(_sanitize_cat_id(rec.get("cat_id", None)))

    def _to_idx(ids: set) -> Dict[str, int]:
        # Reserve PAD_IDX=0, start real tokens from 1.
        return {token: i + 1 for i, token in enumerate(sorted(ids))}

    user2idx = _to_idx(user_ids)
    poi2idx = _to_idx(poi_ids)
    cat2idx = _to_idx(cat_ids)
    return user2idx, poi2idx, cat2idx


@dataclass(frozen=True)
class _SeqTensors:
    poi: torch.Tensor  # (L,) long
    cat: torch.Tensor  # (L,) long
    lat: torch.Tensor  # (L,) float
    lon: torch.Tensor  # (L,) float
    tod: torch.Tensor  # (L,) float in [0,1]
    utc_time: torch.Tensor  # (L,) long (UTC epoch seconds; if missing in data, filled with 0)
    tz_offset_min: torch.Tensor  # (L,) long (minutes; if missing in data, filled with 0)


@dataclass(frozen=True)
class GeoCellMeta:
    """
    Use a dataset-specific local equirectangular projection:

      x_m = R * cos(lat0) * (lon_rad - lon0)
      y_m = R * (lat_rad - lat0)

    Then bucket (x_m, y_m) into multi-scale grids (cell_sizes_m).
    """

    cell_sizes_m: List[int]
    lat0_rad: float
    lon0_rad: float
    cos_lat0: float
    x_min_m: float
    y_min_m: float
    grid_w_list: List[int]
    grid_h_list: List[int]


def _build_geo_cell_meta(train_user_seqs: Dict[int, _SeqTensors], cell_sizes_m: List[int]) -> GeoCellMeta:
    cell_sizes_m = [int(x) for x in list(cell_sizes_m)]
    if not cell_sizes_m:
        raise ValueError("cell_sizes_m must be non-empty when use_geo_cell_embedding=True")
    for s in cell_sizes_m:
        if s <= 0:
            raise ValueError(f"Invalid cell_size_m: {s}")

    lats: List[torch.Tensor] = []
    lons: List[torch.Tensor] = []
    for seq in train_user_seqs.values():
        if seq.lat.numel() == 0:
            continue
        lats.append(seq.lat.to(dtype=torch.float32))
        lons.append(seq.lon.to(dtype=torch.float32))
    if not lats:
        raise ValueError("Cannot build geo_cell_meta: empty train lat/lon tensors.")

    lat_all = torch.cat(lats, dim=0)
    lon_all = torch.cat(lons, dim=0)
    lat_rad_all = lat_all * float(DEG2RAD)
    lon_rad_all = lon_all * float(DEG2RAD)

    lat0_rad = float(lat_rad_all.mean().item())
    lon0_rad = float(lon_rad_all.mean().item())
    cos_lat0 = float(math.cos(lat0_rad))

    x_m_all = float(EARTH_RADIUS_M) * cos_lat0 * (lon_rad_all - float(lon0_rad))
    y_m_all = float(EARTH_RADIUS_M) * (lat_rad_all - float(lat0_rad))

    x_min_m = float(x_m_all.min().item())
    x_max_m = float(x_m_all.max().item())
    y_min_m = float(y_m_all.min().item())
    y_max_m = float(y_m_all.max().item())

    grid_w_list: List[int] = []
    grid_h_list: List[int] = []
    for cell_size_m in cell_sizes_m:
        dx = max(0.0, float(x_max_m - x_min_m))
        dy = max(0.0, float(y_max_m - y_min_m))
        grid_w = int(math.floor(dx / float(cell_size_m))) + 1
        grid_h = int(math.floor(dy / float(cell_size_m))) + 1
        grid_w_list.append(max(1, int(grid_w)))
        grid_h_list.append(max(1, int(grid_h)))

    return GeoCellMeta(
        cell_sizes_m=cell_sizes_m,
        lat0_rad=lat0_rad,
        lon0_rad=lon0_rad,
        cos_lat0=cos_lat0,
        x_min_m=x_min_m,
        y_min_m=y_min_m,
        grid_w_list=grid_w_list,
        grid_h_list=grid_h_list,
    )


def _precompute_geo_cell_ids_by_user(
    user_seqs: Dict[int, _SeqTensors], meta: GeoCellMeta
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Precompute per-checkin geo cell bucket ids for each user sequence.

    Returns:
        geo_x_ids_by_user[u]: (L, num_scales) long
        geo_y_ids_by_user[u]: (L, num_scales) long
    """
    num_scales = len(meta.cell_sizes_m)
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for u_idx, seq in user_seqs.items():
        if seq.lat.numel() == 0:
            empty = torch.empty((0, num_scales), dtype=torch.long)
            out[int(u_idx)] = (empty, empty)
            continue

        lat_rad = seq.lat.to(dtype=torch.float32) * float(DEG2RAD)
        lon_rad = seq.lon.to(dtype=torch.float32) * float(DEG2RAD)
        x_m = float(EARTH_RADIUS_M) * float(meta.cos_lat0) * (lon_rad - float(meta.lon0_rad))
        y_m = float(EARTH_RADIUS_M) * (lat_rad - float(meta.lat0_rad))
        x_rel = x_m - float(meta.x_min_m)
        y_rel = y_m - float(meta.y_min_m)

        x_ids_by_scale: List[torch.Tensor] = []
        y_ids_by_scale: List[torch.Tensor] = []
        for cell_size_m, grid_w, grid_h in zip(meta.cell_sizes_m, meta.grid_w_list, meta.grid_h_list):
            cs = float(cell_size_m)
            x_id = torch.floor(x_rel / cs).to(dtype=torch.long) + 1
            y_id = torch.floor(y_rel / cs).to(dtype=torch.long) + 1
            x_id = x_id.clamp(min=1, max=int(grid_w))
            y_id = y_id.clamp(min=1, max=int(grid_h))
            x_ids_by_scale.append(x_id)
            y_ids_by_scale.append(y_id)

        out[int(u_idx)] = (torch.stack(x_ids_by_scale, dim=-1), torch.stack(y_ids_by_scale, dim=-1))
    return out


def _seq_to_tensors(
    seq: List[Dict],
    *,
    poi2idx: Dict[str, int],
    cat2idx: Dict[str, int],
    dtype: torch.dtype,
) -> _SeqTensors:
    poi_idx: List[int] = []
    cat_idx: List[int] = []
    lat_list: List[float] = []
    lon_list: List[float] = []
    tod_list: List[float] = []
    utc_time_list: List[int] = []
    tz_offset_min_list: List[int] = []

    for rec in seq:
        poi = str(rec["poi_id"])
        cat = _sanitize_cat_id(rec.get("cat_id", None))
        poi_idx.append(int(poi2idx[poi]))
        cat_idx.append(int(cat2idx[cat]))
        lat_list.append(float(rec["lat"]))
        lon_list.append(float(rec["lon"]))
        tod_list.append(float(rec.get("tod", 0.0)))
        if "utc_time" in rec and rec["utc_time"] is not None:
            utc_time_list.append(int(rec["utc_time"]))
        elif "timestamp" in rec and rec["timestamp"] is not None:
            utc_time_list.append(int(float(rec["timestamp"])))
        else:
            utc_time_list.append(0)
        tz_offset_min_list.append(int(rec.get("timezone_offset", 0) or 0))

    return _SeqTensors(
        poi=torch.tensor(poi_idx, dtype=torch.long),
        cat=torch.tensor(cat_idx, dtype=torch.long),
        lat=torch.tensor(lat_list, dtype=dtype),
        lon=torch.tensor(lon_list, dtype=dtype),
        tod=torch.tensor(tod_list, dtype=dtype),
        utc_time=torch.tensor(utc_time_list, dtype=torch.long),
        tz_offset_min=torch.tensor(tz_offset_min_list, dtype=torch.long),
    )


def build_split_tensors(
    split: Dict[str, List[Dict]],
    *,
    user2idx: Dict[str, int],
    poi2idx: Dict[str, int],
    cat2idx: Dict[str, int],
    dtype: torch.dtype,
) -> Dict[int, _SeqTensors]:
    out: Dict[int, _SeqTensors] = {}
    for user_id, seq in split.items():
        u_idx = user2idx.get(user_id, None)
        if u_idx is None:
            continue
        if not seq:
            continue
        out[int(u_idx)] = _seq_to_tensors(seq, poi2idx=poi2idx, cat2idx=cat2idx, dtype=dtype)
    return out


def _concat_seq_tensors(parts: List[_SeqTensors]) -> _SeqTensors:
    if not parts:
        return _SeqTensors(
            poi=torch.empty((0,), dtype=torch.long),
            cat=torch.empty((0,), dtype=torch.long),
            lat=torch.empty((0,), dtype=TORCH_DTYPE),
            lon=torch.empty((0,), dtype=TORCH_DTYPE),
            tod=torch.empty((0,), dtype=TORCH_DTYPE),
            utc_time=torch.empty((0,), dtype=torch.long),
            tz_offset_min=torch.empty((0,), dtype=torch.long),
        )
    poi = torch.cat([p.poi for p in parts], dim=0)
    cat = torch.cat([p.cat for p in parts], dim=0)
    lat = torch.cat([p.lat for p in parts], dim=0)
    lon = torch.cat([p.lon for p in parts], dim=0)
    tod = torch.cat([p.tod for p in parts], dim=0)
    utc_time = torch.cat([p.utc_time for p in parts], dim=0)
    tz_offset_min = torch.cat([p.tz_offset_min for p in parts], dim=0)
    return _SeqTensors(poi=poi, cat=cat, lat=lat, lon=lon, tod=tod, utc_time=utc_time, tz_offset_min=tz_offset_min)


class WindowInputDataset(Dataset):
    """
    Each sample corresponds to one labeled check-in (with poi as the target):
    - Input: the k check-ins before that check-in (left padding)
        - Final version is fixed to take the most recent k check-ins (i.e., [pos-k, pos))
    - Output: classifier_token_position determines which token position is used to produce logits (user token by default)
    """

    def __init__(
        self,
        user_seqs: Dict[int, _SeqTensors],
        label_ranges: Dict[int, Tuple[int, int]],
        *,
        window_size: int,
        use_cat_emb: bool,
        use_tod_slot_embedding: bool,
        tod_slot_scales: Optional[List[int]],
        use_geo_cell_embedding: bool,
        geo_cell_ids_by_user: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        geo_cell_num_scales: Optional[int] = None,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.user_seqs = user_seqs
        self.window_size = int(window_size)
        self.use_cat_emb = bool(use_cat_emb)
        self.use_tod_slot_embedding = bool(use_tod_slot_embedding)
        self.tod_slot_scales = [int(x) for x in (tod_slot_scales or [])]
        if self.use_tod_slot_embedding:
            if not self.tod_slot_scales:
                raise ValueError("tod_slot_scales must be non-empty when use_tod_slot_embedding=True")
            for slots_per_day in self.tod_slot_scales:
                if slots_per_day <= 0:
                    raise ValueError(f"Invalid slots_per_day in tod_slot_scales: {slots_per_day}")
        self.use_geo_cell_embedding = bool(use_geo_cell_embedding)
        self.geo_cell_ids_by_user = geo_cell_ids_by_user
        self.geo_cell_num_scales = None if geo_cell_num_scales is None else int(geo_cell_num_scales)
        if self.use_geo_cell_embedding:
            if self.geo_cell_ids_by_user is None:
                raise ValueError("geo_cell_ids_by_user is required when use_geo_cell_embedding=True")
            if self.geo_cell_num_scales is None or self.geo_cell_num_scales <= 0:
                raise ValueError("geo_cell_num_scales must be provided and >0 when use_geo_cell_embedding=True")
        self.pad_idx = int(pad_idx)

        samples: List[Tuple[int, int]] = []
        for u_idx, (start, end) in label_ranges.items():
            start_i = int(start)
            end_i = int(end)
            if end_i <= start_i:
                continue
            samples.extend((int(u_idx), pos) for pos in range(start_i, end_i))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        u_idx, pos = self.samples[index]
        seq = self.user_seqs[u_idx]
        target = seq.poi[pos]
        query_tod = seq.tod[pos]
        if pos > 0:
            last_lat = seq.lat[pos - 1]
            last_lon = seq.lon[pos - 1]
            has_last = torch.tensor(True, dtype=torch.bool)
        else:
            last_lat = torch.zeros((), dtype=seq.lat.dtype)
            last_lon = torch.zeros((), dtype=seq.lon.dtype)
            has_last = torch.tensor(False, dtype=torch.bool)

        if self.window_size == 0:
            window_poi = torch.empty((0,), dtype=torch.long)
            window_cat = torch.empty((0,), dtype=torch.long)
            window_utc_time = torch.empty((0,), dtype=torch.long)
            window_tz_offset_min = torch.empty((0,), dtype=torch.long)
            if self.use_geo_cell_embedding:
                num_scales = int(self.geo_cell_num_scales or 0)
                window_geo_cell_x_ids = torch.empty((0, num_scales), dtype=torch.long)
                window_geo_cell_y_ids = torch.empty((0, num_scales), dtype=torch.long)
        else:
            end = int(pos)
            # recent window: [pos-k, pos)
            start = max(0, end - self.window_size)
            idx_list = list(range(start, end))

            hist_len = len(idx_list)
            pad_len = self.window_size - hist_len
            if idx_list:
                idx = torch.tensor(idx_list, dtype=torch.long)
                hist_poi = seq.poi[idx]
                hist_cat = seq.cat[idx]
                hist_utc_time = seq.utc_time[idx]
                hist_tz_offset_min = seq.tz_offset_min[idx]
                if self.use_geo_cell_embedding:
                    if self.geo_cell_ids_by_user is None:
                        raise RuntimeError("Internal error: geo_cell_ids_by_user is None")
                    x_all, y_all = self.geo_cell_ids_by_user[int(u_idx)]
                    hist_geo_x = x_all[idx]
                    hist_geo_y = y_all[idx]
            else:
                hist_poi = torch.empty((0,), dtype=torch.long)
                hist_cat = torch.empty((0,), dtype=torch.long)
                hist_utc_time = torch.empty((0,), dtype=torch.long)
                hist_tz_offset_min = torch.empty((0,), dtype=torch.long)
                if self.use_geo_cell_embedding:
                    num_scales = int(self.geo_cell_num_scales or 0)
                    hist_geo_x = torch.empty((0, num_scales), dtype=torch.long)
                    hist_geo_y = torch.empty((0, num_scales), dtype=torch.long)

            if pad_len > 0:
                pad_poi = torch.full((pad_len,), fill_value=self.pad_idx, dtype=torch.long)
                pad_cat = torch.full((pad_len,), fill_value=self.pad_idx, dtype=torch.long)
                pad_utc_time = torch.zeros((pad_len,), dtype=torch.long)
                pad_tz_offset_min = torch.zeros((pad_len,), dtype=torch.long)
                window_poi = torch.cat([pad_poi, hist_poi], dim=0)
                window_cat = torch.cat([pad_cat, hist_cat], dim=0)
                window_utc_time = torch.cat([pad_utc_time, hist_utc_time], dim=0)
                window_tz_offset_min = torch.cat([pad_tz_offset_min, hist_tz_offset_min], dim=0)
                if self.use_geo_cell_embedding:
                    num_scales = int(self.geo_cell_num_scales or 0)
                    pad_geo = torch.zeros((pad_len, num_scales), dtype=torch.long)
                    window_geo_cell_x_ids = torch.cat([pad_geo, hist_geo_x], dim=0)
                    window_geo_cell_y_ids = torch.cat([pad_geo, hist_geo_y], dim=0)
            else:
                window_poi = hist_poi
                window_cat = hist_cat
                window_utc_time = hist_utc_time
                window_tz_offset_min = hist_tz_offset_min
                if self.use_geo_cell_embedding:
                    window_geo_cell_x_ids = hist_geo_x
                    window_geo_cell_y_ids = hist_geo_y

        out: Dict[str, torch.Tensor] = {
            "user": torch.tensor(u_idx, dtype=torch.long),
            "window_poi": window_poi,
            "target_poi": target.to(dtype=torch.long),
            # ST-Prior reranker query fields (independent of model feature flags)
            "query_tod": query_tod.to(dtype=seq.tod.dtype),
            "last_lat": last_lat.to(dtype=seq.lat.dtype),
            "last_lon": last_lon.to(dtype=seq.lon.dtype),
            "has_last": has_last,
        }
        if self.use_cat_emb:
            out["window_cat"] = window_cat
        if self.use_tod_slot_embedding:
            if window_utc_time.numel() != self.window_size or window_tz_offset_min.numel() != self.window_size:
                raise RuntimeError("Internal error: window time tensors must match window_size.")
            # local_ts = utc_time + timezone_offset*60; tod_sec = local_ts mod 86400
            local_ts = window_utc_time + window_tz_offset_min * 60
            tod_sec = torch.remainder(local_ts, SECONDS_PER_DAY).to(dtype=torch.long)

            slot_ids_by_scale: List[torch.Tensor] = []
            for slots_per_day in self.tod_slot_scales:
                slots_per_day_int = int(slots_per_day)
                slot = (tod_sec * slots_per_day_int) // SECONDS_PER_DAY  # [0, slots_per_day-1]
                slot_id = slot + 1  # reserve 0 for PAD
                slot_ids_by_scale.append(slot_id.to(dtype=torch.long))
            window_tod_slot_ids = torch.stack(slot_ids_by_scale, dim=-1)  # (k, num_scales)

            pad_mask = window_poi.eq(int(self.pad_idx)).unsqueeze(-1)
            window_tod_slot_ids = window_tod_slot_ids.masked_fill(pad_mask, 0)
            out["window_tod_slot_ids"] = window_tod_slot_ids
        if self.use_geo_cell_embedding:
            if window_geo_cell_x_ids.size(0) != self.window_size or window_geo_cell_y_ids.size(0) != self.window_size:
                raise RuntimeError("Internal error: window geo cell tensors must match window_size.")
            out["window_geo_cell_x_ids"] = window_geo_cell_x_ids
            out["window_geo_cell_y_ids"] = window_geo_cell_y_ids
        return out


def update_metrics(logits: torch.Tensor, targets: torch.Tensor, top_k_list: List[int], metrics: Dict[str, float]) -> None:
    """
    logits: (N, num_pois)
    targets: (N,)
    """
    with torch.no_grad():
        max_k = max(top_k_list)
        _, topk_indices = torch.topk(logits, k=max_k, dim=1)
        correct = topk_indices.eq(targets.unsqueeze(1))
        for k in top_k_list:
            metrics[f"Acc@{k}"] += correct[:, :k].any(dim=1).float().sum().item()

        true_scores = logits.gather(1, targets.unsqueeze(1))
        higher = (logits > true_scores).sum(dim=1).float()
        ranks = higher + 1.0
        metrics["MRR"] += (1.0 / ranks).sum().item()
        metrics["count"] += targets.size(0)


def update_metrics_reranked_candidates(
    candidate_idx: torch.Tensor,  # (N, K) long, POI indices
    candidate_scores: torch.Tensor,  # (N, K) float, rescored within candidates
    targets: torch.Tensor,  # (N,) long
    top_k_list: List[int],
    metrics: Dict[str, float],
) -> None:
    with torch.no_grad():
        n, k = candidate_idx.shape
        max_k = min(int(max(top_k_list)), int(k))
        _, rel = torch.topk(candidate_scores, k=max_k, dim=1)
        top_pois = candidate_idx.gather(1, rel)
        correct = top_pois.eq(targets.unsqueeze(1))
        for kk in top_k_list:
            kkk = min(int(kk), int(max_k))
            metrics[f"Acc@{kk}"] += correct[:, :kkk].any(dim=1).float().sum().item()

        mask = candidate_idx.eq(targets.unsqueeze(1))
        in_set = mask.any(dim=1)
        tgt_score = (candidate_scores * mask.to(dtype=candidate_scores.dtype)).sum(dim=1)
        higher = (candidate_scores > tgt_score.unsqueeze(1)).sum(dim=1).float()
        rank = higher + 1.0
        mrr = torch.where(in_set, 1.0 / rank, torch.zeros_like(rank))
        metrics["MRR"] += mrr.sum().item()
        metrics["count"] += targets.size(0)


def compute_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    count = max(int(metrics.get("count", 0)), 1)
    avg: Dict[str, float] = {}
    for k, v in metrics.items():
        if k == "count":
            continue
        avg[k] = float(v) / float(count)
    return avg


def format_metrics(metrics: Dict[str, float], top_k_list: List[int]) -> str:
    ordered_keys = [f"Acc@{k}" for k in top_k_list] + ["MRR"]
    parts = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    extra_keys = sorted(k for k in metrics.keys() if k not in ordered_keys)
    for key in extra_keys:
        parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


def _maybe_load_checkpoint(model: nn.Module, optimizer: Optional[optim.Optimizer], device: torch.device) -> None:
    if not LOAD_MODEL_PATH:
        return
    ckpt_path = str(LOAD_MODEL_PATH)
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)}")

    model_state = ckpt.get("model_state_dict", None)
    if model_state is None:
        raise KeyError("Checkpoint must contain 'model_state_dict'")
    model.load_state_dict(model_state)

    opt_state = ckpt.get("optimizer_state_dict", None)
    if optimizer is not None and opt_state is not None:
        optimizer.load_state_dict(opt_state)
        print("Loaded optimizer_state_dict (resume training).")


@torch.no_grad()
def evaluate(
    model: NextPOIWindowSelfAttn,
    data_loader: DataLoader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    top_k_list: List[int],
    st_prior_tables: Optional[STPriorTables] = None,
    st_prior_candidate_k: int = 20,
    st_prior_lambda_time: float = 0.3,
    st_prior_lambda_user: float = 0.5,
    st_prior_lambda_dist: float = 0.5,
    st_prior_sigma_min_km: float = 0.5,
    st_prior_tau_km: float = -1.0,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, float] = {f"Acc@{k}": 0.0 for k in top_k_list}
    metrics["MRR"] = 0.0
    metrics["count"] = 0.0

    for batch in data_loader:
        user = batch["user"].to(device=device, non_blocking=True)
        window_poi = batch["window_poi"].to(device=device, non_blocking=True)
        window_cat = batch.get("window_cat", None)
        if window_cat is not None:
            window_cat = window_cat.to(device=device, non_blocking=True)
        window_tod_slot_ids = batch.get("window_tod_slot_ids", None)
        if window_tod_slot_ids is not None:
            window_tod_slot_ids = window_tod_slot_ids.to(device=device, non_blocking=True)
        window_geo_cell_x_ids = batch.get("window_geo_cell_x_ids", None)
        window_geo_cell_y_ids = batch.get("window_geo_cell_y_ids", None)
        if window_geo_cell_x_ids is not None:
            window_geo_cell_x_ids = window_geo_cell_x_ids.to(device=device, non_blocking=True)
        if window_geo_cell_y_ids is not None:
            window_geo_cell_y_ids = window_geo_cell_y_ids.to(device=device, non_blocking=True)
        targets = batch["target_poi"].to(device=device, non_blocking=True)

        logits = model(
            user,
            window_poi,
            window_cat,
            window_tod_slot_ids=window_tod_slot_ids,
            window_geo_cell_x_ids=window_geo_cell_x_ids,
            window_geo_cell_y_ids=window_geo_cell_y_ids,
        )
        if st_prior_tables is None:
            update_metrics(logits, targets, top_k_list, metrics)
        else:
            query_tod = batch["query_tod"].to(device=device, dtype=dtype, non_blocking=True)
            last_lat = batch["last_lat"].to(device=device, dtype=dtype, non_blocking=True)
            last_lon = batch["last_lon"].to(device=device, dtype=dtype, non_blocking=True)
            has_last = batch["has_last"].to(device=device, non_blocking=True)
            tau = float(st_prior_tau_km) if float(st_prior_tau_km) > 0.0 else float(st_prior_tables.tau_p75_km)

            cand_idx, cand_scores = rerank_topk(
                logits=logits,
                user_idx=user,
                query_tod=query_tod,
                last_lat=last_lat,
                last_lon=last_lon,
                has_last=has_last,
                tables=st_prior_tables,
                candidate_k=int(st_prior_candidate_k),
                lambda_time=float(st_prior_lambda_time),
                lambda_user=float(st_prior_lambda_user),
                lambda_dist=float(st_prior_lambda_dist),
                sigma_min_km=float(st_prior_sigma_min_km),
                tau_km=float(tau),
            )
            update_metrics_reranked_candidates(cand_idx, cand_scores, targets, top_k_list, metrics)

    return compute_metrics(metrics)


def main() -> None:
    global DATA_PATH
    global WINDOW_SIZE
    global NUM_EPOCHS
    global BATCH_SIZE
    global EVAL_BATCH_SIZE
    global LR
    global WEIGHT_DECAY
    global USE_CAT_EMBEDDING
    global USE_USER_EMBEDDING
    global USE_POSITIONAL_ENCODING
    global USE_TOD_SLOT_EMBEDDING
    global TOD_SLOT_SCALES
    global TOD_SLOT_EMB_DIM
    global SELF_ATTN_D_MODEL
    global SELF_ATTN_NUM_LAYERS
    global SELF_ATTN_NUM_HEADS
    global SELF_ATTN_DROPOUT
    global SELF_ATTN_NORM_FIRST
    global POI_EMB_DIM
    global CAT_EMB_DIM
    global USER_EMB_DIM
    global EMBEDDING_DROPOUT
    global OUTPUT_DROPOUT
    global TORCH_DTYPE
    global DEVICE
    global SAVE_MODEL_DIR
    global LOAD_MODEL_PATH
    global CHECKPOINT_PATH
    global TEST_ON_EVERY_VAL_IMPROVEMENT
    global CLASSIFIER_TOKEN_POSITION
    global USE_ST_PRIOR_RE_RANKER
    global ST_PRIOR_CANDIDATE_K
    global ST_PRIOR_TIME_BINS
    global ST_PRIOR_ALPHA
    global ST_PRIOR_USER_BIN_MIN_COUNT
    global ST_PRIOR_SIGMA_MIN_KM
    global ST_PRIOR_TAU_KM
    global ST_PRIOR_TAU_SAMPLE_CAP
    global ST_PRIOR_LAMBDA_TIME
    global ST_PRIOR_LAMBDA_USER
    global ST_PRIOR_LAMBDA_DIST
    global USE_GEO_CELL_EMBEDDING
    global GEO_CELL_SIZES_M
    global GEO_CELL_EMB_DIM
    global USE_POI_EMBEDDING_CONTRASTIVE_LEARNING
    global TOP_K_CANDIDATES_FOR_POI_EMBEDDING_CONTRASTIVE_LEARNING
    global POI_EMBEDDING_CONTRASTIVE_LEARNING_FORCE_LABEL_INTO_CANDIDATES_STRATEGY
    global POI_EMBEDDING_CONTRASTIVE_LEARNING_PROJ_DIM
    global USE_MLP_FOR_CL_INSTEAD_OF_SIMPLE_PROJ
    global POI_EMBEDDING_CONTRASTIVE_LEARNING_TEMPERATURE
    global POI_EMBEDDING_CONTRASTIVE_LEARNING_NORMALIZE_EMBEDDINGS
    global POI_EMBEDDING_CONTRASTIVE_LEARNING_LOSS_WEIGHT
    global POI_ECL_NEGATIVE_SOURCE

    args = parse_args()

    if args.data_path is not None:
        DATA_PATH = str(args.data_path)
    if args.epochs is not None:
        NUM_EPOCHS = int(args.epochs)
    if args.batch_size is not None:
        BATCH_SIZE = int(args.batch_size)
    if args.eval_batch_size is not None:
        EVAL_BATCH_SIZE = int(args.eval_batch_size)
    if args.lr is not None:
        LR = float(args.lr)
    if args.weight_decay is not None:
        WEIGHT_DECAY = float(args.weight_decay)

    if args.window_size is not None:
        WINDOW_SIZE = int(args.window_size)
    if args.k is not None:
        WINDOW_SIZE = int(args.k)

    if args.use_cat_emb is not None:
        USE_CAT_EMBEDDING = bool(args.use_cat_emb)
    if args.use_user_emb is not None:
        USE_USER_EMBEDDING = bool(args.use_user_emb)
    if args.use_positional_encoding is not None:
        USE_POSITIONAL_ENCODING = bool(args.use_positional_encoding)
    if args.use_tod_slot_embedding is not None:
        USE_TOD_SLOT_EMBEDDING = bool(args.use_tod_slot_embedding)
    if args.use_geo_cell_embedding is not None:
        USE_GEO_CELL_EMBEDDING = bool(args.use_geo_cell_embedding)
    if args.classifier_token_position is not None and args.classifier_token_strategy is not None:
        raise ValueError("Use either --classifier_token_position or --classifier_token_strategy, not both.")
    if args.classifier_token_position is not None and args.output_token_strategy is not None:
        raise ValueError("Use either --classifier_token_position or --output_token_strategy, not both.")
    if args.classifier_token_strategy is not None and args.output_token_strategy is not None:
        raise ValueError("Use either --classifier_token_strategy or --output_token_strategy, not both.")
    if args.classifier_token_position is not None:
        CLASSIFIER_TOKEN_POSITION = str(args.classifier_token_position).strip().lower()
    elif args.classifier_token_strategy is not None:
        CLASSIFIER_TOKEN_POSITION = str(args.classifier_token_strategy).strip().lower()
    elif args.output_token_strategy is not None:
        CLASSIFIER_TOKEN_POSITION = str(args.output_token_strategy).strip().lower()

    if args.poi_emb_dim is not None:
        POI_EMB_DIM = int(args.poi_emb_dim)
    if args.cat_emb_dim is not None:
        CAT_EMB_DIM = int(args.cat_emb_dim)
    if args.user_emb_dim is not None:
        USER_EMB_DIM = int(args.user_emb_dim)
    if args.tod_slot_emb_dim is not None:
        TOD_SLOT_EMB_DIM = int(args.tod_slot_emb_dim)
    if args.geo_cell_emb_dim is not None:
        GEO_CELL_EMB_DIM = int(args.geo_cell_emb_dim)
    if args.self_attn_d_model is not None:
        SELF_ATTN_D_MODEL = int(args.self_attn_d_model)
    if args.self_attn_num_layers is not None:
        SELF_ATTN_NUM_LAYERS = int(args.self_attn_num_layers)
    if args.self_attn_num_heads is not None:
        SELF_ATTN_NUM_HEADS = int(args.self_attn_num_heads)
    if args.self_attn_dropout is not None:
        SELF_ATTN_DROPOUT = float(args.self_attn_dropout)
    if args.norm_first is not None:
        SELF_ATTN_NORM_FIRST = bool(args.norm_first)
    if args.embedding_dropout is not None:
        EMBEDDING_DROPOUT = float(args.embedding_dropout)
    if args.output_dropout is not None:
        OUTPUT_DROPOUT = float(args.output_dropout)

    if USE_TOD_SLOT_EMBEDDING:
        resolved_scales = [int(x) for x in (args.tod_slot_scales or TOD_SLOT_SCALES_DEFAULT)]
        if not resolved_scales:
            raise ValueError("Resolved tod_slot_scales is empty while use_tod_slot_embedding=True")
        TOD_SLOT_SCALES = resolved_scales

    if USE_GEO_CELL_EMBEDDING:
        resolved_geo_sizes = [int(x) for x in (args.geo_cell_sizes_m or GEO_CELL_SIZES_M_DEFAULT)]
        if not resolved_geo_sizes:
            raise ValueError("Resolved geo_cell_sizes_m is empty while use_geo_cell_embedding=True")
        GEO_CELL_SIZES_M = resolved_geo_sizes
    else:
        GEO_CELL_SIZES_M = None

    if args.use_poi_embedding_contrastive_learning is not None:
        USE_POI_EMBEDDING_CONTRASTIVE_LEARNING = bool(args.use_poi_embedding_contrastive_learning)
    if args.top_k_candidates_for_poi_embedding_contrastive_learning is not None:
        TOP_K_CANDIDATES_FOR_POI_EMBEDDING_CONTRASTIVE_LEARNING = int(args.top_k_candidates_for_poi_embedding_contrastive_learning)
    if args.poi_embedding_contrastive_learning_force_label_into_candidates_strategy is not None:
        POI_EMBEDDING_CONTRASTIVE_LEARNING_FORCE_LABEL_INTO_CANDIDATES_STRATEGY = str(
            args.poi_embedding_contrastive_learning_force_label_into_candidates_strategy
        ).strip()
    if args.poi_embedding_contrastive_learning_proj_dim is not None:
        POI_EMBEDDING_CONTRASTIVE_LEARNING_PROJ_DIM = int(args.poi_embedding_contrastive_learning_proj_dim)
    if args.use_mlp_for_cl_instead_of_simple_proj is not None:
        USE_MLP_FOR_CL_INSTEAD_OF_SIMPLE_PROJ = bool(args.use_mlp_for_cl_instead_of_simple_proj)
    if args.poi_embedding_contrastive_learning_temperature is not None:
        POI_EMBEDDING_CONTRASTIVE_LEARNING_TEMPERATURE = float(args.poi_embedding_contrastive_learning_temperature)
    if args.poi_embedding_contrastive_learning_normalize_embeddings is not None:
        POI_EMBEDDING_CONTRASTIVE_LEARNING_NORMALIZE_EMBEDDINGS = bool(args.poi_embedding_contrastive_learning_normalize_embeddings)
    if args.poi_embedding_contrastive_learning_loss_weight is not None:
        POI_EMBEDDING_CONTRASTIVE_LEARNING_LOSS_WEIGHT = float(args.poi_embedding_contrastive_learning_loss_weight)
    if args.poi_ecl_negative_source is not None:
        POI_ECL_NEGATIVE_SOURCE = str(args.poi_ecl_negative_source).strip().lower()

    if args.dtype is not None:
        TORCH_DTYPE = torch.bfloat16 if str(args.dtype).lower() == "bf16" else torch.float32
    DEVICE = _resolve_device(args.device)

    if args.save_model_dir is not None:
        SAVE_MODEL_DIR = str(args.save_model_dir)
    if args.load_model_path is not None:
        LOAD_MODEL_PATH = str(args.load_model_path)
    if args.test_on_every_val_improvement is not None:
        TEST_ON_EVERY_VAL_IMPROVEMENT = bool(args.test_on_every_val_improvement)
    if args.checkpoint_path is not None:
        CHECKPOINT_PATH = str(args.checkpoint_path)

    USE_ST_PRIOR_RE_RANKER = bool(getattr(args, "use_st_prior_re_ranker", False))
    if args.st_prior_candidate_k is not None:
        ST_PRIOR_CANDIDATE_K = int(args.st_prior_candidate_k)
    if args.st_prior_time_bins is not None:
        ST_PRIOR_TIME_BINS = int(args.st_prior_time_bins)
    if args.st_prior_alpha is not None:
        ST_PRIOR_ALPHA = float(args.st_prior_alpha)
    if args.st_prior_user_bin_min_count is not None:
        ST_PRIOR_USER_BIN_MIN_COUNT = int(args.st_prior_user_bin_min_count)
    if args.st_prior_sigma_min_km is not None:
        ST_PRIOR_SIGMA_MIN_KM = float(args.st_prior_sigma_min_km)
    if args.st_prior_tau_km is not None:
        ST_PRIOR_TAU_KM = float(args.st_prior_tau_km)
    if args.st_prior_tau_sample_cap is not None:
        ST_PRIOR_TAU_SAMPLE_CAP = int(args.st_prior_tau_sample_cap)
    if args.st_prior_lambda_time is not None:
        ST_PRIOR_LAMBDA_TIME = float(args.st_prior_lambda_time)
    if args.st_prior_lambda_user is not None:
        ST_PRIOR_LAMBDA_USER = float(args.st_prior_lambda_user)
    if args.st_prior_lambda_dist is not None:
        ST_PRIOR_LAMBDA_DIST = float(args.st_prior_lambda_dist)

    set_seed(args.seed)

    print(f"DATA_PATH={DATA_PATH}")
    print(f"WINDOW_SIZE={WINDOW_SIZE}")
    print(f"WINDOW_SAMPLING={WINDOW_SAMPLING}")
    print(f"DEVICE={DEVICE} DTYPE={TORCH_DTYPE}")
    print(
        f"d_model={SELF_ATTN_D_MODEL} heads={SELF_ATTN_NUM_HEADS} layers={SELF_ATTN_NUM_LAYERS} dropout={SELF_ATTN_DROPOUT} norm_first={SELF_ATTN_NORM_FIRST}"
    )
    print(
        f"poi_emb_dim={POI_EMB_DIM} cat_emb_dim={CAT_EMB_DIM} user_emb_dim={(SELF_ATTN_D_MODEL if USER_EMB_DIM is None else USER_EMB_DIM)} "
        f"use_cat_emb={USE_CAT_EMBEDDING} use_user_emb={USE_USER_EMBEDDING} use_positional_encoding={USE_POSITIONAL_ENCODING}"
    )
    print(
        f"use_tod_slot_embedding={USE_TOD_SLOT_EMBEDDING} tod_slot_scales={(TOD_SLOT_SCALES or [])} tod_slot_emb_dim={TOD_SLOT_EMB_DIM}"
    )
    print(
        f"use_geo_cell_embedding={USE_GEO_CELL_EMBEDDING} geo_cell_sizes_m={(GEO_CELL_SIZES_M or [])} geo_cell_emb_dim={GEO_CELL_EMB_DIM}"
    )
    print(
        "poi_ecl: "
        f"use={USE_POI_EMBEDDING_CONTRASTIVE_LEARNING} "
        f"top_k={TOP_K_CANDIDATES_FOR_POI_EMBEDDING_CONTRASTIVE_LEARNING} "
        f"force_label_strategy={POI_EMBEDDING_CONTRASTIVE_LEARNING_FORCE_LABEL_INTO_CANDIDATES_STRATEGY} "
        f"proj_dim={POI_EMBEDDING_CONTRASTIVE_LEARNING_PROJ_DIM} "
        f"use_mlp_proj={USE_MLP_FOR_CL_INSTEAD_OF_SIMPLE_PROJ} "
        f"temperature={POI_EMBEDDING_CONTRASTIVE_LEARNING_TEMPERATURE} "
        f"normalize={POI_EMBEDDING_CONTRASTIVE_LEARNING_NORMALIZE_EMBEDDINGS} "
        f"loss_weight={POI_EMBEDDING_CONTRASTIVE_LEARNING_LOSS_WEIGHT} "
        f"negative_source={POI_ECL_NEGATIVE_SOURCE}"
    )
    print(f"classifier_token_position={CLASSIFIER_TOKEN_POSITION}")
    print(f"batch_size={BATCH_SIZE} eval_batch_size={EVAL_BATCH_SIZE} lr={LR} wd={WEIGHT_DECAY}")
    print(
        f"use_st_prior_re_ranker={USE_ST_PRIOR_RE_RANKER} candidate_k={ST_PRIOR_CANDIDATE_K} "
        f"time_bins={ST_PRIOR_TIME_BINS} alpha={ST_PRIOR_ALPHA} "
        f"lambda_time={ST_PRIOR_LAMBDA_TIME} lambda_user={ST_PRIOR_LAMBDA_USER} lambda_dist={ST_PRIOR_LAMBDA_DIST} "
        f"sigma_min_km={ST_PRIOR_SIGMA_MIN_KM} tau_km={ST_PRIOR_TAU_KM}"
    )

    # load pkl
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    train_data = data["train"]
    val_data = data["val"]
    test_data = data["test"]

    # vocab
    user2idx, poi2idx, cat2idx = build_vocab(data)
    num_users = len(user2idx) + 1  # + pad row
    num_pois = len(poi2idx) + 1
    num_cats = len(cat2idx) + 1
    print(f"#users={num_users} #pois={num_pois} #cats={num_cats} (pad_idx=0)")

    st_prior_tables: Optional[STPriorTables] = None
    if USE_ST_PRIOR_RE_RANKER:
        print("Building ST-Prior tables from train split ...")
        st_prior_tables = build_st_prior_tables(
            train_split=train_data,
            user2idx=user2idx,
            poi2idx=poi2idx,
            time_bins=ST_PRIOR_TIME_BINS,
            alpha=ST_PRIOR_ALPHA,
            user_bin_min_count=ST_PRIOR_USER_BIN_MIN_COUNT,
            seed=int(args.seed),
            tau_km=ST_PRIOR_TAU_KM,
            tau_sample_cap=ST_PRIOR_TAU_SAMPLE_CAP,
        )
        st_prior_tables.to(device=DEVICE, dtype=TORCH_DTYPE)
        print(f"ST-Prior built: tau_p75_km={st_prior_tables.tau_p75_km:.2f}")

    # split tensors
    train_tensors = build_split_tensors(train_data, user2idx=user2idx, poi2idx=poi2idx, cat2idx=cat2idx, dtype=TORCH_DTYPE)
    val_tensors = build_split_tensors(val_data, user2idx=user2idx, poi2idx=poi2idx, cat2idx=cat2idx, dtype=TORCH_DTYPE)
    test_tensors = build_split_tensors(test_data, user2idx=user2idx, poi2idx=poi2idx, cat2idx=cat2idx, dtype=TORCH_DTYPE)

    # build per-user label ranges
    all_users = set(train_tensors.keys()) | set(val_tensors.keys()) | set(test_tensors.keys())

    train_label_ranges: Dict[int, Tuple[int, int]] = {}
    train_combined: Dict[int, _SeqTensors] = {}

    val_label_ranges: Dict[int, Tuple[int, int]] = {}
    val_combined: Dict[int, _SeqTensors] = {}

    test_label_ranges: Dict[int, Tuple[int, int]] = {}
    test_combined: Dict[int, _SeqTensors] = {}

    for u in sorted(all_users):
        tr = train_tensors.get(u, None)
        va = val_tensors.get(u, None)
        te = test_tensors.get(u, None)

        # train: only train seq
        if tr is not None:
            train_combined[u] = tr
            train_label_ranges[u] = (0, int(tr.poi.size(0)))

        # val: train + val
        parts_tv = [p for p in [tr, va] if p is not None]
        if parts_tv:
            combined_tv = _concat_seq_tensors(parts_tv)
            val_combined[u] = combined_tv
            tr_len = 0 if tr is None else int(tr.poi.size(0))
            va_len = 0 if va is None else int(va.poi.size(0))
            if va_len > 0:
                val_label_ranges[u] = (tr_len, tr_len + va_len)

        # test: train + val + test
        parts_tvt = [p for p in [tr, va, te] if p is not None]
        if parts_tvt:
            combined_tvt = _concat_seq_tensors(parts_tvt)
            test_combined[u] = combined_tvt
            tr_len = 0 if tr is None else int(tr.poi.size(0))
            va_len = 0 if va is None else int(va.poi.size(0))
            te_len = 0 if te is None else int(te.poi.size(0))
            if te_len > 0:
                start = tr_len + va_len
                test_label_ranges[u] = (start, start + te_len)

    geo_cell_meta: Optional[GeoCellMeta] = None
    geo_train_ids: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    geo_val_ids: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    geo_test_ids: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    if USE_GEO_CELL_EMBEDDING:
        geo_cell_meta = _build_geo_cell_meta(train_combined, list(GEO_CELL_SIZES_M or []))
        geo_train_ids = _precompute_geo_cell_ids_by_user(train_combined, geo_cell_meta)
        geo_val_ids = _precompute_geo_cell_ids_by_user(val_combined, geo_cell_meta)
        geo_test_ids = _precompute_geo_cell_ids_by_user(test_combined, geo_cell_meta)

    train_ds = WindowInputDataset(
        train_combined,
        train_label_ranges,
        window_size=WINDOW_SIZE,
        use_cat_emb=USE_CAT_EMBEDDING,
        use_tod_slot_embedding=USE_TOD_SLOT_EMBEDDING,
        tod_slot_scales=TOD_SLOT_SCALES,
        use_geo_cell_embedding=USE_GEO_CELL_EMBEDDING,
        geo_cell_ids_by_user=geo_train_ids,
        geo_cell_num_scales=(None if geo_cell_meta is None else len(geo_cell_meta.cell_sizes_m)),
        pad_idx=PAD_IDX,
    )
    val_ds = WindowInputDataset(
        val_combined,
        val_label_ranges,
        window_size=WINDOW_SIZE,
        use_cat_emb=USE_CAT_EMBEDDING,
        use_tod_slot_embedding=USE_TOD_SLOT_EMBEDDING,
        tod_slot_scales=TOD_SLOT_SCALES,
        use_geo_cell_embedding=USE_GEO_CELL_EMBEDDING,
        geo_cell_ids_by_user=geo_val_ids,
        geo_cell_num_scales=(None if geo_cell_meta is None else len(geo_cell_meta.cell_sizes_m)),
        pad_idx=PAD_IDX,
    )
    test_ds = WindowInputDataset(
        test_combined,
        test_label_ranges,
        window_size=WINDOW_SIZE,
        use_cat_emb=USE_CAT_EMBEDDING,
        use_tod_slot_embedding=USE_TOD_SLOT_EMBEDDING,
        tod_slot_scales=TOD_SLOT_SCALES,
        use_geo_cell_embedding=USE_GEO_CELL_EMBEDDING,
        geo_cell_ids_by_user=geo_test_ids,
        geo_cell_num_scales=(None if geo_cell_meta is None else len(geo_cell_meta.cell_sizes_m)),
        pad_idx=PAD_IDX,
    )

    print(f"#train_samples={len(train_ds)} #val_samples={len(val_ds)} #test_samples={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = NextPOIWindowSelfAttn(
        num_users=num_users,
        num_pois=num_pois,
        num_cats=num_cats,
        window_size=WINDOW_SIZE,
        d_model=SELF_ATTN_D_MODEL,
        nhead=SELF_ATTN_NUM_HEADS,
        num_layers=SELF_ATTN_NUM_LAYERS,
        dropout=SELF_ATTN_DROPOUT,
        embedding_dropout=EMBEDDING_DROPOUT,
        output_dropout=OUTPUT_DROPOUT,
        norm_first=SELF_ATTN_NORM_FIRST,
        poi_emb_dim=POI_EMB_DIM,
        cat_emb_dim=CAT_EMB_DIM,
        user_emb_dim=USER_EMB_DIM,
        use_poi_embedding_contrastive_learning=USE_POI_EMBEDDING_CONTRASTIVE_LEARNING,
        poi_embedding_contrastive_learning_proj_dim=POI_EMBEDDING_CONTRASTIVE_LEARNING_PROJ_DIM,
        use_mlp_for_cl_instead_of_simple_proj=USE_MLP_FOR_CL_INSTEAD_OF_SIMPLE_PROJ,
        use_tod_slot_embedding=USE_TOD_SLOT_EMBEDDING,
        tod_slot_scales=TOD_SLOT_SCALES,
        tod_slot_emb_dim=TOD_SLOT_EMB_DIM,
        use_geo_cell_embedding=USE_GEO_CELL_EMBEDDING,
        geo_cell_sizes_m=(None if geo_cell_meta is None else geo_cell_meta.cell_sizes_m),
        geo_cell_grid_w=(None if geo_cell_meta is None else geo_cell_meta.grid_w_list),
        geo_cell_grid_h=(None if geo_cell_meta is None else geo_cell_meta.grid_h_list),
        geo_cell_emb_dim=GEO_CELL_EMB_DIM,
        use_user_emb=USE_USER_EMBEDDING,
        use_cat_emb=USE_CAT_EMBEDDING,
        use_positional_encoding=USE_POSITIONAL_ENCODING,
        classifier_token_position=CLASSIFIER_TOKEN_POSITION,
        pad_idx=PAD_IDX,
    ).to(device=DEVICE, dtype=TORCH_DTYPE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    _maybe_load_checkpoint(model=model, optimizer=optimizer, device=DEVICE)

    best_val_mrr = -1.0
    best_val_epoch: Optional[int] = None
    best_model_state: Optional[Dict] = None
    last_test_metrics: Optional[Dict[str, float]] = None
    last_test_epoch: Optional[int] = None

    train_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        total_ce_loss = 0.0
        total_cl_loss = 0.0
        total_missing_label = 0.0

        for batch in train_loader:
            user = batch["user"].to(device=DEVICE, non_blocking=True)
            window_poi = batch["window_poi"].to(device=DEVICE, non_blocking=True)
            window_cat = batch.get("window_cat", None)
            if window_cat is not None:
                window_cat = window_cat.to(device=DEVICE, non_blocking=True)
            window_tod_slot_ids = batch.get("window_tod_slot_ids", None)
            if window_tod_slot_ids is not None:
                window_tod_slot_ids = window_tod_slot_ids.to(device=DEVICE, non_blocking=True)
            window_geo_cell_x_ids = batch.get("window_geo_cell_x_ids", None)
            window_geo_cell_y_ids = batch.get("window_geo_cell_y_ids", None)
            if window_geo_cell_x_ids is not None:
                window_geo_cell_x_ids = window_geo_cell_x_ids.to(device=DEVICE, non_blocking=True)
            if window_geo_cell_y_ids is not None:
                window_geo_cell_y_ids = window_geo_cell_y_ids.to(device=DEVICE, non_blocking=True)
            targets = batch["target_poi"].to(device=DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, user_out = model(
                user,
                window_poi,
                window_cat,
                window_tod_slot_ids=window_tod_slot_ids,
                window_geo_cell_x_ids=window_geo_cell_x_ids,
                window_geo_cell_y_ids=window_geo_cell_y_ids,
                return_user_out=True,
            )
            ce_loss = criterion(logits, targets)
            loss = ce_loss
            if USE_POI_EMBEDDING_CONTRASTIVE_LEARNING:
                cl_loss, cl_stats = model.compute_poi_embedding_contrastive_learning_loss(
                    user_out=user_out,
                    logits_poi=logits,
                    target_poi=targets,
                    top_k_candidates=int(TOP_K_CANDIDATES_FOR_POI_EMBEDDING_CONTRASTIVE_LEARNING),
                    temperature=float(POI_EMBEDDING_CONTRASTIVE_LEARNING_TEMPERATURE),
                    normalize_embeddings=bool(POI_EMBEDDING_CONTRASTIVE_LEARNING_NORMALIZE_EMBEDDINGS),
                    force_label_into_candidates_strategy=str(POI_EMBEDDING_CONTRASTIVE_LEARNING_FORCE_LABEL_INTO_CANDIDATES_STRATEGY),
                    negative_source=str(POI_ECL_NEGATIVE_SOURCE),
                )
                loss = loss + float(POI_EMBEDDING_CONTRASTIVE_LEARNING_LOSS_WEIGHT) * cl_loss
                total_cl_loss += float(cl_loss.detach().item()) * targets.size(0)
                total_missing_label += float(cl_stats.get("missing_label_rate", 0.0)) * targets.size(0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_ce_loss += float(ce_loss.detach().item()) * targets.size(0)
            total_count += targets.size(0)

        denom = max(total_count, 1)
        avg_train_loss = total_loss / denom
        avg_ce_loss = total_ce_loss / denom
        if USE_POI_EMBEDDING_CONTRASTIVE_LEARNING:
            avg_cl_loss = total_cl_loss / denom
            avg_missing = total_missing_label / denom
            print(
                f"[Epoch {epoch}] train loss: {avg_train_loss:.4f} (ce={avg_ce_loss:.4f} cl={avg_cl_loss:.4f} "
                f"missing_label_rate={avg_missing:.4f})"
            )
        else:
            print(f"[Epoch {epoch}] train loss: {avg_train_loss:.4f} (ce={avg_ce_loss:.4f})")

        val_metrics = evaluate(
            model,
            val_loader,
            device=DEVICE,
            dtype=TORCH_DTYPE,
            top_k_list=TOP_K_LIST,
            st_prior_tables=st_prior_tables,
            st_prior_candidate_k=ST_PRIOR_CANDIDATE_K,
            st_prior_lambda_time=ST_PRIOR_LAMBDA_TIME,
            st_prior_lambda_user=ST_PRIOR_LAMBDA_USER,
            st_prior_lambda_dist=ST_PRIOR_LAMBDA_DIST,
            st_prior_sigma_min_km=ST_PRIOR_SIGMA_MIN_KM,
            st_prior_tau_km=ST_PRIOR_TAU_KM,
        )
        val_mrr = float(val_metrics.get("MRR", 0.0))
        print(f"[Epoch {epoch}] val: {format_metrics(val_metrics, TOP_K_LIST)}")

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_val_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            if TEST_ON_EVERY_VAL_IMPROVEMENT:
                print(f"New best val MRR: {best_val_mrr:.4f}, evaluating on test...")
                test_metrics = evaluate(
                    model,
                    test_loader,
                    device=DEVICE,
                    dtype=TORCH_DTYPE,
                    top_k_list=TOP_K_LIST,
                    st_prior_tables=st_prior_tables,
                    st_prior_candidate_k=ST_PRIOR_CANDIDATE_K,
                    st_prior_lambda_time=ST_PRIOR_LAMBDA_TIME,
                    st_prior_lambda_user=ST_PRIOR_LAMBDA_USER,
                    st_prior_lambda_dist=ST_PRIOR_LAMBDA_DIST,
                    st_prior_sigma_min_km=ST_PRIOR_SIGMA_MIN_KM,
                    st_prior_tau_km=ST_PRIOR_TAU_KM,
                )
                print(f"[Epoch {epoch}] test: {format_metrics(test_metrics, TOP_K_LIST)}")
                last_test_metrics = test_metrics
                last_test_epoch = epoch
            else:
                print(f"New best val MRR: {best_val_mrr:.4f}, test deferred to end.")

    if not TEST_ON_EVERY_VAL_IMPROVEMENT and best_model_state is not None:
        model.load_state_dict(best_model_state)
        last_test_epoch = best_val_epoch
        last_test_metrics = evaluate(
            model,
            test_loader,
            device=DEVICE,
            dtype=TORCH_DTYPE,
            top_k_list=TOP_K_LIST,
            st_prior_tables=st_prior_tables,
            st_prior_candidate_k=ST_PRIOR_CANDIDATE_K,
            st_prior_lambda_time=ST_PRIOR_LAMBDA_TIME,
            st_prior_lambda_user=ST_PRIOR_LAMBDA_USER,
            st_prior_lambda_dist=ST_PRIOR_LAMBDA_DIST,
            st_prior_sigma_min_km=ST_PRIOR_SIGMA_MIN_KM,
            st_prior_tau_km=ST_PRIOR_TAU_KM,
        )

    elapsed = time.time() - train_start
    final_test_str = format_metrics(last_test_metrics or {}, TOP_K_LIST)
    print(f"Final test (best epoch {last_test_epoch}, elapsed {elapsed:.2f}s): {final_test_str}")

    if SAVE_MODEL_DIR:
        os.makedirs(str(SAVE_MODEL_DIR), exist_ok=True)
        dataset_tag = _dataset_tag_from_path(DATA_PATH)
        timestamp = _now_sydney_timestamp()
        test_acc1 = None if not last_test_metrics else last_test_metrics.get("Acc@1", None)
        test_mrr = None if not last_test_metrics else last_test_metrics.get("MRR", None)
        val_mrr_tag = _metric_tag(best_val_mrr)
        test_acc1_tag = _metric_tag(test_acc1)
        test_mrr_tag = _metric_tag(test_mrr)

        filename = (
            f"self_attn_window_{dataset_tag}_{timestamp}"
            f"_k{WINDOW_SIZE}_valMRR{val_mrr_tag}_testAcc1{test_acc1_tag}_testMRR{test_mrr_tag}.pkl"
        )
        save_path = os.path.join(str(SAVE_MODEL_DIR), filename)
        payload = {
            "created_at_sydney": timestamp,
            "data_path": DATA_PATH,
            "model_name": "NextPOIWindowSelfAttn",
            "metrics": {
                "best_val_epoch": best_val_epoch,
                "best_val_mrr": best_val_mrr,
                "best_test_epoch": last_test_epoch,
                "test": last_test_metrics,
            },
            "hparams": {
                "window_size": WINDOW_SIZE,
                "window_sampling": WINDOW_SAMPLING,
                "poi_emb_dim": POI_EMB_DIM,
                "cat_emb_dim": CAT_EMB_DIM,
                "user_emb_dim": (SELF_ATTN_D_MODEL if USER_EMB_DIM is None else USER_EMB_DIM),
                "use_cat_emb": USE_CAT_EMBEDDING,
                "use_user_emb": USE_USER_EMBEDDING,
                "use_positional_encoding": USE_POSITIONAL_ENCODING,
                "use_tod_slot_embedding": USE_TOD_SLOT_EMBEDDING,
                "tod_slot_scales": TOD_SLOT_SCALES,
                "tod_slot_emb_dim": TOD_SLOT_EMB_DIM,
                "use_geo_cell_embedding": USE_GEO_CELL_EMBEDDING,
                "geo_cell_sizes_m": GEO_CELL_SIZES_M,
                "geo_cell_emb_dim": GEO_CELL_EMB_DIM,
                "use_poi_embedding_contrastive_learning": USE_POI_EMBEDDING_CONTRASTIVE_LEARNING,
                "top_k_candidates_for_poi_embedding_contrastive_learning": TOP_K_CANDIDATES_FOR_POI_EMBEDDING_CONTRASTIVE_LEARNING,
                "poi_embedding_contrastive_learning_force_label_into_candidates_strategy": POI_EMBEDDING_CONTRASTIVE_LEARNING_FORCE_LABEL_INTO_CANDIDATES_STRATEGY,
                "poi_embedding_contrastive_learning_proj_dim": POI_EMBEDDING_CONTRASTIVE_LEARNING_PROJ_DIM,
                "use_mlp_for_cl_instead_of_simple_proj": USE_MLP_FOR_CL_INSTEAD_OF_SIMPLE_PROJ,
                "poi_embedding_contrastive_learning_temperature": POI_EMBEDDING_CONTRASTIVE_LEARNING_TEMPERATURE,
                "poi_embedding_contrastive_learning_normalize_embeddings": POI_EMBEDDING_CONTRASTIVE_LEARNING_NORMALIZE_EMBEDDINGS,
                "poi_embedding_contrastive_learning_loss_weight": POI_EMBEDDING_CONTRASTIVE_LEARNING_LOSS_WEIGHT,
                "poi_ecl_negative_source": POI_ECL_NEGATIVE_SOURCE,
                "classifier_token_position": CLASSIFIER_TOKEN_POSITION,
                "self_attn_d_model": SELF_ATTN_D_MODEL,
                "self_attn_num_layers": SELF_ATTN_NUM_LAYERS,
                "self_attn_num_heads": SELF_ATTN_NUM_HEADS,
                "self_attn_dropout": SELF_ATTN_DROPOUT,
                "self_attn_norm_first": SELF_ATTN_NORM_FIRST,
                "embedding_dropout": EMBEDDING_DROPOUT,
                "output_dropout": OUTPUT_DROPOUT,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "dtype": str(TORCH_DTYPE),
            },
            "model_state_dict": best_model_state if best_model_state is not None else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_sizes": {"num_users": num_users, "num_pois": num_pois, "num_cats": num_cats, "pad_idx": PAD_IDX},
        }
        torch.save(payload, save_path)
        print(f"Saved checkpoint to {save_path}")

    if CHECKPOINT_PATH:
        ckpt_path = str(CHECKPOINT_PATH)
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        timestamp = _now_sydney_timestamp()
        payload = {
            "created_at_sydney": timestamp,
            "data_path": DATA_PATH,
            "model_name": "NextPOIWindowSelfAttn",
            "metrics": {
                "best_val_epoch": best_val_epoch,
                "best_val_mrr": best_val_mrr,
                "best_test_epoch": last_test_epoch,
                "test": last_test_metrics,
            },
            "hparams": {
                "window_size": WINDOW_SIZE,
                "window_sampling": WINDOW_SAMPLING,
                "poi_emb_dim": POI_EMB_DIM,
                "cat_emb_dim": CAT_EMB_DIM,
                "user_emb_dim": (SELF_ATTN_D_MODEL if USER_EMB_DIM is None else USER_EMB_DIM),
                "use_cat_emb": USE_CAT_EMBEDDING,
                "use_user_emb": USE_USER_EMBEDDING,
                "use_positional_encoding": USE_POSITIONAL_ENCODING,
                "use_tod_slot_embedding": USE_TOD_SLOT_EMBEDDING,
                "tod_slot_scales": TOD_SLOT_SCALES,
                "tod_slot_emb_dim": TOD_SLOT_EMB_DIM,
                "use_geo_cell_embedding": USE_GEO_CELL_EMBEDDING,
                "geo_cell_sizes_m": GEO_CELL_SIZES_M,
                "geo_cell_emb_dim": GEO_CELL_EMB_DIM,
                "use_poi_embedding_contrastive_learning": USE_POI_EMBEDDING_CONTRASTIVE_LEARNING,
                "top_k_candidates_for_poi_embedding_contrastive_learning": TOP_K_CANDIDATES_FOR_POI_EMBEDDING_CONTRASTIVE_LEARNING,
                "poi_embedding_contrastive_learning_force_label_into_candidates_strategy": POI_EMBEDDING_CONTRASTIVE_LEARNING_FORCE_LABEL_INTO_CANDIDATES_STRATEGY,
                "poi_embedding_contrastive_learning_proj_dim": POI_EMBEDDING_CONTRASTIVE_LEARNING_PROJ_DIM,
                "use_mlp_for_cl_instead_of_simple_proj": USE_MLP_FOR_CL_INSTEAD_OF_SIMPLE_PROJ,
                "poi_embedding_contrastive_learning_temperature": POI_EMBEDDING_CONTRASTIVE_LEARNING_TEMPERATURE,
                "poi_embedding_contrastive_learning_normalize_embeddings": POI_EMBEDDING_CONTRASTIVE_LEARNING_NORMALIZE_EMBEDDINGS,
                "poi_embedding_contrastive_learning_loss_weight": POI_EMBEDDING_CONTRASTIVE_LEARNING_LOSS_WEIGHT,
                "poi_ecl_negative_source": POI_ECL_NEGATIVE_SOURCE,
                "classifier_token_position": CLASSIFIER_TOKEN_POSITION,
                "self_attn_d_model": SELF_ATTN_D_MODEL,
                "self_attn_num_layers": SELF_ATTN_NUM_LAYERS,
                "self_attn_num_heads": SELF_ATTN_NUM_HEADS,
                "self_attn_dropout": SELF_ATTN_DROPOUT,
                "self_attn_norm_first": SELF_ATTN_NORM_FIRST,
                "embedding_dropout": EMBEDDING_DROPOUT,
                "output_dropout": OUTPUT_DROPOUT,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "dtype": str(TORCH_DTYPE),
                "use_st_prior_re_ranker": USE_ST_PRIOR_RE_RANKER,
                "st_prior_candidate_k": ST_PRIOR_CANDIDATE_K,
                "st_prior_time_bins": ST_PRIOR_TIME_BINS,
                "st_prior_alpha": ST_PRIOR_ALPHA,
                "st_prior_user_bin_min_count": ST_PRIOR_USER_BIN_MIN_COUNT,
                "st_prior_sigma_min_km": ST_PRIOR_SIGMA_MIN_KM,
                "st_prior_tau_km": ST_PRIOR_TAU_KM,
                "st_prior_tau_sample_cap": ST_PRIOR_TAU_SAMPLE_CAP,
                "st_prior_lambda_time": ST_PRIOR_LAMBDA_TIME,
                "st_prior_lambda_user": ST_PRIOR_LAMBDA_USER,
                "st_prior_lambda_dist": ST_PRIOR_LAMBDA_DIST,
            },
            "model_state_dict": best_model_state if best_model_state is not None else model.state_dict(),
            "vocab_sizes": {"num_users": num_users, "num_pois": num_pois, "num_cats": num_cats, "pad_idx": PAD_IDX},
        }
        torch.save(payload, ckpt_path)
        print(f"Saved best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
