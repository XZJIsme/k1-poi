import argparse
import datetime
import importlib.util
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def parse_args() -> argparse.Namespace:
    def str2bool(v):
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

    parser = argparse.ArgumentParser(
        description=(
            "Train backbone (with --use_st_prior_re_ranker enabled) and then run large-scale "
            "hyperparameter search for ST-Prior reranker on val (test is used once at the end). "
            "This variant supports batched (vectorized) stage-1 trials for higher throughput."
        )
    )

    parser.add_argument("--exp_id", type=str, default="self_attn_st_prior_reranker_hparam_search_parallel_trials")
    parser.add_argument(
        "--data_path_list",
        type=str,
        nargs="+",
        default=[
            "data/processed/TKY_excluding_cold.pkl",
            "data/processed/NYC_excluding_cold.pkl",
            "data/processed/CA_excluding_cold.pkl",
        ],
    )
    parser.add_argument("--window_size_list", type=str, default="1", help="Comma-separated window sizes, e.g. 1,2,3")

    # training (forwarded to trainer)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default=None)
    parser.add_argument("--norm_first", type=str2bool, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_cat_emb", type=str2bool, default=False)
    parser.add_argument("--use_user_emb", type=str2bool, default=True)
    parser.add_argument("--use_positional_encoding", type=str2bool, default=True)
    parser.add_argument("--use_tod_slot_embedding", type=str2bool, default=True)
    parser.add_argument("--use_geo_cell_embedding", type=str2bool, default=True)
    parser.add_argument("--classifier_token_position", type=str, choices=["first", "last", "mean"], default=None)

    parser.add_argument("--poi_emb_dim", type=int, default=128)
    parser.add_argument("--cat_emb_dim", type=int, default=None)
    parser.add_argument("--user_emb_dim", type=int, default=256)
    parser.add_argument("--tod_slot_emb_dim", type=int, default=64)
    parser.add_argument("--geo_cell_emb_dim", type=int, default=64)
    # POI embedding contrastive learning (forwarded to trainer)
    parser.add_argument("--use_poi_embedding_contrastive_learning", type=str2bool, default=True)
    parser.add_argument("--top_k_candidates_for_poi_embedding_contrastive_learning", type=int, default=200)
    parser.add_argument(
        "--poi_embedding_contrastive_learning_force_label_into_candidates_strategy",
        type=str,
        choices=["replace_lowest", "replace_highest"],
        default="replace_lowest",
    )
    parser.add_argument("--poi_embedding_contrastive_learning_proj_dim", type=int, default=128)
    parser.add_argument("--use_mlp_for_cl_instead_of_simple_proj", type=str2bool, default=False)
    parser.add_argument("--poi_embedding_contrastive_learning_temperature", type=float, default=0.07)
    parser.add_argument("--poi_embedding_contrastive_learning_normalize_embeddings", type=str2bool, default=True)
    parser.add_argument("--poi_embedding_contrastive_learning_loss_weight", type=float, default=1.0)
    parser.add_argument("--poi_ecl_negative_source", type=str, choices=["topk", "random", "mix"], default="topk")
    parser.add_argument(
        "--tod_slot_scales",
        type=int,
        nargs="+",
        default=[6, 12, 24],
        help="Only used when --use_tod_slot_embedding is true. Example: --tod_slot_scales 6 24 48",
    )
    parser.add_argument(
        "--geo_cell_sizes_m",
        type=int,
        nargs="+",
        default=[500, 1000, 2000],
        help="Only used when --use_geo_cell_embedding is true. Example: --geo_cell_sizes_m 500 2000",
    )
    parser.add_argument("--self_attn_d_model", type=int, default=128)
    parser.add_argument("--self_attn_num_layers", type=int, default=2)
    parser.add_argument("--self_attn_num_heads", type=int, default=4)
    parser.add_argument("--self_attn_dropout", type=float, default=None)
    parser.add_argument("--embedding_dropout", type=float, default=None)
    parser.add_argument("--output_dropout", type=float, default=None)

    # devices for training scheduling
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="Comma-separated GPU indices for parallel training.")
    parser.add_argument("--search_device", type=str, default=None, help="Device for search evaluation (default: cuda:<first>).")
    parser.add_argument(
        "--just_run_all_runs_together",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If set, launch all training tasks immediately and distribute them across --devices round-robin. "
            "This allows multiple concurrent training processes per GPU (use with care: can be slow/OOM)."
        ),
    )

    # reranker fixed structure
    parser.add_argument("--st_prior_candidate_k", type=int, default=200)
    parser.add_argument("--st_prior_time_bins", type=int, default=24)
    parser.add_argument("--st_prior_alpha", type=float, default=1.0)
    parser.add_argument("--st_prior_user_bin_min_count", type=int, default=5)
    parser.add_argument("--st_prior_tau_sample_cap", type=int, default=200_000)
    parser.add_argument(
        "--st_prior_score_terms",
        type=str,
        nargs="+",
        choices=["time", "user", "dist", "cl_sim"],
        default=["time", "user", "dist", "cl_sim"],
        help=(
            "Which rerank terms to add to base logits. Base logits are always included."
        ),
    )

    # search budget
    parser.add_argument("--search_trials", type=int, default=20000, help="Random trials on val_tune (stage-1).")
    parser.add_argument("--search_eval_samples", type=int, default=20000, help="Max #val samples used in stage-1.")
    parser.add_argument("--search_top_m", type=int, default=400, help="Keep top-M from stage-1 for full-val eval.")
    parser.add_argument(
        "--st_prior_num_parallel_trials",
        type=int,
        default=20,
        help=(
            "How many random trials to evaluate together in stage-1 (vectorized across trials). "
            "Set to e.g. 10/20 to better utilize the GPU/CPU during search. "
            "Note: stage-2 (top-M full-val eval) is still evaluated serially."
        ),
    )

    # search ranges
    parser.add_argument("--lambda_min", type=float, default=0.0)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    # NOTE: keep defaults aligned with the GeoCellEmbedding (non-CL) version.
    parser.add_argument("--sigma_min_choices_km", type=str, default="0.3,0.5,1.0")
    parser.add_argument("--tau_mult_choices", type=str, default="0.5,1.0,2.0", help="Multiply train p75 tau by these.")

    return parser.parse_args()


def _parse_int_list_csv(s: str) -> List[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    out: List[int] = []
    for p in parts:
        out.append(int(p))
    if not out:
        raise ValueError(f"Expected a non-empty comma-separated int list, got {s!r}")
    return out


def _parse_float_list_csv(s: str) -> List[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    out: List[float] = []
    for p in parts:
        out.append(float(p))
    if not out:
        raise ValueError(f"Expected a non-empty comma-separated float list, got {s!r}")
    return out


def _parse_devices_list(raw: str) -> List[str]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("--devices must be non-empty")
    for p in parts:
        if not p.isdigit():
            raise ValueError(f"Invalid device in --devices: {p!r}")
    if len(set(parts)) != len(parts):
        raise ValueError(f"Duplicate devices in --devices: {raw!r}")
    return parts


def dataset_tag_from_path(data_path: str) -> str:
    base = os.path.basename(str(data_path))
    if base.endswith(".pkl"):
        base = base[:-4]
    parent = os.path.basename(os.path.dirname(str(data_path)))
    if parent and parent != ".":
        return f"{parent}_{base}".replace(" ", "_")
    return base.replace(" ", "_")


def _project_tmp_dir(project_root: str) -> str:
    tmp_root = os.path.join(project_root, ".tmp")
    os.makedirs(tmp_root, exist_ok=True)
    return tmp_root


def snapshot_code(project_root: str, src_project_dir: str) -> Tuple[str, str]:
    """
    Snapshot trainer + model + reranker file under project .tmp.
    Returns (snapshot_root, snapshot_train_path).
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_root = os.path.join(_project_tmp_dir(project_root), f"st_prior_reranker_hparam_search_snapshot_{timestamp}")
    os.makedirs(snapshot_root, exist_ok=True)

    orig_train = os.path.join(src_project_dir, "train_self_attn_with_window_input.py")
    orig_models_dir = os.path.join(src_project_dir, "models")
    orig_reranker = os.path.join(src_project_dir, "st_prior_reranker.py")

    snapshot_train = os.path.join(snapshot_root, "train_self_attn_with_window_input.py")
    shutil.copy2(orig_train, snapshot_train)
    shutil.copy2(orig_reranker, os.path.join(snapshot_root, "st_prior_reranker.py"))

    snapshot_models_dir = os.path.join(snapshot_root, "models")
    os.makedirs(snapshot_models_dir, exist_ok=True)
    for filename in os.listdir(orig_models_dir):
        if filename.endswith(".py"):
            shutil.copy2(os.path.join(orig_models_dir, filename), os.path.join(snapshot_models_dir, filename))

    return snapshot_root, snapshot_train


def _load_snapshot_module(snapshot_root: str, snapshot_train: str):
    """
    Load the snapshot trainer as a python module, so we can reuse its dataset/model builders for eval/search.
    """
    sys.path.insert(0, snapshot_root)
    spec = importlib.util.spec_from_file_location("snapshot_train_module", snapshot_train)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import snapshot module from {snapshot_train}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def build_train_cmd(
    snapshot_train: str,
    args: argparse.Namespace,
    *,
    data_path: str,
    window_size: int,
    device: str,
    checkpoint_path: str,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-u",
        snapshot_train,
        "--data_path",
        str(data_path),
        "--window_size",
        str(window_size),
        "--use_st_prior_re_ranker",
        "--checkpoint_path",
        str(checkpoint_path),
        "--test_on_every_val_improvement",
        "false",
    ]

    def add_opt(flag: str, value) -> None:
        if value is None:
            return
        cmd.extend([flag, str(value)])

    def add_opt_list(flag: str, values: Optional[List[int]]) -> None:
        if values is None:
            return
        cmd.append(flag)
        cmd.extend(str(int(v)) for v in values)

    add_opt("--epochs", args.epochs)
    add_opt("--batch_size", args.batch_size)
    add_opt("--eval_batch_size", args.eval_batch_size)
    add_opt("--lr", args.lr)
    add_opt("--weight_decay", args.weight_decay)
    add_opt("--dtype", args.dtype)
    add_opt("--norm_first", args.norm_first)

    add_opt("--use_cat_emb", args.use_cat_emb)
    add_opt("--use_user_emb", args.use_user_emb)
    add_opt("--use_positional_encoding", args.use_positional_encoding)
    add_opt("--use_tod_slot_embedding", args.use_tod_slot_embedding)
    add_opt("--use_geo_cell_embedding", args.use_geo_cell_embedding)
    add_opt("--classifier_token_position", args.classifier_token_position)

    add_opt("--poi_emb_dim", args.poi_emb_dim)
    add_opt("--cat_emb_dim", args.cat_emb_dim)
    add_opt("--user_emb_dim", args.user_emb_dim)
    add_opt("--tod_slot_emb_dim", args.tod_slot_emb_dim)
    add_opt("--geo_cell_emb_dim", args.geo_cell_emb_dim)
    add_opt_list("--tod_slot_scales", args.tod_slot_scales)
    add_opt_list("--geo_cell_sizes_m", args.geo_cell_sizes_m)
    add_opt("--self_attn_d_model", args.self_attn_d_model)
    add_opt("--self_attn_num_layers", args.self_attn_num_layers)
    add_opt("--self_attn_num_heads", args.self_attn_num_heads)
    add_opt("--self_attn_dropout", args.self_attn_dropout)
    add_opt("--embedding_dropout", args.embedding_dropout)
    add_opt("--output_dropout", args.output_dropout)

    add_opt("--seed", args.seed)

    # POI embedding contrastive learning
    add_opt("--use_poi_embedding_contrastive_learning", args.use_poi_embedding_contrastive_learning)
    add_opt("--top_k_candidates_for_poi_embedding_contrastive_learning", args.top_k_candidates_for_poi_embedding_contrastive_learning)
    add_opt(
        "--poi_embedding_contrastive_learning_force_label_into_candidates_strategy",
        args.poi_embedding_contrastive_learning_force_label_into_candidates_strategy,
    )
    add_opt("--poi_embedding_contrastive_learning_proj_dim", args.poi_embedding_contrastive_learning_proj_dim)
    add_opt("--use_mlp_for_cl_instead_of_simple_proj", args.use_mlp_for_cl_instead_of_simple_proj)
    add_opt("--poi_embedding_contrastive_learning_temperature", args.poi_embedding_contrastive_learning_temperature)
    add_opt("--poi_embedding_contrastive_learning_normalize_embeddings", args.poi_embedding_contrastive_learning_normalize_embeddings)
    add_opt("--poi_embedding_contrastive_learning_loss_weight", args.poi_embedding_contrastive_learning_loss_weight)
    add_opt("--poi_ecl_negative_source", args.poi_ecl_negative_source)

    # fixed reranker structure (weights will be searched later; these are just placeholders for eval during training)
    add_opt("--st_prior_candidate_k", args.st_prior_candidate_k)
    add_opt("--st_prior_time_bins", args.st_prior_time_bins)
    add_opt("--st_prior_alpha", args.st_prior_alpha)
    add_opt("--st_prior_user_bin_min_count", args.st_prior_user_bin_min_count)
    add_opt("--st_prior_tau_sample_cap", args.st_prior_tau_sample_cap)
    add_opt("--st_prior_tau_km", -1.0)
    add_opt("--st_prior_sigma_min_km", 0.5)
    add_opt("--st_prior_lambda_time", 0.3)
    add_opt("--st_prior_lambda_user", 0.5)
    add_opt("--st_prior_lambda_dist", 0.5)

    cmd.extend(["--device", str(device)])
    return cmd


@torch.no_grad()
def _compute_metrics_from_candidates(
    cand_idx: torch.Tensor,  # (N,K) long
    cand_scores: torch.Tensor,  # (N,K) float
    targets: torch.Tensor,  # (N,)
    top_k_list: List[int],
) -> Dict[str, float]:
    n, k = cand_idx.shape
    max_k = min(int(max(top_k_list)), int(k))
    _, rel = torch.topk(cand_scores, k=max_k, dim=1)
    top_pois = cand_idx.gather(1, rel)
    correct = top_pois.eq(targets.unsqueeze(1))
    out: Dict[str, float] = {}
    for kk in top_k_list:
        kkk = min(int(kk), int(max_k))
        out[f"Acc@{kk}"] = float(correct[:, :kkk].any(dim=1).float().mean().item())

    mask = cand_idx.eq(targets.unsqueeze(1))
    in_set = mask.any(dim=1)
    tgt_score = (cand_scores * mask.to(dtype=cand_scores.dtype)).sum(dim=1)
    higher = (cand_scores > tgt_score.unsqueeze(1)).sum(dim=1).float()
    rank = higher + 1.0
    mrr = torch.where(in_set, 1.0 / rank, torch.zeros_like(rank))
    out["MRR"] = float(mrr.mean().item())
    return out


def _make_val_cache(
    *,
    mod,
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    dtype: torch.dtype,
    tables,
    candidate_k: int,
    compute_cl_sim: bool,
    cl_sim_normalize_embeddings: bool,
    top_k_list: Optional[List[int]] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, float]]]:
    """
    Cache per-sample tensors needed for fast large-scale search on val.
    """
    from st_prior_reranker import _haversine_km as haversine_km  # type: ignore
    from st_prior_reranker import _tod_to_bin as tod_to_bin  # type: ignore

    model.eval()

    base_metrics: Optional[Dict[str, float]] = None
    base_metrics_raw: Optional[Dict[str, float]] = None
    if top_k_list is not None:
        base_metrics_raw = {f"Acc@{k}": 0.0 for k in top_k_list}
        base_metrics_raw["MRR"] = 0.0
        base_metrics_raw["count"] = 0.0

    all_cand_idx: List[torch.Tensor] = []
    all_cand_base: List[torch.Tensor] = []
    all_s_time: List[torch.Tensor] = []
    all_dist_user: List[torch.Tensor] = []
    all_sigma_raw: List[torch.Tensor] = []
    all_dist_last: List[torch.Tensor] = []
    all_has_last: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_s_cl_sim: List[torch.Tensor] = []

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

        query_tod = batch["query_tod"].to(device=device, dtype=dtype, non_blocking=True)
        last_lat = batch["last_lat"].to(device=device, dtype=dtype, non_blocking=True)
        last_lon = batch["last_lon"].to(device=device, dtype=dtype, non_blocking=True)
        has_last = batch["has_last"].to(device=device, non_blocking=True).to(dtype=torch.bool)

        if bool(compute_cl_sim):
            out = model(
                user,
                window_poi,
                window_cat,
                window_tod_slot_ids=window_tod_slot_ids,
                window_geo_cell_x_ids=window_geo_cell_x_ids,
                window_geo_cell_y_ids=window_geo_cell_y_ids,
                return_user_out=True,
            )
            if not isinstance(out, tuple) or len(out) != 2:
                raise RuntimeError("Expected model(..., return_user_out=True) to return (logits,user_out)")
            logits, user_out = out
        else:
            logits = model(
                user,
                window_poi,
                window_cat,
                window_tod_slot_ids=window_tod_slot_ids,
                window_geo_cell_x_ids=window_geo_cell_x_ids,
                window_geo_cell_y_ids=window_geo_cell_y_ids,
            )
            user_out = None
        if base_metrics_raw is not None:
            mod.update_metrics(logits, targets, list(top_k_list or []), base_metrics_raw)
        base_scores, cand_idx = torch.topk(logits, k=int(candidate_k), dim=1)  # (B,K)

        b = tod_to_bin(query_tod, int(tables.time_bins))
        poi_lat = tables.poi_latlon[cand_idx, 0]
        poi_lon = tables.poi_latlon[cand_idx, 1]
        s_time = tables.poi_logp_time[cand_idx, b.view(-1, 1).expand(-1, int(candidate_k))]

        mu = tables.user_mu_latlon[user, b, :]  # (B,2)
        mu_lat = mu[:, 0].unsqueeze(1)
        mu_lon = mu[:, 1].unsqueeze(1)
        dist_user = haversine_km(poi_lat.to(torch.float32), poi_lon.to(torch.float32), mu_lat.to(torch.float32), mu_lon.to(torch.float32))  # (B,K)
        sigma_raw = tables.user_sigma_raw_km[user, b]  # (B,)

        dist_last = haversine_km(
            poi_lat.to(torch.float32),
            poi_lon.to(torch.float32),
            last_lat.unsqueeze(1).to(torch.float32),
            last_lon.unsqueeze(1).to(torch.float32),
        )

        if bool(compute_cl_sim):
            if user_out is None:
                raise RuntimeError("Internal error: compute_cl_sim=True but user_out is None")
            if not getattr(model, "use_poi_embedding_contrastive_learning", False):
                raise RuntimeError("compute_cl_sim=True but model.use_poi_embedding_contrastive_learning is False")
            poi_ecl_user_proj = getattr(model, "poi_ecl_user_proj", None)
            poi_ecl_poi_proj = getattr(model, "poi_ecl_poi_proj", None)
            poi_embedding = getattr(model, "poi_embedding", None)
            if poi_ecl_user_proj is None or poi_ecl_poi_proj is None or poi_embedding is None:
                raise RuntimeError("compute_cl_sim=True but POI-CL projection modules are not initialized.")
            u = poi_ecl_user_proj(user_out)  # (B,D)
            p = poi_ecl_poi_proj(poi_embedding(cand_idx))  # (B,K,D)
            if bool(cl_sim_normalize_embeddings):
                u = F.normalize(u, dim=1)
                p = F.normalize(p, dim=2)
            s_cl_sim = torch.einsum("bd,bkd->bk", u, p)  # (B,K)
            all_s_cl_sim.append(s_cl_sim.detach().to("cpu", dtype=torch.float32))

        all_cand_idx.append(cand_idx.detach().to("cpu"))
        all_cand_base.append(base_scores.detach().to("cpu", dtype=torch.float32))
        all_s_time.append(s_time.detach().to("cpu", dtype=torch.float32))
        all_dist_user.append(dist_user.detach().to("cpu", dtype=torch.float32))
        all_sigma_raw.append(sigma_raw.detach().to("cpu", dtype=torch.float32))
        all_dist_last.append(dist_last.detach().to("cpu", dtype=torch.float32))
        all_has_last.append(has_last.detach().to("cpu"))
        all_targets.append(targets.detach().to("cpu"))

    if base_metrics_raw is not None:
        base_metrics = mod.compute_metrics(base_metrics_raw)

    cache = {
        "cand_idx": torch.cat(all_cand_idx, dim=0),
        "cand_base": torch.cat(all_cand_base, dim=0),
        "s_time": torch.cat(all_s_time, dim=0),
        "dist_user": torch.cat(all_dist_user, dim=0),
        "sigma_raw": torch.cat(all_sigma_raw, dim=0),
        "dist_last": torch.cat(all_dist_last, dim=0),
        "has_last": torch.cat(all_has_last, dim=0),
        "targets": torch.cat(all_targets, dim=0),
    }
    if bool(compute_cl_sim):
        cache["s_cl_sim"] = torch.cat(all_s_cl_sim, dim=0)
    return cache, base_metrics


def _eval_hparams_on_cache(
    cache: Dict[str, torch.Tensor],
    *,
    lambda_time: float,
    lambda_user: float,
    lambda_dist: float,
    lambda_cl_sim: float,
    sigma_min_km: float,
    tau_km: float,
    top_k_list: List[int],
    subset_idx: Optional[torch.Tensor] = None,
    include_base: bool = True,
) -> Dict[str, float]:
    cand_idx = cache["cand_idx"]
    cand_base = cache["cand_base"]
    s_time = cache["s_time"]
    dist_user = cache["dist_user"]
    sigma_raw = cache["sigma_raw"]
    dist_last = cache["dist_last"]
    has_last = cache["has_last"]
    targets = cache["targets"]
    s_cl_sim = cache.get("s_cl_sim", None)

    if subset_idx is not None:
        cand_idx = cand_idx[subset_idx]
        cand_base = cand_base[subset_idx]
        s_time = s_time[subset_idx]
        dist_user = dist_user[subset_idx]
        sigma_raw = sigma_raw[subset_idx]
        dist_last = dist_last[subset_idx]
        has_last = has_last[subset_idx]
        targets = targets[subset_idx]
        if s_cl_sim is not None:
            s_cl_sim = s_cl_sim[subset_idx]

    sigma = torch.clamp(sigma_raw, min=float(sigma_min_km)).unsqueeze(1)
    s_user = -dist_user / sigma
    s_dist = (-dist_last / float(tau_km)) * has_last.to(dtype=dist_last.dtype).unsqueeze(1)
    scores = cand_base if bool(include_base) else torch.zeros_like(cand_base)
    scores = scores + float(lambda_time) * s_time + float(lambda_user) * s_user + float(lambda_dist) * s_dist
    if float(lambda_cl_sim) != 0.0:
        if s_cl_sim is None:
            raise ValueError("lambda_cl_sim is non-zero but cache does not contain s_cl_sim")
        scores = scores + float(lambda_cl_sim) * s_cl_sim

    return _compute_metrics_from_candidates(cand_idx, scores, targets, top_k_list)


def _auto_chunk_size(
    *,
    candidate_k: int,
    num_parallel_trials: int,
    device: torch.device,
) -> int:
    """
    Heuristic chunk size to limit peak memory for scores tensor with shape (T, chunk, K).
    """
    k = max(1, int(candidate_k))
    t = max(1, int(num_parallel_trials))
    # (T, chunk, K) float32 => 4 bytes
    # Target ~256MB on CUDA, ~1GB on CPU.
    target_bytes = 256 * 1024 * 1024 if device.type == "cuda" else 1024 * 1024 * 1024
    chunk = int(target_bytes // (4 * t * k))
    return int(max(256, min(8192, chunk)))


@torch.no_grad()
def _eval_hparams_on_cache_parallel_trials(
    cache: Dict[str, torch.Tensor],
    *,
    lambda_time: torch.Tensor,  # (T,)
    lambda_user: torch.Tensor,  # (T,)
    lambda_dist: torch.Tensor,  # (T,)
    lambda_cl_sim: torch.Tensor,  # (T,)
    sigma_min_km: torch.Tensor,  # (T,)
    tau_km: torch.Tensor,  # (T,)
    top_k_list: List[int],
    device: torch.device,
    chunk_size: int,
) -> List[Dict[str, float]]:
    """
    Vectorized evaluation of T hyperparameter configs on the same cached candidates.
    This is only used for stage-1 search throughput; it mirrors _eval_hparams_on_cache metrics.
    """
    cand_idx_cpu = cache["cand_idx"]
    cand_base_cpu = cache["cand_base"]
    s_time_cpu = cache["s_time"]
    dist_user_cpu = cache["dist_user"]
    sigma_raw_cpu = cache["sigma_raw"]
    dist_last_cpu = cache["dist_last"]
    has_last_cpu = cache["has_last"]
    targets_cpu = cache["targets"]
    s_cl_sim_cpu = cache.get("s_cl_sim", None)

    if cand_idx_cpu.dim() != 2:
        raise ValueError(f"cand_idx must be (N,K), got {tuple(cand_idx_cpu.shape)}")
    n, k = cand_idx_cpu.shape
    if n == 0:
        return [{f"Acc@{kk}": 0.0 for kk in top_k_list} | {"MRR": 0.0} for _ in range(int(lambda_time.numel()))]

    t = int(lambda_time.numel())
    if t <= 0:
        return []

    max_k = min(int(max(top_k_list)), int(k))
    if max_k <= 0:
        raise ValueError(f"Invalid max_k computed from top_k_list={top_k_list} and candidate_k={k}")

    # Move trial params to device once.
    lambda_time = lambda_time.to(device=device, dtype=torch.float32)
    lambda_user = lambda_user.to(device=device, dtype=torch.float32)
    lambda_dist = lambda_dist.to(device=device, dtype=torch.float32)
    lambda_cl_sim = lambda_cl_sim.to(device=device, dtype=torch.float32)
    sigma_min_km = sigma_min_km.to(device=device, dtype=torch.float32)
    tau_km = tau_km.to(device=device, dtype=torch.float32)

    acc_sums: Dict[int, torch.Tensor] = {kk: torch.zeros((t,), device=device, dtype=torch.float32) for kk in top_k_list}
    mrr_sum = torch.zeros((t,), device=device, dtype=torch.float32)
    count = 0

    for start in range(0, int(n), int(chunk_size)):
        end = min(int(n), start + int(chunk_size))
        bs = end - start
        count += bs

        cand_idx = cand_idx_cpu[start:end].to(device=device, non_blocking=True)
        cand_base = cand_base_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        s_time = s_time_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        dist_user = dist_user_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        sigma_raw = sigma_raw_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        dist_last = dist_last_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        has_last = has_last_cpu[start:end].to(device=device, non_blocking=True).to(dtype=torch.bool)
        targets = targets_cpu[start:end].to(device=device, non_blocking=True)
        s_cl_sim = None
        if s_cl_sim_cpu is not None:
            s_cl_sim = s_cl_sim_cpu[start:end].to(device=device, dtype=torch.float32, non_blocking=True)

        # scores: (T, bs, K)
        sigma = torch.clamp(sigma_raw.unsqueeze(0).expand(t, bs), min=sigma_min_km.unsqueeze(1)).unsqueeze(2)
        s_user = -dist_user.unsqueeze(0) / sigma
        s_dist = (-dist_last.unsqueeze(0) / tau_km.view(t, 1, 1)) * has_last.view(1, bs, 1).to(dtype=torch.float32)
        scores = (
            cand_base.unsqueeze(0)
            + lambda_time.view(t, 1, 1) * s_time.unsqueeze(0)
            + lambda_user.view(t, 1, 1) * s_user
            + lambda_dist.view(t, 1, 1) * s_dist
        )
        if s_cl_sim is not None:
            scores = scores + lambda_cl_sim.view(t, 1, 1) * s_cl_sim.unsqueeze(0)
        else:
            if bool((lambda_cl_sim != 0.0).any().item()):
                raise ValueError("lambda_cl_sim has non-zero entries but cache does not contain s_cl_sim")

        # topk within candidates
        _, rel = torch.topk(scores, k=max_k, dim=2)
        top_pois = cand_idx.unsqueeze(0).expand(t, bs, k).gather(2, rel)
        correct = top_pois.eq(targets.view(1, bs, 1))
        for kk in top_k_list:
            kkk = min(int(kk), int(max_k))
            acc_sums[int(kk)] += correct[:, :, :kkk].any(dim=2).to(dtype=torch.float32).sum(dim=1)

        # MRR within candidates (0 if target not in candidate set)
        mask = cand_idx.unsqueeze(0).eq(targets.view(1, bs, 1))  # (T, bs, K), broadcast on T
        in_set = mask.any(dim=2)
        tgt_score = (scores * mask.to(dtype=scores.dtype)).sum(dim=2)
        higher = (scores > tgt_score.unsqueeze(2)).sum(dim=2).to(dtype=torch.float32)
        rank = higher + 1.0
        mrr = torch.where(in_set, 1.0 / rank, torch.zeros_like(rank))
        mrr_sum += mrr.sum(dim=1)

    denom = float(max(1, count))
    out: List[Dict[str, float]] = []
    for i in range(t):
        m: Dict[str, float] = {}
        for kk in top_k_list:
            m[f"Acc@{int(kk)}"] = float((acc_sums[int(kk)][i] / denom).item())
        m["MRR"] = float((mrr_sum[i] / denom).item())
        out.append(m)
    return out


def main() -> None:
    args = parse_args()
    exp_scripts_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(exp_scripts_dir, os.pardir))
    src_project_dir = project_root

    window_sizes = _parse_int_list_csv(args.window_size_list)
    devices = _parse_devices_list(args.devices)

    snapshot_root, snapshot_train = snapshot_code(project_root, src_project_dir)
    print("Code snapshot root:", snapshot_root)
    mod = _load_snapshot_module(snapshot_root, snapshot_train)

    # Prepare checkpoints dir
    checkpoints_dir = os.path.join(src_project_dir, "exp_scripts", "results", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Training schedule (one run per dataset/window)
    tasks: List[Dict] = []
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for data_path in args.data_path_list:
        ds_tag = dataset_tag_from_path(data_path)
        for k in window_sizes:
            ckpt_name = f"st_prior_backbone_{ds_tag}_k{k}_{ts}.ckpt"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
            tasks.append(
                {
                    "data_path": str(data_path),
                    "dataset_tag": ds_tag,
                    "window_size": int(k),
                    "checkpoint_path": ckpt_path,
                }
            )

    logs_dir = os.path.join(_project_tmp_dir(project_root), f"logs_st_prior_hparam_search_train_{ts}")
    os.makedirs(logs_dir, exist_ok=True)

    print("Training schedule:")
    print(f"- tasks: {len(tasks)}")
    print(f"- devices: cuda:{devices}")
    print(f"- logs_dir: {logs_dir}")
    print(f"- just_run_all_runs_together: {bool(args.just_run_all_runs_together)}")

    available = list(devices)
    running: Dict[str, Dict] = {}
    running_all: Dict[int, Dict] = {}
    pending = list(tasks)
    train_results: List[Dict] = []

    def _task_tag(t: Dict) -> str:
        return f"dataset={t['dataset_tag']} k={t['window_size']}"

    launched = 0
    if bool(args.just_run_all_runs_together):
        for idx, task in enumerate(list(pending), start=1):
            dev = devices[(idx - 1) % len(devices)]
            launched += 1
            log_path = os.path.join(logs_dir, f"{task['dataset_tag']}_k{task['window_size']}.log")
            cmd = build_train_cmd(
                snapshot_train,
                args,
                data_path=task["data_path"],
                window_size=task["window_size"],
                device=dev,
                checkpoint_path=task["checkpoint_path"],
            )
            log_f = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=project_root, stdout=log_f, stderr=subprocess.STDOUT, text=True)
            running_all[int(proc.pid)] = {"proc": proc, "log_f": log_f, "log_path": log_path, "device": dev, "task": task}
            print(f"[launch {launched}/{len(tasks)} cuda:{dev}] {_task_tag(task)} pid={proc.pid} log={log_path}")
        pending = []

        while running_all:
            done_pids: List[int] = []
            for pid, info in list(running_all.items()):
                ret = info["proc"].poll()
                if ret is None:
                    continue
                info["log_f"].close()
                task = info["task"]
                dev = info["device"]
                train_results.append(
                    {
                        "dataset_tag": task["dataset_tag"],
                        "data_path": task["data_path"],
                        "window_size": task["window_size"],
                        "checkpoint_path": task["checkpoint_path"],
                        "device": dev,
                        "returncode": int(ret),
                        "log_path": os.path.relpath(info["log_path"], project_root),
                    }
                )
                print(f"[done cuda:{dev}] {_task_tag(task)} returncode={ret}")
                done_pids.append(int(pid))
            for pid in done_pids:
                running_all.pop(int(pid), None)
            if running_all and not done_pids:
                time.sleep(2.0)
    else:
        while pending or running:
            while pending and available:
                dev = available.pop(0)
                task = pending.pop(0)
                launched += 1
                log_path = os.path.join(logs_dir, f"{task['dataset_tag']}_k{task['window_size']}.log")
                cmd = build_train_cmd(
                    snapshot_train,
                    args,
                    data_path=task["data_path"],
                    window_size=task["window_size"],
                    device=dev,
                    checkpoint_path=task["checkpoint_path"],
                )
                log_f = open(log_path, "w", encoding="utf-8")
                proc = subprocess.Popen(cmd, cwd=project_root, stdout=log_f, stderr=subprocess.STDOUT, text=True)
                running[dev] = {"proc": proc, "log_f": log_f, "log_path": log_path, "device": dev, "task": task}
                print(f"[launch {launched}/{len(tasks)} cuda:{dev}] {_task_tag(task)} pid={proc.pid} log={log_path}")

            done_devices: List[str] = []
            for dev, info in running.items():
                ret = info["proc"].poll()
                if ret is None:
                    continue
                info["log_f"].close()
                train_results.append(
                    {
                        "dataset_tag": info["task"]["dataset_tag"],
                        "data_path": info["task"]["data_path"],
                        "window_size": info["task"]["window_size"],
                        "checkpoint_path": info["task"]["checkpoint_path"],
                        "device": dev,
                        "returncode": int(ret),
                        "log_path": os.path.relpath(info["log_path"], project_root),
                    }
                )
                print(f"[done cuda:{dev}] {_task_tag(info['task'])} returncode={ret}")
                done_devices.append(dev)

            for dev in done_devices:
                running.pop(dev, None)
                available.append(dev)

            if running and not done_devices:
                time.sleep(2.0)

    # Hyperparameter search (val only)
    top_k_list = [1, 5, 10, 20]
    sigma_choices = _parse_float_list_csv(args.sigma_min_choices_km)
    tau_mult_choices = _parse_float_list_csv(args.tau_mult_choices)

    search_device = args.search_device
    if search_device is None:
        search_device = f"cuda:{devices[0]}" if devices else "cpu"
    device = torch.device(search_device)
    dtype = torch.bfloat16 if str(args.dtype or "fp32").lower() == "bf16" else torch.float32
    score_terms = set(str(x) for x in (args.st_prior_score_terms or []))
    if not score_terms:
        raise ValueError("--st_prior_score_terms must be non-empty")

    rng = random.Random(int(args.seed))
    lambda_min = float(args.lambda_min)
    lambda_max = float(args.lambda_max)

    report_rows: List[Dict] = []

    for tr in train_results:
        if tr["returncode"] != 0:
            report_rows.append({**tr, "status": "train_failed"})
            continue

        data_path = tr["data_path"]
        window_size = int(tr["window_size"])
        ckpt_path = tr["checkpoint_path"]

        # load data
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        user2idx, poi2idx, cat2idx = mod.build_vocab(data)
        num_users = len(user2idx) + 1
        num_pois = len(poi2idx) + 1
        num_cats = len(cat2idx) + 1

        tables = mod.build_st_prior_tables(
            train_split=data["train"],
            user2idx=user2idx,
            poi2idx=poi2idx,
            time_bins=int(args.st_prior_time_bins),
            alpha=float(args.st_prior_alpha),
            user_bin_min_count=int(args.st_prior_user_bin_min_count),
            seed=int(args.seed),
            tau_km=-1.0,
            tau_sample_cap=int(args.st_prior_tau_sample_cap),
        )
        tables.to(device=device, dtype=dtype)

        # load model checkpoint (to recover exact feature flags/hparams)
        ckpt = torch.load(ckpt_path, map_location=device)
        h = ckpt.get("hparams") or {}
        use_tod_slot_embedding = bool(h.get("use_tod_slot_embedding", False))
        tod_slot_scales = h.get("tod_slot_scales", None)
        tod_slot_emb_dim = int(h.get("tod_slot_emb_dim", args.tod_slot_emb_dim or 16))
        cl_sim_requested = "cl_sim" in score_terms
        cl_sim_enabled = bool(cl_sim_requested) and bool(h.get("use_poi_embedding_contrastive_learning", False))
        cl_sim_normalize = bool(
            h.get(
                "poi_embedding_contrastive_learning_normalize_embeddings",
                True if args.poi_embedding_contrastive_learning_normalize_embeddings is None else args.poi_embedding_contrastive_learning_normalize_embeddings,
            )
        )

        # build datasets/loaders
        train_tensors = mod.build_split_tensors(data["train"], user2idx=user2idx, poi2idx=poi2idx, cat2idx=cat2idx, dtype=dtype)
        val_tensors = mod.build_split_tensors(data["val"], user2idx=user2idx, poi2idx=poi2idx, cat2idx=cat2idx, dtype=dtype)
        test_tensors = mod.build_split_tensors(data["test"], user2idx=user2idx, poi2idx=poi2idx, cat2idx=cat2idx, dtype=dtype)

        all_users = set(train_tensors.keys()) | set(val_tensors.keys()) | set(test_tensors.keys())

        train_label_ranges: Dict[int, Tuple[int, int]] = {}
        train_combined: Dict[int, object] = {}
        val_label_ranges: Dict[int, Tuple[int, int]] = {}
        val_combined: Dict[int, object] = {}
        test_label_ranges: Dict[int, Tuple[int, int]] = {}
        test_combined: Dict[int, object] = {}

        for u in sorted(all_users):
            tr_seq = train_tensors.get(u, None)
            va_seq = val_tensors.get(u, None)
            te_seq = test_tensors.get(u, None)

            if tr_seq is not None:
                train_combined[u] = tr_seq
                train_label_ranges[u] = (0, int(tr_seq.poi.size(0)))

            parts_tv = [p for p in [tr_seq, va_seq] if p is not None]
            if parts_tv:
                combined_tv = mod._concat_seq_tensors(parts_tv)
                val_combined[u] = combined_tv
                tr_len = 0 if tr_seq is None else int(tr_seq.poi.size(0))
                va_len = 0 if va_seq is None else int(va_seq.poi.size(0))
                if va_len > 0:
                    val_label_ranges[u] = (tr_len, tr_len + va_len)

            parts_tvt = [p for p in [tr_seq, va_seq, te_seq] if p is not None]
            if parts_tvt:
                combined_tvt = mod._concat_seq_tensors(parts_tvt)
                test_combined[u] = combined_tvt
                tr_len = 0 if tr_seq is None else int(tr_seq.poi.size(0))
                va_len = 0 if va_seq is None else int(va_seq.poi.size(0))
                te_len = 0 if te_seq is None else int(te_seq.poi.size(0))
                if te_len > 0:
                    start = tr_len + va_len
                    test_label_ranges[u] = (start, start + te_len)

        use_cat_emb = bool(True if args.use_cat_emb is None else args.use_cat_emb)
        use_geo_cell_embedding = bool(True if args.use_geo_cell_embedding is None else args.use_geo_cell_embedding)

        geo_cell_meta = None
        geo_val_ids = None
        geo_test_ids = None
        if use_geo_cell_embedding:
            resolved_geo_sizes_m = list(args.geo_cell_sizes_m or getattr(mod, "GEO_CELL_SIZES_M_DEFAULT", [500, 2000]))
            geo_cell_meta = mod._build_geo_cell_meta(train_combined, resolved_geo_sizes_m)
            geo_val_ids = mod._precompute_geo_cell_ids_by_user(val_combined, geo_cell_meta)
            geo_test_ids = mod._precompute_geo_cell_ids_by_user(test_combined, geo_cell_meta)

        val_ds = mod.WindowInputDataset(
            val_combined,
            val_label_ranges,
            window_size=int(window_size),
            use_cat_emb=use_cat_emb,
            use_tod_slot_embedding=use_tod_slot_embedding,
            tod_slot_scales=tod_slot_scales,
            use_geo_cell_embedding=use_geo_cell_embedding,
            geo_cell_ids_by_user=geo_val_ids,
            geo_cell_num_scales=(None if geo_cell_meta is None else len(geo_cell_meta.cell_sizes_m)),
            pad_idx=int(mod.PAD_IDX),
        )
        test_ds = mod.WindowInputDataset(
            test_combined,
            test_label_ranges,
            window_size=int(window_size),
            use_cat_emb=use_cat_emb,
            use_tod_slot_embedding=use_tod_slot_embedding,
            tod_slot_scales=tod_slot_scales,
            use_geo_cell_embedding=use_geo_cell_embedding,
            geo_cell_ids_by_user=geo_test_ids,
            geo_cell_num_scales=(None if geo_cell_meta is None else len(geo_cell_meta.cell_sizes_m)),
            pad_idx=int(mod.PAD_IDX),
        )

        eval_bs = int(args.eval_batch_size or getattr(mod, "EVAL_BATCH_SIZE", 4096))
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

        model = mod.NextPOIWindowSelfAttn(
            num_users=num_users,
            num_pois=num_pois,
            num_cats=num_cats,
            window_size=int(h.get("window_size", window_size)),
            d_model=int(h.get("self_attn_d_model", args.self_attn_d_model or 128)),
            nhead=int(h.get("self_attn_num_heads", args.self_attn_num_heads or 4)),
            num_layers=int(h.get("self_attn_num_layers", args.self_attn_num_layers or 2)),
            dropout=float(h.get("self_attn_dropout", args.self_attn_dropout or 0.1)),
            embedding_dropout=float(h.get("embedding_dropout", args.embedding_dropout or 0.0)),
            output_dropout=float(h.get("output_dropout", args.output_dropout or 0.0)),
            norm_first=bool(h.get("self_attn_norm_first", True)),
            poi_emb_dim=int(h.get("poi_emb_dim", args.poi_emb_dim or 128)),
            cat_emb_dim=int(h.get("cat_emb_dim", args.cat_emb_dim or 32)),
            user_emb_dim=h.get("user_emb_dim", args.user_emb_dim),
            use_poi_embedding_contrastive_learning=bool(h.get("use_poi_embedding_contrastive_learning", False)),
            poi_embedding_contrastive_learning_proj_dim=int(
                h.get("poi_embedding_contrastive_learning_proj_dim", args.poi_embedding_contrastive_learning_proj_dim or 256)
            ),
            use_mlp_for_cl_instead_of_simple_proj=bool(
                h.get(
                    "use_mlp_for_cl_instead_of_simple_proj",
                    False if args.use_mlp_for_cl_instead_of_simple_proj is None else args.use_mlp_for_cl_instead_of_simple_proj,
                )
            ),
            use_tod_slot_embedding=bool(h.get("use_tod_slot_embedding", use_tod_slot_embedding)),
            tod_slot_scales=h.get("tod_slot_scales", tod_slot_scales),
            tod_slot_emb_dim=int(h.get("tod_slot_emb_dim", tod_slot_emb_dim)),
            use_geo_cell_embedding=bool(h.get("use_geo_cell_embedding", use_geo_cell_embedding)),
            geo_cell_sizes_m=(None if geo_cell_meta is None else geo_cell_meta.cell_sizes_m),
            geo_cell_grid_w=(None if geo_cell_meta is None else geo_cell_meta.grid_w_list),
            geo_cell_grid_h=(None if geo_cell_meta is None else geo_cell_meta.grid_h_list),
            geo_cell_emb_dim=int(h.get("geo_cell_emb_dim", args.geo_cell_emb_dim or getattr(mod, "GEO_CELL_EMB_DIM", 64))),
            use_user_emb=bool(h.get("use_user_emb", True if args.use_user_emb is None else args.use_user_emb)),
            use_cat_emb=bool(h.get("use_cat_emb", use_cat_emb)),
            use_positional_encoding=bool(h.get("use_positional_encoding", True if args.use_positional_encoding is None else args.use_positional_encoding)),
            classifier_token_position=str(h.get("classifier_token_position", "first")),
            pad_idx=int(mod.PAD_IDX),
        ).to(device=device, dtype=dtype)
        model.load_state_dict(ckpt["model_state_dict"])

        # cache val once
        cache_val, val_base_metrics = _make_val_cache(
            mod=mod,
            model=model,
            data_loader=val_loader,
            device=device,
            dtype=dtype,
            tables=tables,
            candidate_k=int(args.st_prior_candidate_k),
            compute_cl_sim=bool(cl_sim_enabled),
            cl_sim_normalize_embeddings=bool(cl_sim_normalize),
            top_k_list=top_k_list,
        )

        n_val = int(cache_val["targets"].size(0))
        eval_n = min(int(args.search_eval_samples), n_val)
        subset_idx = None
        if eval_n < n_val:
            subset_idx = torch.randperm(n_val)[:eval_n]

        # stage-1 random search
        trials = int(args.search_trials)
        top_m = int(args.search_top_m)
        scored: List[Tuple[float, Dict]] = []
        num_parallel = int(args.st_prior_num_parallel_trials)
        if num_parallel <= 0:
            raise ValueError(f"--st_prior_num_parallel_trials must be > 0, got {args.st_prior_num_parallel_trials}")
        chunk_size = _auto_chunk_size(candidate_k=int(args.st_prior_candidate_k), num_parallel_trials=num_parallel, device=device)

        # Slice stage-1 cache once (avoid per-trial advanced indexing overhead).
        cache_stage1 = cache_val
        if subset_idx is not None:
            cache_stage1 = {k: v[subset_idx] for k, v in cache_val.items()}

        for start in range(0, trials, num_parallel):
            t = min(num_parallel, trials - start)
            l1_list: List[float] = []
            l2_list: List[float] = []
            l3_list: List[float] = []
            l4_list: List[float] = []
            sigma_list: List[float] = []
            tau_list: List[float] = []
            cfgs: List[Dict] = []
            for _ in range(int(t)):
                l1 = rng.uniform(lambda_min, lambda_max) if "time" in score_terms else 0.0
                l2 = rng.uniform(lambda_min, lambda_max) if "user" in score_terms else 0.0
                l3 = rng.uniform(lambda_min, lambda_max) if "dist" in score_terms else 0.0
                l4 = rng.uniform(lambda_min, lambda_max) if bool(cl_sim_enabled) else 0.0
                sigma_min = rng.choice(sigma_choices)
                tau = float(tables.tau_p75_km) * rng.choice(tau_mult_choices)
                l1_list.append(float(l1))
                l2_list.append(float(l2))
                l3_list.append(float(l3))
                l4_list.append(float(l4))
                sigma_list.append(float(sigma_min))
                tau_list.append(float(tau))
                cfgs.append(
                    {
                        "lambda_time": float(l1),
                        "lambda_user": float(l2),
                        "lambda_dist": float(l3),
                        "lambda_cl_sim": float(l4),
                        "sigma_min_km": float(sigma_min),
                        "tau_km": float(tau),
                    }
                )

            metrics_list = _eval_hparams_on_cache_parallel_trials(
                cache_stage1,
                lambda_time=torch.tensor(l1_list, dtype=torch.float32),
                lambda_user=torch.tensor(l2_list, dtype=torch.float32),
                lambda_dist=torch.tensor(l3_list, dtype=torch.float32),
                lambda_cl_sim=torch.tensor(l4_list, dtype=torch.float32),
                sigma_min_km=torch.tensor(sigma_list, dtype=torch.float32),
                tau_km=torch.tensor(tau_list, dtype=torch.float32),
                top_k_list=top_k_list,
                device=device,
                chunk_size=chunk_size,
            )
            for m, cfg in zip(metrics_list, cfgs):
                scored.append((float(m["MRR"]), cfg))

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = [cfg for _score, cfg in scored[: max(1, top_m)]]

        # stage-2 full val eval on top-M
        best_cfg = None
        best_val_mrr = -1.0
        best_val_metrics: Dict[str, float] = {}
        for cfg in keep:
            m = _eval_hparams_on_cache(
                cache_val,
                lambda_time=cfg["lambda_time"],
                lambda_user=cfg["lambda_user"],
                lambda_dist=cfg["lambda_dist"],
                lambda_cl_sim=cfg.get("lambda_cl_sim", 0.0),
                sigma_min_km=cfg["sigma_min_km"],
                tau_km=cfg["tau_km"],
                top_k_list=top_k_list,
                subset_idx=None,
            )
            if float(m["MRR"]) > best_val_mrr:
                best_val_mrr = float(m["MRR"])
                best_val_metrics = m
                best_cfg = cfg

        # final test (once) with best cfg
        cache_test, test_base_metrics = _make_val_cache(
            mod=mod,
            model=model,
            data_loader=test_loader,
            device=device,
            dtype=dtype,
            tables=tables,
            candidate_k=int(args.st_prior_candidate_k),
            compute_cl_sim=bool(cl_sim_enabled),
            cl_sim_normalize_embeddings=bool(cl_sim_normalize),
            top_k_list=top_k_list,
        )
        test_metrics = _eval_hparams_on_cache(
            cache_test,
            lambda_time=best_cfg["lambda_time"],
            lambda_user=best_cfg["lambda_user"],
            lambda_dist=best_cfg["lambda_dist"],
            lambda_cl_sim=best_cfg.get("lambda_cl_sim", 0.0),
            sigma_min_km=best_cfg["sigma_min_km"],
            tau_km=best_cfg["tau_km"],
            top_k_list=top_k_list,
            subset_idx=None,
        )
        test_rerank_only_metrics = _eval_hparams_on_cache(
            cache_test,
            lambda_time=best_cfg["lambda_time"],
            lambda_user=best_cfg["lambda_user"],
            lambda_dist=best_cfg["lambda_dist"],
            lambda_cl_sim=best_cfg.get("lambda_cl_sim", 0.0),
            sigma_min_km=best_cfg["sigma_min_km"],
            tau_km=best_cfg["tau_km"],
            top_k_list=top_k_list,
            subset_idx=None,
            include_base=False,
        )

        report_rows.append(
            {
                **tr,
                "status": "ok",
                "val_base": val_base_metrics or {},
                "val_best": best_val_metrics,
                "test_base": test_base_metrics or {},
                "test": test_metrics,
                "test_rerank_only": test_rerank_only_metrics,
                "best_cfg": best_cfg,
                "score_terms": sorted(list(score_terms)),
                "cl_sim_enabled": bool(cl_sim_enabled),
                "cl_sim_normalize": bool(cl_sim_normalize),
                "tau_p75_km": float(tables.tau_p75_km),
                "val_size": n_val,
                "stage1_trials": trials,
                "stage1_eval_samples": eval_n,
                "stage1_parallel_trials": int(args.st_prior_num_parallel_trials),
            }
        )

    # write markdown
    results_dir = os.path.join(src_project_dir, "exp_scripts", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_md = os.path.join(results_dir, f"{args.exp_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    lines: List[str] = []
    lines.append(f"# {args.exp_id}")
    lines.append("")
    lines.append(f"- Code snapshot root: `{os.path.relpath(snapshot_root, project_root)}`")
    lines.append(f"- Wrapper invocation: `{html_escape(' '.join(sys.argv))}`")
    lines.append(f"- Training logs dir: `{os.path.relpath(logs_dir, project_root)}`")
    lines.append(f"- Checkpoints dir: `{os.path.relpath(checkpoints_dir, project_root)}`")
    lines.append(f"- Search device: `{str(device)}`")
    lines.append("")

    def _fmt_metrics(m: Dict[str, float]) -> str:
        return " ".join(f"{k}={float(v):.4f}" for k, v in m.items())

    def _fmt_metric_value(m: Dict[str, float], key: str) -> str:
        if not m or key not in m:
            return "NA"
        return f"{float(m[key]):.4f}"

    def _md_cell(value) -> str:
        return html_escape(str(value)).replace("|", "\\|")

    def _summary_dataset_name(r: Dict) -> str:
        candidates = [
            os.path.basename(str(r.get("data_path", ""))),
            str(r.get("dataset_tag", "")),
        ]
        for raw in candidates:
            name = raw[:-4] if raw.endswith(".pkl") else raw
            if name.startswith("processed_"):
                name = name[len("processed_") :]
            for suffix in ("_excluding_cold", "_including_cold"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
            if name:
                return name
        return ""

    summary_metric_keys = [f"Acc@{int(k)}" for k in top_k_list] + ["MRR"]
    lines.append("## Summary")
    lines.append("")
    for r in report_rows:
        base_metrics = r.get("test_base", {})
        rerank_metrics = r.get("test", {})
        lines.append(f"### {_md_cell(_summary_dataset_name(r))}, k={_md_cell(r.get('window_size', ''))}")
        lines.append("")
        lines.append("|  | " + " | ".join(summary_metric_keys) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(summary_metric_keys)) + " |")
        lines.append(
            "| w/o reranker | "
            + " | ".join(_fmt_metric_value(base_metrics, key) for key in summary_metric_keys)
            + " |"
        )
        lines.append(
            "| w/ reranker | "
            + " | ".join(_fmt_metric_value(rerank_metrics, key) for key in summary_metric_keys)
            + " |"
        )
        lines.append("")

    lines.append("## Results")
    lines.append("")

    for r in report_rows:
        lines.append(f"### {r.get('dataset_tag','')} k={r.get('window_size','')}")
        lines.append("")
        lines.append(f"- status: {r.get('status')}")
        lines.append(f"- data_path: `{r.get('data_path')}`")
        lines.append(f"- checkpoint_path: `{os.path.relpath(r.get('checkpoint_path',''), project_root)}`")
        lines.append(f"- train_device: cuda:{r.get('device')} returncode={r.get('returncode')} log={r.get('log_path')}")
        if "score_terms" in r:
            lines.append(f"- score_terms: {', '.join(r.get('score_terms') or [])}")
        if r.get("status") != "ok":
            lines.append("")
            continue
        cfg = r["best_cfg"]
        lines.append(f"- tau_p75_km: {r['tau_p75_km']:.2f}  (tau_km used: {cfg['tau_km']:.2f})")
        lines.append(
            "- best_cfg: "
            + f"lambda_time={cfg['lambda_time']:.4f} lambda_user={cfg['lambda_user']:.4f} lambda_dist={cfg['lambda_dist']:.4f} "
            + f"lambda_cl_sim={float(cfg.get('lambda_cl_sim', 0.0)):.4f} "
            + f"sigma_min_km={cfg['sigma_min_km']:.2f} tau_km={cfg['tau_km']:.2f}"
        )
        lines.append(
            f"- search: trials={r['stage1_trials']} stage1_eval_samples={r['stage1_eval_samples']} val_size={r['val_size']}"
        )
        if "stage1_parallel_trials" in r:
            lines.append(f"- stage1_parallel_trials: {int(r['stage1_parallel_trials'])}")
        lines.append(f"- val(base): {_fmt_metrics(r.get('val_base', {}))}")
        lines.append(f"- val(rerank best): {_fmt_metrics(r['val_best'])}")
        lines.append(f"- test(base): {_fmt_metrics(r.get('test_base', {}))}")
        lines.append(f"- test(rerank final once): {_fmt_metrics(r['test'])}")
        lines.append(f"- test(rerank score w/o base): {_fmt_metrics(r.get('test_rerank_only', {}))}")
        lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote report to {out_md}")


if __name__ == "__main__":
    main()
