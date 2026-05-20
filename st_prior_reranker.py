from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


def _haversine_km(
    lat1_deg: torch.Tensor,
    lon1_deg: torch.Tensor,
    lat2_deg: torch.Tensor,
    lon2_deg: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized haversine distance in kilometers.

    Shapes are broadcastable (e.g., (B, K) vs (B, 1)).
    """
    r = 6371.0
    deg2rad = math.pi / 180.0

    lat1 = lat1_deg * deg2rad
    lon1 = lon1_deg * deg2rad
    lat2 = lat2_deg * deg2rad
    lon2 = lon2_deg * deg2rad

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat * 0.5) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon * 0.5) ** 2
    c = 2.0 * torch.asin(torch.sqrt(torch.clamp(a, min=0.0, max=1.0)))
    return r * c


def _tod_to_bin(tod: torch.Tensor, num_bins: int) -> torch.Tensor:
    # tod in [0, 1], map to [0, num_bins-1]
    b = torch.floor(torch.clamp(tod, 0.0, 1.0 - 1e-12) * float(int(num_bins))).to(dtype=torch.long)
    return torch.clamp(b, 0, int(num_bins) - 1)


def _estimate_tau_p75_km_from_train(
    train_split: Dict[str, List[Dict]],
    *,
    sample_cap: int = 200_000,
    seed: int = 42,
) -> float:
    """
    Estimate a global distance scale (tau) as the 75th percentile of consecutive check-in distances in train.
    Uses a capped random subsample to avoid huge memory.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))

    dists: List[float] = []
    seen = 0

    for _u, seq in train_split.items():
        if not seq or len(seq) < 2:
            continue
        prev = seq[0]
        for rec in seq[1:]:
            seen += 1
            d = float(
                _haversine_km(
                    torch.tensor(float(prev["lat"])),
                    torch.tensor(float(prev["lon"])),
                    torch.tensor(float(rec["lat"])),
                    torch.tensor(float(rec["lon"])),
                ).item()
            )
            if len(dists) < int(sample_cap):
                dists.append(d)
            else:
                j = int(torch.randint(low=0, high=seen + 1, size=(1,), generator=g).item())
                if j < int(sample_cap):
                    dists[j] = d
            prev = rec

    if not dists:
        return 6.0
    dists.sort()
    idx = int(0.75 * (len(dists) - 1))
    return float(dists[idx])


@dataclass
class STPriorTables:
    time_bins: int
    alpha: float
    user_bin_min_count: int

    poi_latlon: torch.Tensor  # (num_pois, 2) float32, idx aligned (incl PAD=0)
    poi_logp_time: torch.Tensor  # (num_pois, B) float32, log P(bin|poi)

    user_mu_latlon: torch.Tensor  # (num_users, B, 2) float32, per-bin center (with backoff applied)
    user_sigma_raw_km: torch.Tensor  # (num_users, B) float32, per-bin radius (before sigma_min clamp)
    user_bin_count: torch.Tensor  # (num_users, B) int32

    tau_p75_km: float

    def to(self, device: torch.device, dtype: torch.dtype) -> "STPriorTables":
        self.poi_latlon = self.poi_latlon.to(device=device, dtype=dtype)
        self.poi_logp_time = self.poi_logp_time.to(device=device, dtype=dtype)
        self.user_mu_latlon = self.user_mu_latlon.to(device=device, dtype=dtype)
        self.user_sigma_raw_km = self.user_sigma_raw_km.to(device=device, dtype=dtype)
        self.user_bin_count = self.user_bin_count.to(device=device)
        return self


def build_st_prior_tables(
    *,
    train_split: Dict[str, List[Dict]],
    user2idx: Dict[str, int],
    poi2idx: Dict[str, int],
    time_bins: int = 24,
    alpha: float = 1.0,
    user_bin_min_count: int = 5,
    seed: int = 42,
    tau_km: float = -1.0,
    tau_sample_cap: int = 200_000,
) -> STPriorTables:
    """
    Build 3 priors from train split:
      - poi_loc[p]
      - poi_time_hist[p, b] => log P(b|p)
      - user_time_center[u, b] => mu(u,b), sigma(u,b)
    All IDs are mapped by user2idx/poi2idx (PAD is assumed 0).
    """
    num_users = int(max(user2idx.values(), default=0) + 1)
    num_pois = int(max(poi2idx.values(), default=0) + 1)
    B = int(time_bins)

    # POI location mean
    poi_sum = torch.zeros((num_pois, 2), dtype=torch.float64)
    poi_cnt = torch.zeros((num_pois,), dtype=torch.int64)

    # POI time histogram counts
    poi_time_cnt = torch.zeros((num_pois, B), dtype=torch.int64)

    # User per-bin mean (first pass)
    user_sum = torch.zeros((num_users, B, 2), dtype=torch.float64)
    user_cnt = torch.zeros((num_users, B), dtype=torch.int64)

    # User overall mean (for backoff)
    user_sum_all = torch.zeros((num_users, 2), dtype=torch.float64)
    user_cnt_all = torch.zeros((num_users,), dtype=torch.int64)

    for user_id, seq in train_split.items():
        u_idx = user2idx.get(user_id, None)
        if u_idx is None:
            continue
        u = int(u_idx)
        for rec in seq:
            poi_id = str(rec["poi_id"])
            p_idx = poi2idx.get(poi_id, None)
            if p_idx is None:
                continue
            p = int(p_idx)
            lat = float(rec["lat"])
            lon = float(rec["lon"])
            tod = float(rec.get("tod", 0.0))
            b = int(min(max(int(tod * B), 0), B - 1))

            poi_sum[p, 0] += lat
            poi_sum[p, 1] += lon
            poi_cnt[p] += 1

            poi_time_cnt[p, b] += 1

            user_sum[u, b, 0] += lat
            user_sum[u, b, 1] += lon
            user_cnt[u, b] += 1

            user_sum_all[u, 0] += lat
            user_sum_all[u, 1] += lon
            user_cnt_all[u] += 1

    poi_latlon = torch.zeros((num_pois, 2), dtype=torch.float32)
    valid_poi = poi_cnt > 0
    poi_latlon[valid_poi, :] = (poi_sum[valid_poi, :] / poi_cnt[valid_poi].unsqueeze(1)).to(dtype=torch.float32)

    # log P(bin|poi) with smoothing
    denom = poi_time_cnt.sum(dim=1, keepdim=True).to(dtype=torch.float64) + float(alpha) * float(B)
    prob = (poi_time_cnt.to(dtype=torch.float64) + float(alpha)) / torch.clamp(denom, min=1.0)
    poi_logp_time = torch.log(torch.clamp(prob.to(dtype=torch.float32), min=1e-12))

    # user mean per bin, plus overall mean for backoff
    user_mu = torch.zeros((num_users, B, 2), dtype=torch.float64)
    mask = user_cnt > 0
    user_mu[mask, :] = user_sum[mask, :] / user_cnt[mask].unsqueeze(1).to(dtype=torch.float64)

    user_mu_all = torch.zeros((num_users, 2), dtype=torch.float64)
    mask_all = user_cnt_all > 0
    user_mu_all[mask_all, :] = user_sum_all[mask_all, :] / user_cnt_all[mask_all].unsqueeze(1).to(dtype=torch.float64)

    # user sigma (second pass): mean haversine distance to mu(u,b)
    user_sigma_sum = torch.zeros((num_users, B), dtype=torch.float64)
    user_sigma_sum_all = torch.zeros((num_users,), dtype=torch.float64)

    for user_id, seq in train_split.items():
        u_idx = user2idx.get(user_id, None)
        if u_idx is None:
            continue
        u = int(u_idx)
        for rec in seq:
            lat = float(rec["lat"])
            lon = float(rec["lon"])
            tod = float(rec.get("tod", 0.0))
            b = int(min(max(int(tod * B), 0), B - 1))
            if int(user_cnt[u, b].item()) > 0:
                mu_lat = float(user_mu[u, b, 0].item())
                mu_lon = float(user_mu[u, b, 1].item())
            else:
                mu_lat = float(user_mu_all[u, 0].item())
                mu_lon = float(user_mu_all[u, 1].item())

            d = float(
                _haversine_km(
                    torch.tensor(lat),
                    torch.tensor(lon),
                    torch.tensor(mu_lat),
                    torch.tensor(mu_lon),
                ).item()
            )
            user_sigma_sum[u, b] += d
            user_sigma_sum_all[u] += d

    user_sigma_raw = torch.zeros((num_users, B), dtype=torch.float32)
    for u in range(num_users):
        total_all = int(user_cnt_all[u].item())
        sigma_all = float(user_sigma_sum_all[u].item() / total_all) if total_all > 0 else 0.0
        for b in range(B):
            c = int(user_cnt[u, b].item())
            if c > 0:
                user_sigma_raw[u, b] = float(user_sigma_sum[u, b].item() / c)
            else:
                user_sigma_raw[u, b] = float(sigma_all)

    # apply backoff for mu based on user_bin_min_count
    user_mu_backoff = user_mu.clone()
    for u in range(num_users):
        mu_all_lat = float(user_mu_all[u, 0].item())
        mu_all_lon = float(user_mu_all[u, 1].item())
        for b in range(B):
            if int(user_cnt[u, b].item()) < int(user_bin_min_count):
                user_mu_backoff[u, b, 0] = mu_all_lat
                user_mu_backoff[u, b, 1] = mu_all_lon

    if float(tau_km) > 0.0:
        tau_p75 = float(tau_km)
    else:
        tau_p75 = _estimate_tau_p75_km_from_train(
            train_split,
            sample_cap=int(tau_sample_cap),
            seed=int(seed),
        )

    return STPriorTables(
        time_bins=B,
        alpha=float(alpha),
        user_bin_min_count=int(user_bin_min_count),
        poi_latlon=poi_latlon,
        poi_logp_time=poi_logp_time,
        user_mu_latlon=user_mu_backoff.to(dtype=torch.float32),
        user_sigma_raw_km=user_sigma_raw,
        user_bin_count=user_cnt.to(dtype=torch.int32),
        tau_p75_km=float(tau_p75),
    )


@torch.no_grad()
def rerank_topk(
    *,
    logits: torch.Tensor,  # (B, num_pois)
    user_idx: torch.Tensor,  # (B,)
    query_tod: torch.Tensor,  # (B,)
    last_lat: torch.Tensor,  # (B,)
    last_lon: torch.Tensor,  # (B,)
    has_last: torch.Tensor,  # (B,) bool/int
    tables: STPriorTables,
    candidate_k: int,
    lambda_time: float,
    lambda_user: float,
    lambda_dist: float,
    sigma_min_km: float,
    tau_km: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (candidate_idx, rescored_scores) where rescoring is done within candidate set only.
    """
    B, num_pois = logits.shape
    K = int(min(int(candidate_k), int(num_pois)))
    cand_scores, cand_idx = torch.topk(logits, k=K, dim=1)

    b = _tod_to_bin(query_tod, tables.time_bins)  # (B,)

    poi_lat = tables.poi_latlon[cand_idx, 0]
    poi_lon = tables.poi_latlon[cand_idx, 1]

    # POI time prior: log P(bin|poi)
    s_time = tables.poi_logp_time[cand_idx, b.view(-1, 1).expand(-1, K)]

    # user-time spatial prior: -dist(p, mu(u,b)) / sigma(u,b)
    mu = tables.user_mu_latlon[user_idx, b, :]  # (B, 2)
    mu_lat = mu[:, 0].unsqueeze(1)
    mu_lon = mu[:, 1].unsqueeze(1)
    dist_user = _haversine_km(poi_lat, poi_lon, mu_lat, mu_lon)  # (B, K)
    sigma_raw = tables.user_sigma_raw_km[user_idx, b]  # (B,)
    sigma = torch.clamp(sigma_raw, min=float(sigma_min_km)).unsqueeze(1)
    s_user = -dist_user / sigma

    # last-step distance prior: -dist(p, last) / tau
    last_lat_b = last_lat.unsqueeze(1)
    last_lon_b = last_lon.unsqueeze(1)
    dist_last = _haversine_km(poi_lat, poi_lon, last_lat_b, last_lon_b)  # (B, K)
    s_dist = -dist_last / float(tau_km)
    has_last_f = has_last.to(dtype=logits.dtype).view(-1, 1)
    s_dist = s_dist * has_last_f

    rescored = cand_scores + float(lambda_time) * s_time + float(lambda_user) * s_user + float(lambda_dist) * s_dist
    return cand_idx, rescored

