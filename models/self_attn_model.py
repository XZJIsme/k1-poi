from __future__ import annotations

from typing import Optional, Tuple, List, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelfAttnBlock(nn.Module):
    """
    Transformer Block with Self-Attention only, no FFN

    For the window-input setting:
    - Input sequence length is fixed at 1+k (user token + k historical checkin tokens)
    - No causal mask needed (everything in the window is "past")
    - Padding is masked via key_padding_mask (padding tokens cannot be attended to as key/value)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float, norm_first: bool) -> None:
        super().__init__()
        self.norm_first = bool(norm_first)
        self.attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(nhead),
            dropout=float(dropout),
            batch_first=True,  # (B, T, C)
        )
        self.norm = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else None
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, d_model)
            key_padding_mask: (B, T) bool, True means "masked / ignore" (as key/value).
        Returns:
            x: (B, T, d_model)
            attn_w: None or attention weights (shape depends on PyTorch version/settings)
        """
        attn_w = None
        if self.norm_first:
            x_norm = self.norm(x)
            if need_weights:
                try:
                    attn_out, attn_w = self.attn(
                        x_norm,
                        x_norm,
                        x_norm,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                        average_attn_weights=bool(average_attn_weights),
                    )
                except TypeError:
                    attn_out, attn_w = self.attn(
                        x_norm,
                        x_norm,
                        x_norm,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                    )
            else:
                attn_out, _ = self.attn(
                    x_norm,
                    x_norm,
                    x_norm,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
            if self.dropout is not None:
                attn_out = self.dropout(attn_out)
            x = x + attn_out
            self.last_attn_weights = attn_w
            return x, attn_w

        if need_weights:
            try:
                attn_out, attn_w = self.attn(
                    x,
                    x,
                    x,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=bool(average_attn_weights),
                )
            except TypeError:
                attn_out, attn_w = self.attn(
                    x,
                    x,
                    x,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                )
        else:
            attn_out, _ = self.attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
        if self.dropout is not None:
            attn_out = self.dropout(attn_out)
        x = x + attn_out
        x = self.norm(x)
        self.last_attn_weights = attn_w
        return x, attn_w


class _ResidualTwoLayerMLP(nn.Module):
    """
    Two-layer MLP with residual on the first layer output:

      h1 = Linear(x -> D)
      h2 = Linear(GELU(h1) -> D)
      y  = h2 + h1

    Works for both (B, D_in) and (B, K, D_in) inputs.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(in_dim), int(out_dim))
        self.fc2 = nn.Linear(int(out_dim), int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.fc1(x)
        h2 = self.fc2(F.gelu(h1))
        return h2 + h1


class NextPOIWindowSelfAttn(nn.Module):
    """
    Window-input Self-Attention model without teacher forcing.

    Inputs:
      - user_id: (B,)
      - window: k historical check-ins with left padding. Each check-in is
        represented by poi emb || (cat emb) || (tod-slot emb) || (geo-cell emb),
        then linearly projected to d_model.
      - sequence tokens: [user_token] + [k check-in tokens]

    Output:
      - logits: (B, num_pois)
        classifier_token_position selects which token output is fed into the output head.
    """

    def __init__(
        self,
        *,
        num_users: int,
        num_pois: int,
        num_cats: int,
        window_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        embedding_dropout: float = 0.0,
        output_dropout: float = 0.0,
        norm_first: bool = True,
        poi_emb_dim: int = 128,
        cat_emb_dim: int = 32,
        user_emb_dim: Optional[int] = None,
        # POI embedding contrastive learning (default: off)
        use_poi_embedding_contrastive_learning: bool = False,
        poi_embedding_contrastive_learning_proj_dim: int = 256,
        use_mlp_for_cl_instead_of_simple_proj: bool = False,
        use_tod_slot_embedding: bool = False,
        tod_slot_scales: Optional[List[int]] = None,
        tod_slot_emb_dim: int = 16,
        use_geo_cell_embedding: bool = False,
        geo_cell_sizes_m: Optional[List[int]] = None,
        geo_cell_grid_w: Optional[List[int]] = None,
        geo_cell_grid_h: Optional[List[int]] = None,
        geo_cell_emb_dim: int = 16,
        use_user_emb: bool = True,
        use_cat_emb: bool = True,
        use_positional_encoding: bool = True,
        classifier_token_position: str = "first",
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if int(window_size) < 0:
            raise ValueError(f"window_size must be >=0, got {window_size}")
        if int(d_model) % int(nhead) != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.pad_idx = int(pad_idx)
        self.use_user_emb = bool(use_user_emb)
        self.use_cat_emb = bool(use_cat_emb)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.use_tod_slot_embedding = bool(use_tod_slot_embedding)
        self.tod_slot_scales = [int(x) for x in (tod_slot_scales or [])]
        self.tod_slot_emb_dim = int(tod_slot_emb_dim)
        self.use_geo_cell_embedding = bool(use_geo_cell_embedding)
        self.geo_cell_sizes_m = [int(x) for x in (geo_cell_sizes_m or [])]
        self.geo_cell_grid_w = [int(x) for x in (geo_cell_grid_w or [])]
        self.geo_cell_grid_h = [int(x) for x in (geo_cell_grid_h or [])]
        self.geo_cell_emb_dim = int(geo_cell_emb_dim)
        if self.use_tod_slot_embedding:
            if not self.tod_slot_scales:
                raise ValueError("tod_slot_scales must be non-empty when use_tod_slot_embedding=True")
            for slots_per_day in self.tod_slot_scales:
                if slots_per_day <= 0:
                    raise ValueError(f"Invalid slots_per_day in tod_slot_scales: {slots_per_day}")
            if self.tod_slot_emb_dim <= 0:
                raise ValueError(f"tod_slot_emb_dim must be > 0, got {tod_slot_emb_dim}")
        if self.use_geo_cell_embedding:
            if not self.geo_cell_sizes_m:
                raise ValueError("geo_cell_sizes_m must be non-empty when use_geo_cell_embedding=True")
            if not self.geo_cell_grid_w or not self.geo_cell_grid_h:
                raise ValueError("geo_cell_grid_w/geo_cell_grid_h must be provided when use_geo_cell_embedding=True")
            if len(self.geo_cell_grid_w) != len(self.geo_cell_sizes_m) or len(self.geo_cell_grid_h) != len(
                self.geo_cell_sizes_m
            ):
                raise ValueError(
                    "geo_cell_grid_w/geo_cell_grid_h must have the same length as geo_cell_sizes_m "
                    f"(sizes={len(self.geo_cell_sizes_m)} w={len(self.geo_cell_grid_w)} h={len(self.geo_cell_grid_h)})"
                )
            if self.geo_cell_emb_dim <= 0:
                raise ValueError(f"geo_cell_emb_dim must be > 0, got {geo_cell_emb_dim}")
            for w in self.geo_cell_grid_w:
                if int(w) <= 0:
                    raise ValueError(f"Invalid geo_cell_grid_w: {w}")
            for h in self.geo_cell_grid_h:
                if int(h) <= 0:
                    raise ValueError(f"Invalid geo_cell_grid_h: {h}")
        position = str(classifier_token_position).strip().lower()
        if position not in {"first", "last", "mean"}:
            raise ValueError(
                f"classifier_token_position must be one of ['first','last','mean'], got {classifier_token_position!r}"
            )
        self.classifier_token_position = position

        resolved_user_emb_dim = int(d_model) if user_emb_dim is None else int(user_emb_dim)
        if resolved_user_emb_dim <= 0:
            raise ValueError(f"user_emb_dim must be > 0, got {user_emb_dim}")

        # Embed the user token to user_emb_dim, then project it to d_model for self-attention.
        self.user_embedding = nn.Embedding(int(num_users), int(resolved_user_emb_dim), padding_idx=int(pad_idx))
        self.user_proj = nn.Linear(int(resolved_user_emb_dim), int(d_model))

        self.poi_embedding = nn.Embedding(int(num_pois), int(poi_emb_dim), padding_idx=int(pad_idx))
        self.cat_embedding = (
            nn.Embedding(int(num_cats), int(cat_emb_dim), padding_idx=int(pad_idx)) if self.use_cat_emb else None
        )
        self.tod_slot_embeddings = (
            nn.ModuleList(
                [
                    nn.Embedding(int(slots_per_day) + 1, int(self.tod_slot_emb_dim), padding_idx=int(pad_idx))
                    for slots_per_day in self.tod_slot_scales
                ]
            )
            if self.use_tod_slot_embedding
            else None
        )
        self.geo_x_embeddings = (
            nn.ModuleList(
                [
                    nn.Embedding(int(w) + 1, int(self.geo_cell_emb_dim), padding_idx=int(pad_idx))
                    for w in self.geo_cell_grid_w
                ]
            )
            if self.use_geo_cell_embedding
            else None
        )
        self.geo_y_embeddings = (
            nn.ModuleList(
                [
                    nn.Embedding(int(h) + 1, int(self.geo_cell_emb_dim), padding_idx=int(pad_idx))
                    for h in self.geo_cell_grid_h
                ]
            )
            if self.use_geo_cell_embedding
            else None
        )

        checkin_feature_dim = int(poi_emb_dim)
        if self.use_cat_emb:
            checkin_feature_dim += int(cat_emb_dim)
        if self.use_tod_slot_embedding:
            checkin_feature_dim += len(self.tod_slot_scales) * int(self.tod_slot_emb_dim)
        if self.use_geo_cell_embedding:
            checkin_feature_dim += len(self.geo_cell_sizes_m) * int(self.geo_cell_emb_dim)

        self.checkin_proj = nn.Linear(int(checkin_feature_dim), int(d_model))
        self.embedding_dropout = nn.Dropout(float(embedding_dropout)) if float(embedding_dropout) > 0 else None

        if self.use_positional_encoding:
            # learnable PE, default all zeros => "no PE" at init
            self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + int(window_size), int(d_model)))
        else:
            self.pos_embedding = None

        # Debug helper: set to True (e.g., in Debug Console) to store per-layer attention weights.
        self.record_attn_weights: bool = False
        self.layers = nn.ModuleList(
            [
                _SelfAttnBlock(
                    d_model=int(d_model),
                    nhead=int(nhead),
                    dropout=float(dropout),
                    norm_first=bool(norm_first),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_dropout = nn.Dropout(float(output_dropout)) if float(output_dropout) > 0 else None
        self.output_layer = nn.Linear(int(d_model), int(num_pois))

        # POI embedding contrastive learning modules (optional)
        self.use_poi_embedding_contrastive_learning = bool(use_poi_embedding_contrastive_learning)
        self.poi_embedding_contrastive_learning_proj_dim = int(poi_embedding_contrastive_learning_proj_dim)
        self.use_mlp_for_cl_instead_of_simple_proj = bool(use_mlp_for_cl_instead_of_simple_proj)
        if self.use_poi_embedding_contrastive_learning:
            if self.poi_embedding_contrastive_learning_proj_dim <= 0:
                raise ValueError(
                    "poi_embedding_contrastive_learning_proj_dim must be > 0, got "
                    f"{poi_embedding_contrastive_learning_proj_dim}"
                )
            # CL uses projected user representation vs projected candidate POI embeddings (cosine/dot + temperature).
            if self.use_mlp_for_cl_instead_of_simple_proj:
                self.poi_ecl_user_proj = _ResidualTwoLayerMLP(int(d_model), self.poi_embedding_contrastive_learning_proj_dim)
                self.poi_ecl_poi_proj = _ResidualTwoLayerMLP(int(poi_emb_dim), self.poi_embedding_contrastive_learning_proj_dim)
            else:
                self.poi_ecl_user_proj = nn.Linear(int(d_model), self.poi_embedding_contrastive_learning_proj_dim)
                self.poi_ecl_poi_proj = nn.Linear(int(poi_emb_dim), self.poi_embedding_contrastive_learning_proj_dim)
        else:
            self.poi_ecl_user_proj = None
            self.poi_ecl_poi_proj = None

    def compute_poi_embedding_contrastive_learning_loss(
        self,
        *,
        user_out: torch.Tensor,  # (B, d_model)
        logits_poi: torch.Tensor,  # (B, num_pois)
        target_poi: torch.Tensor,  # (B,)
        top_k_candidates: int,
        temperature: float,
        normalize_embeddings: bool,
        force_label_into_candidates_strategy: str,
        negative_source: str = "topk",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not bool(self.use_poi_embedding_contrastive_learning):
            raise RuntimeError("compute_poi_embedding_contrastive_learning_loss called but use_poi_embedding_contrastive_learning=False")
        if self.poi_ecl_user_proj is None or self.poi_ecl_poi_proj is None:
            raise RuntimeError("Internal error: POI ECL modules are not initialized.")

        if user_out.dim() != 2:
            raise ValueError(f"user_out must be (B,d), got {tuple(user_out.shape)}")
        if logits_poi.dim() != 2:
            raise ValueError(f"logits_poi must be (B,V), got {tuple(logits_poi.shape)}")
        if target_poi.dim() != 1:
            raise ValueError(f"target_poi must be (B,), got {tuple(target_poi.shape)}")
        if int(logits_poi.size(0)) != int(user_out.size(0)) or int(target_poi.size(0)) != int(user_out.size(0)):
            raise ValueError(
                f"Batch size mismatch: user_out={tuple(user_out.shape)} logits_poi={tuple(logits_poi.shape)} target_poi={tuple(target_poi.shape)}"
            )

        B, V = logits_poi.shape
        k = int(top_k_candidates)
        if k <= 0:
            raise ValueError(f"top_k_candidates must be > 0, got {top_k_candidates}")
        k = min(k, int(V))
        tau = float(temperature)
        if not (tau > 0.0):
            raise ValueError(f"temperature must be > 0, got {temperature}")

        strat = str(force_label_into_candidates_strategy).strip().lower()
        if strat not in {"replace_lowest", "replace_highest"}:
            raise ValueError(
                "force_label_into_candidates_strategy must be one of {'replace_lowest','replace_highest'}, got "
                f"{force_label_into_candidates_strategy!r}"
            )

        neg_src = str(negative_source).strip().lower()
        if neg_src not in {"topk", "random", "mix"}:
            raise ValueError(f"negative_source must be one of {{topk,random,mix}}, got {negative_source!r}")

        if neg_src == "topk":
            cand_ids = torch.topk(logits_poi, k=k, dim=1).indices  # (B, K)
        elif neg_src == "random":
            # Note: may contain duplicates; intended as a simple alternative negative sampler.
            low = 1 if int(self.pad_idx) == 0 else 0
            cand_ids = torch.randint(low=low, high=int(V), size=(int(B), int(k)), device=logits_poi.device)  # (B,K)
            if int(self.pad_idx) == 0:
                # already excluded by low=1
                pass
            else:
                cand_ids = cand_ids.clone()
                cand_ids[cand_ids == int(self.pad_idx)] = 0 if int(self.pad_idx) != 0 else 1
        else:
            # mix: half topk + half random (may contain duplicates in random part)
            k_top = max(1, int(k) // 2)
            k_rand = int(k) - int(k_top)
            top_ids = torch.topk(logits_poi, k=k_top, dim=1).indices  # (B,k_top)
            low = 1 if int(self.pad_idx) == 0 else 0
            rand_ids = torch.randint(low=low, high=int(V), size=(int(B), int(k_rand)), device=logits_poi.device)  # (B,k_rand)
            cand_ids = torch.cat([top_ids, rand_ids], dim=1)  # (B,K)

        target = target_poi.to(device=cand_ids.device)
        in_cand = cand_ids.eq(target.unsqueeze(1))
        missing = ~in_cand.any(dim=1)
        missing_rate = float(missing.float().mean().item()) if B > 0 else 0.0
        if bool(missing.any()):
            replace_pos = (k - 1) if strat == "replace_lowest" else 0
            cand_ids = cand_ids.clone()
            cand_ids[missing, replace_pos] = target[missing]
            in_cand = cand_ids.eq(target.unsqueeze(1))

        # pos idx in [0..K-1]
        pos_idx = in_cand.to(dtype=torch.float32).argmax(dim=1).to(dtype=torch.long)  # (B,)

        cand_poi_emb = self.poi_embedding(cand_ids)  # (B,K,poi_emb_dim)
        u = self.poi_ecl_user_proj(user_out)  # (B,D)
        p = self.poi_ecl_poi_proj(cand_poi_emb)  # (B,K,D)

        if bool(normalize_embeddings):
            u = F.normalize(u, dim=1)
            p = F.normalize(p, dim=2)

        cl_logits = torch.einsum("bd,bkd->bk", u, p) / tau  # (B,K)
        loss = F.cross_entropy(cl_logits, pos_idx)
        stats = {"missing_label_rate": missing_rate, "top_k_candidates": float(k)}
        return loss, stats

    def forward(
        self,
        user_ids: torch.Tensor,
        window_poi_ids: torch.Tensor,
        window_cat_ids: Optional[torch.Tensor],
        window_tod_slot_ids: Optional[torch.Tensor] = None,
        window_geo_cell_x_ids: Optional[torch.Tensor] = None,
        window_geo_cell_y_ids: Optional[torch.Tensor] = None,
        *,
        on_attn: Optional[Callable[[int, torch.Tensor], None]] = None,
        return_attn: bool = False,
        return_user_out: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]]:
        """
        Args:
            user_ids: (B,)
            window_poi_ids: (B, k)
            window_cat_ids: (B, k) or None
            window_geo_cell_x_ids/y_ids: (B, k, num_scales) long or None
        Returns:
            logits: (B, num_pois)
        """
        if window_poi_ids.dim() != 2:
            raise ValueError(f"window_poi_ids must be (B,k), got {tuple(window_poi_ids.shape)}")
        B, k = window_poi_ids.size()
        if k != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got k={k}")

        user_x = self.user_proj(self.user_embedding(user_ids)).unsqueeze(1)  # (B, 1, d_model)
        if not self.use_user_emb:
            user_x = user_x * 0.0

        feats = [self.poi_embedding(window_poi_ids)]
        if self.use_cat_emb:
            if window_cat_ids is None:
                raise ValueError("window_cat_ids is required when use_cat_emb=True")
            feats.append(self.cat_embedding(window_cat_ids))  # type: ignore[arg-type]
        if self.use_tod_slot_embedding:
            if window_tod_slot_ids is None:
                raise ValueError("window_tod_slot_ids is required when use_tod_slot_embedding=True")
            if window_tod_slot_ids.dim() != 3:
                raise ValueError(
                    f"window_tod_slot_ids must be (B,k,num_scales), got {tuple(window_tod_slot_ids.shape)}"
                )
            if window_tod_slot_ids.size(0) != B or window_tod_slot_ids.size(1) != k:
                raise ValueError(
                    f"window_tod_slot_ids must match (B,k)=({B},{k}), got {tuple(window_tod_slot_ids.shape)}"
                )
            if window_tod_slot_ids.size(2) != len(self.tod_slot_scales):
                raise ValueError(
                    f"window_tod_slot_ids last dim must be num_scales={len(self.tod_slot_scales)}, got {window_tod_slot_ids.size(2)}"
                )
            if self.tod_slot_embeddings is None:
                raise RuntimeError("Internal error: tod_slot_embeddings is None when use_tod_slot_embedding=True")
            for scale_index, emb in enumerate(self.tod_slot_embeddings):
                feats.append(emb(window_tod_slot_ids[:, :, scale_index]))
        if self.use_geo_cell_embedding:
            if window_geo_cell_x_ids is None or window_geo_cell_y_ids is None:
                raise ValueError("window_geo_cell_x_ids/window_geo_cell_y_ids are required when use_geo_cell_embedding=True")
            if window_geo_cell_x_ids.dim() != 3 or window_geo_cell_y_ids.dim() != 3:
                raise ValueError(
                    "window_geo_cell_x_ids/window_geo_cell_y_ids must be (B,k,num_scales), got "
                    f"x={tuple(window_geo_cell_x_ids.shape)} y={tuple(window_geo_cell_y_ids.shape)}"
                )
            if window_geo_cell_x_ids.size(0) != B or window_geo_cell_x_ids.size(1) != k:
                raise ValueError(
                    f"window_geo_cell_x_ids must match (B,k)=({B},{k}), got {tuple(window_geo_cell_x_ids.shape)}"
                )
            if window_geo_cell_y_ids.size(0) != B or window_geo_cell_y_ids.size(1) != k:
                raise ValueError(
                    f"window_geo_cell_y_ids must match (B,k)=({B},{k}), got {tuple(window_geo_cell_y_ids.shape)}"
                )
            if window_geo_cell_x_ids.size(2) != len(self.geo_cell_sizes_m) or window_geo_cell_y_ids.size(2) != len(
                self.geo_cell_sizes_m
            ):
                raise ValueError(
                    f"geo ids last dim must be num_scales={len(self.geo_cell_sizes_m)}, got "
                    f"x={window_geo_cell_x_ids.size(2)} y={window_geo_cell_y_ids.size(2)}"
                )
            if self.geo_x_embeddings is None or self.geo_y_embeddings is None:
                raise RuntimeError("Internal error: geo embeddings are None when use_geo_cell_embedding=True")
            for scale_index, (x_emb, y_emb) in enumerate(zip(self.geo_x_embeddings, self.geo_y_embeddings)):
                geo_emb_s = x_emb(window_geo_cell_x_ids[:, :, scale_index]) + y_emb(window_geo_cell_y_ids[:, :, scale_index])
                feats.append(geo_emb_s)

        checkin_feat = torch.cat(feats, dim=-1)  # (B, k, checkin_feature_dim)
        checkin_x = self.checkin_proj(checkin_feat)  # (B, k, d_model)

        x = torch.cat([user_x, checkin_x], dim=1)  # (B, 1+k, d_model)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x)

        # padding mask: only mask checkin tokens whose poi_id == pad_idx; never mask user token
        checkin_pad = window_poi_ids.eq(int(self.pad_idx))
        user_not_masked = torch.zeros((B, 1), device=checkin_pad.device, dtype=torch.bool)
        key_padding_mask = torch.cat([user_not_masked, checkin_pad], dim=1)  # (B, 1+k)

        need_weights = bool(return_attn) or (on_attn is not None) or bool(getattr(self, "record_attn_weights", False))
        attn_by_layer: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            x, attn_w = layer(
                x,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                average_attn_weights=False,
            )
            if need_weights and attn_w is not None:
                if on_attn is not None:
                    on_attn(layer_idx, attn_w)
                if return_attn:
                    attn_by_layer.append(attn_w)

        if self.classifier_token_position == "first":
            user_out = x[:, 0, :]  # (B, d_model)
        elif self.classifier_token_position == "last":
            if key_padding_mask is None:
                user_out = x[:, -1, :]
            else:
                valid = ~key_padding_mask
                positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(valid)
                last_idx = (positions * valid.long()).amax(dim=1)
                user_out = x[torch.arange(B, device=x.device), last_idx, :]
        elif self.classifier_token_position == "mean":
            valid = (~key_padding_mask).to(dtype=x.dtype).unsqueeze(-1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            user_out = (x * valid).sum(dim=1) / denom
        else:
            raise RuntimeError(f"Unhandled classifier_token_position: {self.classifier_token_position}")
        if self.output_dropout is not None:
            user_out = self.output_dropout(user_out)
        logits = self.output_layer(user_out)  # (B, num_pois)

        # Never predict pad token (idx=pad_idx) as a real POI
        if logits.size(1) > int(self.pad_idx):
            logits[:, int(self.pad_idx)] = -1e9

        if return_attn:
            if bool(return_user_out):
                return logits, attn_by_layer, user_out
            return logits, attn_by_layer
        if bool(return_user_out):
            return logits, user_out
        return logits
