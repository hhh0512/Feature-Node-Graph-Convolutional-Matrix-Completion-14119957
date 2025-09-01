#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funk-SVD baseline for MovieLens 100K with an optional *feature-augmented* mode.

Baseline (default, --use_features 0):
    r_hat = mu + b_u + b_i + P_u^T Q_i

Feature-augmented (--use_features 1):
    r_hat = mu + b_u + b_i + P_u^T Q_i + w_user^T x_u + w_item^T z_i
where x_u, z_i are user/item feature vectors built from u.user / u.item.
Optimization via SGD with L2 regularization.

Usage:
    python svd_ml100k_feat.py --data_dir /path/to/ml-100k --split 1 \
        --factors 64 --lr 0.01 --reg 0.02 --epochs 20 --seed 42 \
        --use_features 1 --feat_reg 0.01
"""
import argparse
import csv
import math
import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np


def load_ratings(path: str) -> List[Tuple[int, int, float]]:
    ratings: List[Tuple[int, int, float]] = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if not line.strip():
                continue
            u, i, r, _ = line.strip().split("\t")
            ratings.append((int(u), int(i), float(r)))
    return ratings


def load_users(path: str) -> Dict[int, Dict]:
    users = {}
    with open(path, "r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row:
                continue
            uid = int(row[0])
            users[uid] = {
                "age": int(row[1]),
                "gender": row[2],
                "occupation": row[3],
                "zip": row[4],
            }
    return users


def load_items(path: str) -> Dict[int, Dict]:
    items = {}
    with open(path, "r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row:
                continue
            mid = int(row[0])
            title = row[1]
            release_date = row[2]  # e.g., "01-Jan-1995"
            imdb_url = row[4]
            genres = list(map(int, row[5:]))
            # Parse release year if possible
            year = None
            if release_date and release_date.strip() and release_date != "":
                try:
                    year = int(release_date[-4:])
                except Exception:
                    year = None
            items[mid] = {
                "title": title,
                "release_year": year,
                "imdb_url": imdb_url,
                "genres": genres,
            }
    return items


def remap_ids(
    ratings: List[Tuple[int, int, float]]
) -> Tuple[List[Tuple[int, int, float]], Dict[int, int], Dict[int, int]]:
    users = sorted({u for u, _, _ in ratings})
    items = sorted({i for _, i, _ in ratings})
    u_map = {u: idx for idx, u in enumerate(users)}
    i_map = {i: idx for idx, i in enumerate(items)}
    remapped = [(u_map[u], i_map[i], r) for u, i, r in ratings]
    return remapped, u_map, i_map


def align_with_maps(
    ratings: List[Tuple[int, int, float]],
    u_map: Dict[int, int],
    i_map: Dict[int, int],
) -> List[Tuple[int, int, float]]:
    aligned = []
    for u, i, r in ratings:
        if u in u_map and i in i_map:
            aligned.append((u_map[u], i_map[i], r))
    return aligned


class FeatureBuilder:
    """Builds aligned user/item feature matrices from u.user and u.item dicts."""

    def __init__(self, users: Dict[int, Dict], items: Dict[int, Dict]):
        self.users_raw = users
        self.items_raw = items
        self.occupations = sorted({u["occupation"] for u in users.values()}) if users else []
        self.occ_index = {occ: k for k, occ in enumerate(self.occupations)}
        self.gender_vals = ["M", "F"]
        self.gender_index = {g: k for k, g in enumerate(self.gender_vals)}
        self.num_genres = 19
        self.user_age_mean = None
        self.user_age_std = None
        self.item_year_mean = None
        self.item_year_std = None

    @staticmethod
    def _zscore(x_list: List[float]) -> Tuple[Optional[float], Optional[float]]:
        vals = [x for x in x_list if x is not None]
        if not vals:
            return None, None
        mean = float(np.mean(vals))
        std = float(np.std(vals)) or 1.0
        return mean, std

    def fit_stats(self):
        ages = [u["age"] for u in self.users_raw.values()] if self.users_raw else []
        self.user_age_mean, self.user_age_std = self._zscore(ages)
        years = [it["release_year"] for it in self.items_raw.values() if it["release_year"] is not None] if self.items_raw else []
        self.item_year_mean, self.item_year_std = self._zscore(years)

    def user_vector(self, raw_uid: int) -> np.ndarray:
        if not self.users_raw:
            return np.zeros(0, dtype=np.float32)
        u = self.users_raw.get(raw_uid, None)
        if u is None:
            return np.zeros(self.user_dim, dtype=np.float32)
        feats = []
        if self.user_age_mean is not None and self.user_age_std is not None:
            age_z = (u["age"] - self.user_age_mean) / self.user_age_std
            feats.append(age_z)
        else:
            feats.append(0.0)
        g_vec = [0.0] * len(self.gender_vals)
        g = u.get("gender", None)
        if g in self.gender_index:
            g_vec[self.gender_index[g]] = 1.0
        feats.extend(g_vec)
        occ_vec = [0.0] * len(self.occupations)
        occ = u.get("occupation", None)
        if occ in self.occ_index:
            occ_vec[self.occ_index[occ]] = 1.0
        feats.extend(occ_vec)
        return np.array(feats, dtype=np.float32)

    def item_vector(self, raw_iid: int) -> np.ndarray:
        if not self.items_raw:
            return np.zeros(0, dtype=np.float32)
        it = self.items_raw.get(raw_iid, None)
        if it is None:
            return np.zeros(self.item_dim, dtype=np.float32)
        feats = []
        genres = it.get("genres", [])
        if genres and len(genres) == self.num_genres:
            feats.extend(genres)
        else:
            feats.extend([0.0] * self.num_genres)
        if it.get("release_year", None) is not None and self.item_year_mean is not None:
            year_z = (it["release_year"] - self.item_year_mean) / self.item_year_std
            feats.append(year_z)
        else:
            feats.append(0.0)
        return np.array(feats, dtype=np.float32)

    @property
    def user_dim(self) -> int:
        return 1 + len(self.gender_vals) + len(self.occupations)

    @property
    def item_dim(self) -> int:
        return self.num_genres + 1


class SVDwithFeatures:
    def __init__(self, num_users, num_items, n_factors=64, lr=0.01, reg=0.02, seed=42,
                 use_features=False, user_feat_dim=0, item_feat_dim=0, feat_reg=0.01):
        rng = np.random.default_rng(seed)
        self.mu = 0.0
        self.bu = np.zeros(num_users, dtype=np.float32)
        self.bi = np.zeros(num_items, dtype=np.float32)
        self.P = 0.1 * rng.standard_normal((num_users, n_factors)).astype(np.float32)
        self.Q = 0.1 * rng.standard_normal((num_items, n_factors)).astype(np.float32)
        self.lr = lr
        self.reg = reg
        self.use_features = use_features
        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.feat_reg = feat_reg
        if self.use_features:
            self.Wu = np.zeros(self.user_feat_dim, dtype=np.float32) if self.user_feat_dim > 0 else None
            self.Wi = np.zeros(self.item_feat_dim, dtype=np.float32) if self.item_feat_dim > 0 else None
        else:
            self.Wu = None
            self.Wi = None

    def predict(self, u, i, xu=None, zi=None):
        base = self.mu + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))
        if self.use_features:
            add_u = float(np.dot(self.Wu, xu)) if (self.Wu is not None and xu is not None) else 0.0
            add_i = float(np.dot(self.Wi, zi)) if (self.Wi is not None and zi is not None) else 0.0
            return base + add_u + add_i
        return base

    def fit(self, train, n_epochs=20, shuffle=True, verbose=True,
            u_inv_map=None, i_inv_map=None, feat_builder=None):
        if not train:
            raise ValueError("Empty training set.")
        self.mu = float(np.mean([r for _, _, r in train]))
        if self.use_features and feat_builder is not None and u_inv_map is not None and i_inv_map is not None:
            Ux = np.zeros((len(self.bu), feat_builder.user_dim), dtype=np.float32)
            Iz = np.zeros((len(self.bi), feat_builder.item_dim), dtype=np.float32)
            for raw_u, idx_u in u_inv_map.items():
                Ux[idx_u] = feat_builder.user_vector(raw_u)
            for raw_i, idx_i in i_inv_map.items():
                Iz[idx_i] = feat_builder.item_vector(raw_i)
        else:
            Ux, Iz = None, None
        for epoch in range(1, n_epochs + 1):
            if shuffle:
                random.shuffle(train)
            sse = 0.0
            for u, i, r in train:
                xu = Ux[u] if (self.use_features and Ux is not None) else None
                zi = Iz[i] if (self.use_features and Iz is not None) else None
                pred = self.predict(u, i, xu, zi)
                err = r - pred
                sse += err * err
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                Pu = self.P[u]
                Qi = self.Q[i]
                self.P[u] += self.lr * (err * Qi - self.reg * Pu)
                self.Q[i] += self.lr * (err * Pu - self.reg * Qi)
                if self.use_features and xu is not None and self.Wu is not None:
                    self.Wu += self.lr * (err * xu - self.feat_reg * self.Wu)
                if self.use_features and zi is not None and self.Wi is not None:
                    self.Wi += self.lr * (err * zi - self.feat_reg * self.Wi)
            rmse = math.sqrt(sse / len(train))
            if verbose:
                print(f"Epoch {epoch:02d}/{n_epochs} - train RMSE: {rmse:.4f}")

    def rmse(self, data, u_inv_map=None, i_inv_map=None, feat_builder=None):
        sse = 0.0
        for u, i, r in data:
            if self.use_features and feat_builder is not None and u_inv_map is not None and i_inv_map is not None:
                raw_u = list(u_inv_map.keys())[list(u_inv_map.values()).index(u)]
                raw_i = list(i_inv_map.keys())[list(i_inv_map.values()).index(i)]
                xu = feat_builder.user_vector(raw_u)
                zi = feat_builder.item_vector(raw_i)
                pred = self.predict(u, i, xu, zi)
            else:
                pred = self.predict(u, i)
            err = r - pred
            sse += err * err
        return math.sqrt(sse / len(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_features", type=int, default=0)
    ap.add_argument("--feat_reg", type=float, default=0.01)
    args = ap.parse_args()

    base_path = os.path.join(args.data_dir, f"u{args.split}.base")
    test_path = os.path.join(args.data_dir, f"u{args.split}.test")
    users_path = os.path.join(args.data_dir, "u.user")
    items_path = os.path.join(args.data_dir, "u.item")

    train_raw = load_ratings(base_path)
    test_raw = load_ratings(test_path)
    print(f"Loaded train: {len(train_raw)} ratings, test: {len(test_raw)} ratings.")
    users = load_users(users_path) if os.path.exists(users_path) else {}
    items = load_items(items_path) if os.path.exists(items_path) else {}
    if args.use_features:
        feat_builder = FeatureBuilder(users, items)
        feat_builder.fit_stats()
        print(f"User feat dim: {feat_builder.user_dim} | Item feat dim: {feat_builder.item_dim}")
    else:
        feat_builder = None
    train, u_map, i_map = remap_ids(train_raw)
    num_users = len(u_map)
    num_items = len(i_map)
    test = align_with_maps(test_raw, u_map, i_map)
    print(f"Num users: {num_users}, num items: {num_items}. Test-aligned pairs: {len(test)}.")
    u_inv_map = {raw: idx for raw, idx in u_map.items()}
    i_inv_map = {raw: idx for raw, idx in i_map.items()}
    model = SVDwithFeatures(
        num_users, num_items, n_factors=args.factors, lr=args.lr, reg=args.reg, seed=args.seed,
        use_features=bool(args.use_features),
        user_feat_dim=(feat_builder.user_dim if (feat_builder and args.use_features) else 0),
        item_feat_dim=(feat_builder.item_dim if (feat_builder and args.use_features) else 0),
        feat_reg=args.feat_reg,
    )
    model.fit(train, n_epochs=args.epochs, shuffle=True, verbose=True,
              u_inv_map=u_inv_map, i_inv_map=i_inv_map, feat_builder=feat_builder)
    train_rmse = model.rmse(train, u_inv_map=u_inv_map, i_inv_map=i_inv_map, feat_builder=feat_builder)
    test_rmse = model.rmse(test, u_inv_map=u_inv_map, i_inv_map=i_inv_map, feat_builder=feat_builder)
    print(f"Final RMSE — train: {train_rmse:.4f} | test: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
