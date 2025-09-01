#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic SVD (matrix factorization) for MovieLens 100K on the official split,
kept minimal and reproducible.

- Trains on u1.base and evaluates on u1.test (you can change the split id).
- Uses only observed ratings (no imputation). 
- Model: r̂_ui = μ + b_u + b_i + p_u · q_i
- Optimization: plain SGD with L2 regularization.

It also loads user/item "features" files (u.user, u.item) for reference,
but does not inject them into the minimal SVD, keeping it as the classic baseline.
You can extend the code to use these later if needed.

Usage:
    python svd_ml100k.py --data_dir /path/to/ml-100k --split 1 \
        --factors 64 --lr 0.01 --reg 0.02 --epochs 20 --seed 42

Outputs training/test RMSE.

Author: ChatGPT
"""
import argparse
import csv
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np


def load_ratings(path: str) -> List[Tuple[int, int, float]]:
    """Load ratings from a file like u1.base/u1.test: user\titem\trating\ttimestamp"""
    ratings: List[Tuple[int, int, float]] = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            if not line.strip():
                continue
            u, i, r, _ = line.strip().split("\t")
            ratings.append((int(u), int(i), float(r)))
    return ratings


def load_users(path: str) -> Dict[int, Dict]:
    """Load u.user (user id | age | gender | occupation | zip)"""
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
    """Load u.item (movie id | title | release date | video release date | IMDb URL | genres[19])"""
    items = {}
    with open(path, "r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if not row:
                continue
            mid = int(row[0])
            title = row[1]
            release_date = row[2]
            video_release_date = row[3]
            imdb_url = row[4]
            genres = list(map(int, row[5:]))
            items[mid] = {
                "title": title,
                "release_date": release_date,
                "video_release_date": video_release_date,
                "imdb_url": imdb_url,
                "genres": genres,
            }
    return items


class FunkSVD:
    def __init__(
        self,
        num_users: int,
        num_items: int,
        n_factors: int = 64,
        lr: float = 0.01,
        reg: float = 0.02,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.mu = 0.0
        self.bu = np.zeros(num_users, dtype=np.float32)
        self.bi = np.zeros(num_items, dtype=np.float32)
        self.P = 0.1 * rng.standard_normal((num_users, n_factors)).astype(np.float32)
        self.Q = 0.1 * rng.standard_normal((num_items, n_factors)).astype(np.float32)
        self.lr = lr
        self.reg = reg
        self.n_factors = n_factors

    def predict(self, u: int, i: int) -> float:
        return self.mu + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))

    def fit(
        self,
        train: List[Tuple[int, int, float]],
        n_epochs: int = 20,
        shuffle: bool = True,
        verbose: bool = True,
    ):
        if not train:
            raise ValueError("Empty training set.")
        # Compute global mean on train
        self.mu = float(np.mean([r for _, _, r in train]))
        # Training
        for epoch in range(1, n_epochs + 1):
            if shuffle:
                random.shuffle(train)
            sse = 0.0
            for u, i, r in train:
                # indices are assumed zero-based outside (we'll remap before training)
                pred = self.mu + self.bu[u] + self.bi[i] + float(np.dot(self.P[u], self.Q[i]))
                err = r - pred
                sse += err * err
                # SGD updates
                bu_old = self.bu[u]
                bi_old = self.bi[i]
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                # latent factors
                Pu = self.P[u]
                Qi = self.Q[i]
                self.P[u] += self.lr * (err * Qi - self.reg * Pu)
                self.Q[i] += self.lr * (err * Pu - self.reg * Qi)
            rmse = math.sqrt(sse / len(train))
            if verbose:
                print(f"Epoch {epoch:02d}/{n_epochs} - train RMSE: {rmse:.4f}")

    def rmse(self, data: List[Tuple[int, int, float]]) -> float:
        sse = 0.0
        for u, i, r in data:
            err = r - self.predict(u, i)
            sse += err * err
        return math.sqrt(sse / len(data))


def remap_ids(
    ratings: List[Tuple[int, int, float]]
) -> Tuple[List[Tuple[int, int, float]], Dict[int, int], Dict[int, int]]:
    """Remap raw 1-based user/movie ids to contiguous 0-based indices."""
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
    """Keep only pairs present in the training maps and convert to 0-based."""
    aligned = []
    for u, i, r in ratings:
        if u in u_map and i in i_map:
            aligned.append((u_map[u], i_map[i], r))
    return aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Path to ml-100k directory")
    ap.add_argument("--split", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Use u{split}.base/test")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--reg", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base_path = os.path.join(args.data_dir, f"u{args.split}.base")
    test_path = os.path.join(args.data_dir, f"u{args.split}.test")
    users_path = os.path.join(args.data_dir, "u.user")
    items_path = os.path.join(args.data_dir, "u.item")

    # --- Load ratings
    train_raw = load_ratings(base_path)
    test_raw = load_ratings(test_path)
    print(f"Loaded train: {len(train_raw)} ratings, test: {len(test_raw)} ratings.")

    # --- Load "features" for reference (not used by the minimal SVD model)
    if os.path.exists(users_path):
        users = load_users(users_path)
        print(f"Loaded users: {len(users)} (w/ age, gender, occupation, zip)")
    else:
        users = {}

    if os.path.exists(items_path):
        items = load_items(items_path)
        print(f"Loaded items: {len(items)} (w/ titles, dates, genres[19])")
    else:
        items = {}

    # --- Remap ids using only the training set (avoid test leakage)
    train, u_map, i_map = remap_ids(train_raw)
    num_users = len(u_map)
    num_items = len(i_map)
    test = align_with_maps(test_raw, u_map, i_map)
    print(f"Num users: {num_users}, num items: {num_items}. Test-aligned pairs: {len(test)}.")

    # --- Train model
    model = FunkSVD(num_users, num_items, n_factors=args.factors, lr=args.lr, reg=args.reg, seed=args.seed)
    model.fit(train, n_epochs=args.epochs, shuffle=True, verbose=True)

    # --- Evaluate
    train_rmse = model.rmse(train)
    test_rmse = model.rmse(test)
    print(f"Final RMSE — train: {train_rmse:.4f} | test: {test_rmse:.4f}")


if __name__ == "__main__":
    main()
