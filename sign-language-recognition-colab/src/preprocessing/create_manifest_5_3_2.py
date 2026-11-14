#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script phân chia dữ liệu Train / Val / Test chuẩn tỷ lệ 5 / 3 / 2 (50% / 30% / 20%)
dành riêng cho Model 1 (hand_gesture_app), kết hợp Dataset ASL gốc và Dataset tự chụp.

Quy trình:
1. Sao lưu manifest 70/15/15 cũ thành:
   - train_backup_70_15_15.csv
   - val_backup_70_15_15.csv
   - test_backup_70_15_15.csv
2. Phân chia tập ASL gốc theo 'label':
   - 50% Train, 30% Val, 20% Test (Stratified)
3. Phân chia tập Tự chụp theo từng thành viên (Bao, Dat, Khai, Tien, Viet) & từng label:
   - Sử dụng Largest Remainder Method (Phương pháp phần dư lớn nhất)
   - Đảm bảo mỗi nhãn và tổng số mẫu của mỗi thành viên đạt đúng 50% / 30% / 20%.
4. Hợp nhất tập Gốc và Tự chụp, xáo trộn ngẫu nhiên (shuffle seed=42) thành:
   - train.csv
   - val.csv
   - test.csv
5. Xuất ra thư mục: data/hybrid_manifest
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import math
import shutil
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def allocate_5_3_2_for_member(member_df: pd.DataFrame, seed: int = 42):
    """
    Phân chia chính xác tỷ lệ 5/3/2 (50% Train, 30% Val, 20% Test) theo từng label
    của một thành viên bằng Largest Remainder Method, đảm bảo tổng số mẫu của thành viên
    cũng đạt đúng 50% / 30% / 20%.
    """
    label_groups = []

    for label, grp in sorted(member_df.groupby("label"), key=lambda x: str(x[0])):
        n = len(grp)
        shuffled = grp.sample(frac=1.0, random_state=seed)
        exact_tr = n * 0.50
        exact_va = n * 0.30
        exact_te = n * 0.20

        if n >= 3:
            floors = [int(math.floor(exact_tr)), int(math.floor(exact_va)), int(math.floor(exact_te))]
            rems = [exact_tr - floors[0], exact_va - floors[1], exact_te - floors[2]]
            leftover = n - sum(floors)
            order = sorted(range(3), key=lambda idx: (rems[idx], -idx), reverse=True)
            for idx in order[:leftover]:
                floors[idx] += 1
            n_tr, n_va, n_te = floors[0], floors[1], floors[2]
            if n >= 5 and n_te == 0:
                n_te = 1
                if n_tr > 1:
                    n_tr -= 1
        elif n == 2:
            n_tr, n_va, n_te = 1, 1, 0
        else:
            n_tr, n_va, n_te = 1, 0, 0

        label_groups.append({
            "label": label,
            "df": shuffled,
            "n": n,
            "n_tr": n_tr,
            "n_va": n_va,
            "n_te": n_te,
        })

    # Cân bằng tổng của thành viên đạt đúng 50% / 30% / 20%
    total_n = sum(item["n"] for item in label_groups)
    if total_n >= 10:
        t_floors = [
            int(math.floor(total_n * 0.50)),
            int(math.floor(total_n * 0.30)),
            int(math.floor(total_n * 0.20)),
        ]
        t_rems = [
            total_n * 0.50 - t_floors[0],
            total_n * 0.30 - t_floors[1],
            total_n * 0.20 - t_floors[2],
        ]
        for idx in sorted(range(3), key=lambda i: (t_rems[i], -i), reverse=True)[: total_n - sum(t_floors)]:
            t_floors[idx] += 1
        target_tr, target_va, target_te = t_floors[0], t_floors[1], t_floors[2]

        for _ in range(total_n):
            cur_tr = sum(item["n_tr"] for item in label_groups)
            cur_va = sum(item["n_va"] for item in label_groups)
            cur_te = sum(item["n_te"] for item in label_groups)
            if (cur_tr, cur_va, cur_te) == (target_tr, target_va, target_te):
                break
            if cur_tr < target_tr and cur_va > target_va:
                for item in label_groups:
                    if item["n_va"] > 1:
                        item["n_va"] -= 1
                        item["n_tr"] += 1
                        break
            elif cur_va < target_va and cur_tr > target_tr:
                for item in label_groups:
                    if item["n_tr"] > 1:
                        item["n_tr"] -= 1
                        item["n_va"] += 1
                        break
            elif cur_te < target_te and cur_tr > target_tr:
                for item in label_groups:
                    if item["n_tr"] > 1:
                        item["n_tr"] -= 1
                        item["n_te"] += 1
                        break

    train_parts, val_parts, test_parts = [], [], []
    for item in label_groups:
        sub_df = item["df"]
        n_tr = item["n_tr"]
        n_va = item["n_va"]
        n_te = item["n_te"]

        train_parts.append(sub_df.iloc[:n_tr])
        val_parts.append(sub_df.iloc[n_tr:n_tr + n_va])
        test_parts.append(sub_df.iloc[n_tr + n_va:n_tr + n_va + n_te])

    m_tr = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    m_va = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    m_te = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return m_tr, m_va, m_te


def split_captured_dataframe(df_cap: pd.DataFrame, seed: int = 42):
    """Phân chia toàn bộ dataset tự chụp cho từng thành viên."""
    if df_cap is None or len(df_cap) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train_list, val_list, test_list = [], [], []
    for member, member_df in sorted(df_cap.groupby("member"), key=lambda x: str(x[0])):
        m_tr, m_va, m_te = allocate_5_3_2_for_member(member_df, seed=seed)
        train_list.append(m_tr)
        val_list.append(m_va)
        test_list.append(m_te)

    train_cap = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_cap = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_cap = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    return train_cap, val_cap, test_cap


def run_pipeline():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    landmarks_dir = data_dir / "landmarks"
    manifest_dir = data_dir / "hybrid_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = landmarks_dir / "landmarks_raw.csv"
    cap_csv = landmarks_dir / "landmarks_captured.csv"

    print("=" * 72)
    print("BẮT ĐẦU PHÂN CHIA DỮ LIỆU HYBRID 5 / 3 / 2 CHO MODEL HIỆN TẠI")
    print("=" * 72)

    # 1. Tạo file sao lưu 70/15/15 từ các file landmarks cũ
    old_train = landmarks_dir / "train_landmarks.csv"
    old_val = landmarks_dir / "val_landmarks.csv"
    old_test = landmarks_dir / "test_landmarks.csv"

    bk_train = manifest_dir / "train_backup_70_15_15.csv"
    bk_val = manifest_dir / "val_backup_70_15_15.csv"
    bk_test = manifest_dir / "test_backup_70_15_15.csv"

    # Sao lưu từ landmarks cũ nếu có (dùng shutil.copy để cập nhật timestamp mới)
    if old_train.exists():
        shutil.copy(old_train, bk_train)
        print(f"📦 Đã tạo sao lưu: {bk_train.name} ({bk_train.stat().st_size / (1024*1024):.2f} MB)")
    if old_val.exists():
        shutil.copy(old_val, bk_val)
        print(f"📦 Đã tạo sao lưu: {bk_val.name} ({bk_val.stat().st_size / (1024*1024):.2f} MB)")
    if old_test.exists():
        shutil.copy(old_test, bk_test)
        print(f"📦 Đã tạo sao lưu: {bk_test.name} ({bk_test.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Đọc và phân chia tập ASL gốc (landmarks_raw.csv)
    print(f"\n1. Đang nạp dataset ASL gốc: {raw_csv}")
    df_raw = pd.read_csv(raw_csv)
    print(f"   -> Tổng mẫu ASL gốc: {len(df_raw):,} mẫu")

    seed = 42
    # 50% Train, 50% Còn lại
    train_raw, rem_raw = train_test_split(df_raw, test_size=0.50, stratify=df_raw['label'], random_state=seed)
    # 30% Val, 20% Test trong tổng thể (tức 60% Val, 40% Test trong 50% còn lại)
    val_raw, test_raw = train_test_split(rem_raw, test_size=0.40, stratify=rem_raw['label'], random_state=seed)
    print(f"   ASL Gốc -> Train (50%): {len(train_raw):,} | Val (30%): {len(val_raw):,} | Test (20%): {len(test_raw):,}")

    # 3. Đọc và phân chia tập tự chụp (landmarks_captured.csv)
    print(f"\n2. Đang nạp dataset tự chụp đa thành viên: {cap_csv}")
    df_cap = pd.read_csv(cap_csv)
    print(f"   -> Tổng mẫu tự chụp: {len(df_cap):,} mẫu ({df_cap['member'].value_counts().to_dict()})")

    train_cap, val_cap, test_cap = split_captured_dataframe(df_cap, seed=seed)
    print(f"   Tự chụp -> Train (50%): {len(train_cap):,} | Val (30%): {len(val_cap):,} | Test (20%): {len(test_cap):,}")

    # 4. Hợp nhất và Xáo trộn ngẫu nhiên (Shuffle)
    print("\n3. Đang hợp nhất tập gốc và tự chụp, xáo trộn toàn diện (seed=42)...")
    train_combined = pd.concat([train_raw, train_cap], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_combined   = pd.concat([val_raw, val_cap], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_combined  = pd.concat([test_raw, test_cap], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_path = manifest_dir / "train.csv"
    val_path   = manifest_dir / "val.csv"
    test_path  = manifest_dir / "test.csv"

    train_combined.to_csv(train_path, index=False)
    val_combined.to_csv(val_path, index=False)
    test_combined.to_csv(test_path, index=False)

    total_combined = len(train_combined) + len(val_combined) + len(test_combined)

    print("\n" + "=" * 72)
    print("📊 KẾT QUẢ PHÂN CHIA TẬP DỮ LIỆU HYBRID MANIFEST (TỶ LỆ 5 / 3 / 2):")
    print("=" * 72)
    print(f"{'Tập':<12} | {'Tổng mẫu':<10} | {'Tỷ lệ':<8} | {'Gốc (Raw)':<12} | {'Tự chụp':<10}")
    print("-" * 72)
    print(f"{'Train (50%)':<12} | {len(train_combined):<10,} | {len(train_combined)/total_combined*100:6.2f}% | {len(train_raw):<12,} | {len(train_cap):<10,}")
    print(f"{'Val   (30%)':<12} | {len(val_combined):<10,} | {len(val_combined)/total_combined*100:6.2f}% | {len(val_raw):<12,} | {len(val_cap):<10,}")
    print(f"{'Test  (20%)':<12} | {len(test_combined):<10,} | {len(test_combined)/total_combined*100:6.2f}% | {len(test_raw):<12,} | {len(test_cap):<10,}")
    print("-" * 72)
    print(f"{'TỔNG CỘNG':<12} | {total_combined:<10,} | 100.00% | {len(df_raw):<12,} | {len(df_cap):<10,}")
    print("=" * 72)

    # Chi tiết theo từng thành viên
    print("\n👥 Chi tiết phân chia 5/3/2 theo từng thành viên:")
    members = sorted(df_cap["member"].unique())
    for m in members:
        m_tr = len(train_cap[train_cap["member"] == m])
        m_va = len(val_cap[val_cap["member"] == m])
        m_te = len(test_cap[test_cap["member"] == m])
        m_tot = m_tr + m_va + m_te
        print(f"  • {m:8s} ({m_tot:3d} mẫu): Train={m_tr:3d} ({m_tr/m_tot*100:5.1f}%) | Val={m_va:3d} ({m_va/m_tot*100:5.1f}%) | Test={m_te:3d} ({m_te/m_tot*100:5.1f}%)")

    print("\n📁 Danh sách các file trong data/hybrid_manifest:")
    for f in sorted(manifest_dir.glob("*.csv")):
        print(f"  - {f.name:30s}: {f.stat().st_size / (1024*1024):6.2f} MB ({len(pd.read_csv(f)):,} dòng)")


if __name__ == "__main__":
    run_pipeline()
