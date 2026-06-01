# -*- coding: utf-8 -*-
"""
v21_app.py
==========
v19 モデル対応 予想アプリ（バックテスト＋当日予想 両対応）

【機能】
  タブ1: バックテスト
    - collect_v19_data.py で作ったCSVを読み、過去オッズでEVバックテスト
    - EV閾値・最低確率・上限点数を変えて回収率を即計算
    - 確率帯×EV帯の回収率マトリクスを表示
  タブ2: 当日予想
    - kyotei.fun の当日ページから出走表＋展示＋コースIN＋オッズを取得
    - v19モデルで確率を出し、EV>閾値の買い目だけを採用
    - ※当日のレースで展示・コースINが未確定なら、その特徴量は欠損として推論

【依存】
  - lgb_p1_v19.txt / lgb_p2_v19.txt / lgb_p3_v19.txt （学習スクリプトの出力）
  - lgb_p2_v19_features.json / lgb_p3_v19_features.json （特徴量名）
  - 学習スクリプトと同じディレクトリで実行する想定
"""

import os
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import lightgbm as lgb
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=+9), 'JST')

# ============================================================
# 0. モデル＆設定
# ============================================================
MODEL_DIR = "."   # モデルファイルがあるディレクトリ

@st.cache_resource
def load_model(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return lgb.Booster(model_file=path)
    except Exception as e:
        st.warning(f"モデル読み込み失敗 {filename}: {e}")
        return None

@st.cache_resource
def load_features(filename: str) -> Optional[List[str]]:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

m_p1 = load_model("lgb_p1_v19.txt")
m_p2 = load_model("lgb_p2_v19.txt")
m_p3 = load_model("lgb_p3_v19.txt")
features_p1 = load_features("lgb_p1_v19_features.json")
features_p2 = load_features("lgb_p2_v19_features.json")
features_p3 = load_features("lgb_p3_v19_features.json")

JCD_NAME = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川", 6:"浜名湖",
    7:"蒲郡", 8:"常滑", 9:"津", 10:"三国", 11:"びわこ", 12:"住之江",
    13:"尼崎", 14:"鳴門", 15:"丸亀", 16:"児島", 17:"宮島", 18:"徳山",
    19:"下関", 20:"若松", 21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村"
}
JCD_FROM_NAME = {v: k for k, v in JCD_NAME.items()}


# ============================================================
# 1. レースデータから特徴量を作る共通関数
# ============================================================
def make_race_features(racer_rows: List[Dict]) -> pd.DataFrame:
    """6艇分のdict (lane, cls_val, age, ..., tenji, course_in) のリストから
    学習時と同じ41列の特徴量DataFrameを作る。"""
    df = pd.DataFrame(racer_rows).sort_values("lane").reset_index(drop=True)

    # 偏差・順位
    df["win_dev"]    = df["n_win"]  - df["n_win"].mean()
    df["motor_dev"]  = df["m_2ren"] - df["m_2ren"].mean()
    df["st_dev"]     = df["avg_st"].mean() - df["avg_st"]
    df["tenji_dev"]  = df["tenji"].mean() - df["tenji"]
    df["win_rank"]   = df["n_win"].rank(ascending=False, method="min").astype(int)
    df["motor_rank"] = df["m_2ren"].rank(ascending=False, method="min").astype(int)
    df["st_rank"]    = df["avg_st"].rank(ascending=True,  method="min").astype(int)
    df["tenji_rank"] = df["tenji"].rank(ascending=True,   method="min").astype(int)
    # 前付け
    df["maezuke"]    = (df["lane"] != df["course_in"]).astype(int)
    df["course_diff"] = df["course_in"] - df["lane"]
    # 隣接差
    for col in ["avg_st", "n_win", "tenji"]:
        for direction, shift in [("in", 1), ("out", -1)]:
            vals = []
            for i in range(len(df)):
                j = i - shift
                if 0 <= j < len(df):
                    vals.append(df.loc[i, col] - df.loc[j, col])
                else:
                    vals.append(0.0)
            df[f"{col}_diff_{direction}"] = vals
    return df


def predict_combo_probs(features_df: pd.DataFrame, race_jcd: int) -> Dict[str, float]:
    """6艇の特徴量から、120点の3連単確率 {'1-2-3': 0.087, ...} を返す。
    p1:単独 / p2:1着艇情報を条件付与 / p3:1着・2着艇の情報を条件付与"""
    if not (m_p1 and m_p2 and m_p3):
        return {}

    df = features_df.copy()
    df["jcd"] = race_jcd
    base_cols = features_p1 if features_p1 else [c for c in df.columns if c not in ("name",)]

    # p1: 各艇の1着率
    p1 = {}
    for _, row in df.iterrows():
        x = row[base_cols].values.reshape(1, -1).astype(float)
        p1[int(row["lane"])] = float(m_p1.predict(x)[0])
    s = sum(p1.values())
    if s > 0:
        p1 = {k: v/s for k, v in p1.items()}

    combos = {}
    for w1 in range(1, 7):
        # 1着 = w1 とした条件で、各候補艇の2着確率
        w1_row = df[df["lane"]==w1].iloc[0]
        p2_raw = {}
        for cand in range(1, 7):
            if cand == w1: continue
            cand_row = df[df["lane"]==cand].iloc[0]
            feat = {f: cand_row[f] for f in base_cols if f in cand_row.index}
            # 1着艇の特徴を w1_ プレフィックスで付加
            for f in ["lane","cls_val","avg_st","n_win","m_2ren","tenji","course_in","maezuke"]:
                feat[f"w1_{f}"] = w1_row[f]
            feat["w1_lane_diff"]   = cand_row["lane"]      - w1_row["lane"]
            feat["w1_course_diff"] = cand_row["course_in"] - w1_row["course_in"]
            x = np.array([feat.get(c, 0.0) for c in features_p2]).reshape(1, -1).astype(float)
            p2_raw[cand] = float(m_p2.predict(x)[0])
        s2 = sum(p2_raw.values())
        p2 = {k: v/s2 if s2>0 else 0 for k, v in p2_raw.items()}

        for w2 in range(1, 7):
            if w2 == w1: continue
            w2_row = df[df["lane"]==w2].iloc[0]
            p3_raw = {}
            for cand in range(1, 7):
                if cand in (w1, w2): continue
                cand_row = df[df["lane"]==cand].iloc[0]
                feat = {f: cand_row[f] for f in base_cols if f in cand_row.index}
                for f in ["lane","cls_val","avg_st","n_win","m_2ren","tenji","course_in","maezuke"]:
                    feat[f"w1_{f}"] = w1_row[f]
                feat["w1_lane_diff"]   = cand_row["lane"]      - w1_row["lane"]
                feat["w1_course_diff"] = cand_row["course_in"] - w1_row["course_in"]
                for f in ["lane","cls_val","avg_st","n_win","m_2ren","tenji","course_in"]:
                    feat[f"w2_{f}"] = w2_row[f]
                feat["w2_lane_diff"] = cand_row["lane"] - w2_row["lane"]
                x = np.array([feat.get(c, 0.0) for c in features_p3]).reshape(1, -1).astype(float)
                p3_raw[cand] = float(m_p3.predict(x)[0])
            s3 = sum(p3_raw.values())
            p3 = {k: v/s3 if s3>0 else 0 for k, v in p3_raw.items()}

            for w3 in range(1, 7):
                if w3 in (w1, w2): continue
                combos[f"{w1}-{w2}-{w3}"] = p1[w1] * p2[w2] * p3[w3]
    return combos


# ============================================================
# 2. 当日データ取得（節度ある1リクエスト）
# ============================================================
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
req_sess = requests.Session()
req_sess.headers.update(UA)

RE_NG_NUM = re.compile(r"ng1?r(\d)")
RE_CLS    = re.compile(r"([A12B]{2})")
RE_WEIGHT = re.compile(r"(\d+)kg", re.IGNORECASE)
RE_AGE    = re.compile(r"\((\d{2})\)")
CLS_MAP   = {"A1":4, "A2":3, "B1":2, "B2":1}


def _lane_from_class(td) -> Optional[int]:
    div = td.find("div", class_=lambda c: c and "ng1r" in c)
    if not div: return None
    for cls in div.get("class", []):
        m = re.match(r"ng1r(\d)$", cls)
        if m: return int(m.group(1))
    return None


def fetch_race_data(date: str, jcd: int, rno: int) -> Optional[Tuple[List[Dict], Dict, Dict[str,float], Optional[int]]]:
    """1レース取得。返り値: (6艇のrows, lane_to_rank, odds_3t, payoff_3t) または None"""
    url = f"https://info.kyotei.fun/info-{date}-{jcd:02d}-{rno}.html"
    try:
        r = req_sess.get(url, timeout=15)
        r.encoding = r.apparent_encoding
        if r.status_code != 200 or len(r.text) < 5000:
            return None
    except requests.RequestException:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    # 着順
    lane_to_rank = {}
    for i, d in enumerate(soup.find_all("div", class_="jyuni")[:6]):
        t = d.get_text(strip=True)
        if t.isdigit(): lane_to_rank[i+1] = int(t)

    # 出走表本体
    base = {i+1: {
        "lane": i+1, "age":30, "cls_val":1, "weight":50, "f_count":0, "avg_st":0.17,
        "n_win":0.0, "n_2ren":0.0, "l_win":0.0, "l_2ren":0.0, "m_2ren":0.0, "b_2ren":0.0,
        "tenji":6.80, "course_in": i+1,
    } for i in range(6)}
    current_label = ""
    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td","th"])
        if not tds: continue
        if len(tds) >= 7:
            current_label = tds[0].get_text(strip=True).replace("\n","").replace(" ","").replace("\u3000","")
            data_tds = tds[-6:]
        elif len(tds) == 6 and current_label:
            data_tds = tds
        else:
            current_label = ""
            continue
        for i in range(6):
            td = data_tds[i]
            txt = td.get_text(" ", strip=True).replace(" ","").replace("\u3000","").replace("\n","")
            lane = i+1
            if "選手名" in current_label:
                m = RE_AGE.search(txt)
                if m: base[lane]["age"] = int(m.group(1))
            elif "選手情報" in current_label or "支部" in current_label:
                m_cls = RE_CLS.search(txt)
                if m_cls: base[lane]["cls_val"] = CLS_MAP.get(m_cls.group(1), 1)
                m_w = RE_WEIGHT.search(txt)
                if m_w: base[lane]["weight"] = int(m_w.group(1))
            elif "級過去2期" in current_label:
                m_cls = RE_CLS.search(txt)
                if m_cls: base[lane]["cls_val"] = CLS_MAP.get(m_cls.group(1), 1)
            elif "全国" in current_label and "勝率" in current_label:
                m2 = re.search(r"^([\d\.]+)", txt); mw = re.search(r"\(([\d\.]+)\)", txt)
                if m2: v=float(m2.group(1)); base[lane]["n_2ren"] = v/100.0 if v>1.0 else v
                if mw: base[lane]["n_win"] = float(mw.group(1))
            elif "当地" in current_label and "勝率" in current_label:
                m2 = re.search(r"^([\d\.]+)", txt); mw = re.search(r"\(([\d\.]+)\)", txt)
                if m2: v=float(m2.group(1)); base[lane]["l_2ren"] = v/100.0 if v>1.0 else v
                if mw: base[lane]["l_win"] = float(mw.group(1))
            elif "モータ" in current_label and "2連率" in current_label:
                m = re.search(r"^([\d\.]+)", txt)
                if m: v=float(m.group(1)); base[lane]["m_2ren"] = v/100.0 if v>1.0 else v
            elif "ボート" in current_label and "2連率" in current_label:
                m = re.search(r"^([\d\.]+)", txt)
                if m: v=float(m.group(1)); base[lane]["b_2ren"] = v/100.0 if v>1.0 else v
            elif "平均ST" in current_label:
                try: base[lane]["avg_st"] = float(txt)
                except: pass
            elif "フライング" in current_label:
                try: base[lane]["f_count"] = int(txt)
                except: pass
            elif current_label == "展示":
                try: base[lane]["tenji"] = float(txt)
                except: pass
            elif current_label == "コースIN":
                c = _lane_from_class(td)
                if c: base[lane]["course_in"] = c

    rows = [base[i+1] for i in range(6)]

    # 3連単オッズ
    odds_map = {}
    h3_target = None
    for h3 in soup.find_all("h3"):
        if "3連単" in h3.get_text() and "人気" in h3.get_text():
            h3_target = h3; break
    if h3_target:
        container = h3_target.find_parent("div", id="raceData") or h3_target.parent
        for tbl in container.find_all("table", id="oddsTbl"):
            for tr in tbl.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) != 2: continue
                ng23 = tds[0].find("div", class_="ng23")
                if not ng23: continue
                divs = ng23.find_all("div")
                nums = []
                for d in divs[:3]:
                    m = re.search(r"ng2r(\d)", " ".join(d.get("class", [])))
                    if m: nums.append(int(m.group(1)))
                if len(nums) != 3 or len(set(nums)) != 3: continue
                txt = tds[1].get_text(strip=True).replace(",","")
                try: v = float(txt)
                except: continue
                odds_map[f"{nums[0]}-{nums[1]}-{nums[2]}"] = v

    # 払戻
    payoff = None
    for box in soup.find_all("div", class_="race_result_end_line"):
        label = box.find("div", class_="race_result_end_label")
        if label and label.get_text(strip=True) == "3連単":
            money = box.find("span", class_="race_result_end_money_num")
            if money:
                t = money.get_text(strip=True).replace(",","")
                if t.isdigit(): payoff = int(t)

    return rows, lane_to_rank, odds_map, payoff


# ============================================================
# 3. EV計算・買い目選定
# ============================================================
def select_bets_by_ev(combo_probs: Dict[str, float], odds_map: Dict[str, float],
                       ev_th: float, min_prob: float, max_n: int) -> List[Dict]:
    """確率×オッズで EV を計算し、条件を満たすものを EV 降順で返す。"""
    out = []
    for combo, p in combo_probs.items():
        o = odds_map.get(combo, 0.0)
        if o <= 0: continue
        if p < min_prob: continue
        ev = p * o
        if ev < ev_th: continue
        out.append({"bet":combo, "prob":p, "odds":o, "ev":ev})
    out.sort(key=lambda x: x["ev"], reverse=True)
    return out[:max_n]


# ============================================================
# 4. Streamlit UI
# ============================================================
st.set_page_config(page_title="v21 EVバックテスト＆当日予想", layout="wide")
st.title("🚤 v21 EVバックテスト＆当日予想アプリ")

model_ready = all([m_p1, m_p2, m_p3, features_p1, features_p2, features_p3])
if not model_ready:
    st.error("⚠️ v19モデルファイルが見つかりません。下記のファイルをアプリと同じフォルダに置いてください:")
    st.code("lgb_p1_v19.txt\nlgb_p2_v19.txt\nlgb_p3_v19.txt\n"
            "lgb_p1_v19_features.json\nlgb_p2_v19_features.json\nlgb_p3_v19_features.json")
    st.stop()

# サイドバー
st.sidebar.markdown("### ⚙️ EV判定設定")
ev_th     = st.sidebar.slider("EV閾値 (EV以上を購入)", 1.0, 2.0, 1.10, 0.05)
min_prob  = st.sidebar.slider("最低予想確率 (%) 以上", 0.0, 10.0, 1.0, 0.5) / 100.0
max_bets  = st.sidebar.slider("1レース上限点数", 1, 12, 6, 1)
bet_amt   = st.sidebar.number_input("1点の購入金額 (円)", min_value=100, step=100, value=100)
st.sidebar.caption("EVは『予想確率×オッズ』。1.0でトントン、超えれば理論プラス。")
st.sidebar.caption("ただしモデル確率が過大評価ならEV>1でも負ける。検証必須。")

tab1, tab2 = st.tabs(["📊 バックテスト (過去CSV)", "🎯 当日予想 (1レース)"])

# ============================================================
# Tab1: バックテスト
# ============================================================
with tab1:
    st.markdown("##### collect_v19_data.py で取得したCSVを読み、EV>閾値の買い目だけ買った場合の回収率を測定します。")
    uploaded = st.file_uploader("CSVをアップロード", type=["csv"])
    period_filter = st.text_input("バックテスト期間でフィルタ (例: 20260525,20260531 ← 開始,終了)", "")

    if uploaded and st.button("🚀 バックテスト実行", type="primary"):
        df = pd.read_csv(uploaded, dtype={"date":str, "result_combo":str, "odds_3t_json":str})
        df = df[df["tenji"] > 0]
        df = df[df["payoff_3t"] > 0]
        if period_filter.strip():
            try:
                p_start, p_end = [s.strip() for s in period_filter.split(",")]
                df = df[(df["date"] >= p_start) & (df["date"] <= p_end)]
                st.info(f"期間フィルタ {p_start}〜{p_end} 適用後: {len(df)//6:,}レース")
            except Exception:
                st.warning("期間フィルタの形式が不正です。全期間で実行します。")

        race_keys = df[["date","jcd","rno"]].drop_duplicates().values.tolist()
        st.write(f"対象 {len(race_keys):,} レースを処理します...")
        prog = st.progress(0.0)

        all_results = []
        for idx, (d, j, r) in enumerate(race_keys):
            sub = df[(df["date"]==d)&(df["jcd"]==j)&(df["rno"]==r)]
            if len(sub) != 6: continue
            racers = sub.to_dict("records")

            # オッズ
            odds_json = sub.iloc[0]["odds_3t_json"]
            if not isinstance(odds_json, str) or len(odds_json) < 20:
                continue
            try:
                odds_map = json.loads(odds_json)
            except Exception:
                continue

            # 結果
            result_combo = sub.iloc[0]["result_combo"]
            payoff = int(sub.iloc[0]["payoff_3t"])

            # 予測
            feat_df = make_race_features(racers)
            combo_probs = predict_combo_probs(feat_df, int(j))
            chosen = select_bets_by_ev(combo_probs, odds_map, ev_th, min_prob, max_bets)

            buys = [c["bet"] for c in chosen]
            hit = result_combo in buys
            inv = len(buys) * bet_amt
            ret = payoff * (bet_amt / 100.0) if hit else 0
            all_results.append({
                "date": d, "jcd": int(j), "rno": int(r),
                "n_bets": len(buys),
                "buys": ",".join(buys) if buys else "見送り",
                "result": result_combo,
                "hit": 1 if hit else 0,
                "investment": inv,
                "return": ret,
                "payoff": payoff,
                "sum_ev": round(sum(c["ev"] for c in chosen), 2),
            })
            if idx % 20 == 0 or idx == len(race_keys)-1:
                prog.progress((idx+1)/len(race_keys))

        if not all_results:
            st.error("結果が空でした。期間や設定を見直してください。")
            st.stop()

        res = pd.DataFrame(all_results)
        bet_races = res[res["n_bets"] > 0]
        skip_races = res[res["n_bets"] == 0]

        # サマリ
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("対象レース", f"{len(res):,}")
        c2.metric("買ったレース", f"{len(bet_races):,}",
                  f"見送り {len(skip_races):,}")
        if len(bet_races) > 0:
            tot_inv = bet_races["investment"].sum()
            tot_ret = bet_races["return"].sum()
            hit_rate = bet_races["hit"].sum() / len(bet_races) * 100
            ret_rate = tot_ret / tot_inv * 100 if tot_inv>0 else 0
            c3.metric("回収率", f"{ret_rate:.1f}%",
                      f"投資 {int(tot_inv):,}円 / 回収 {int(tot_ret):,}円")
            c4.metric("的中率", f"{hit_rate:.1f}%",
                      f"{int(bet_races['hit'].sum())}/{len(bet_races)}本")

            if ret_rate >= 100:
                st.success(f"🎉 回収率 {ret_rate:.1f}% — 控除率の壁(75%)を超え、理論プラスの領域です。ただし標本数{len(bet_races)}本での結果なので、追加検証が必要です。")
            elif ret_rate >= 85:
                st.info(f"回収率 {ret_rate:.1f}% — 控除率の壁(75%)は超えていますが、まだプラスに届かず。設定の微調整で改善余地あり。")
            else:
                st.warning(f"回収率 {ret_rate:.1f}% — 控除率の壁を超えていません。設定が厳しすぎるか、モデルがその確率帯で過大評価している可能性。")

        st.markdown("---")
        st.subheader("📋 レース別結果")
        st.dataframe(res, use_container_width=True)

        # 確率帯別の回収率
        st.markdown("---")
        st.subheader("📈 EV帯別の回収率（買い目1点ごとに分解）")
        # 各買い目を1行に展開
        rows_expanded = []
        for r in all_results:
            if r["n_bets"]==0: continue
            buys = r["buys"].split(",")
            per_inv = bet_amt
            for b in buys:
                rows_expanded.append({
                    "buy": b, "result": r["result"], "payoff": r["payoff"],
                    "investment": per_inv,
                    "return": r["payoff"]*(per_inv/100.0) if b == r["result"] else 0,
                })
        df_exp = pd.DataFrame(rows_expanded)
        if not df_exp.empty:
            # ここでは買い目ごとのEV/確率が必要だが簡便のため、回収率全体を表示
            n_bets_total = len(df_exp)
            n_hits = (df_exp["return"]>0).sum()
            inv2 = df_exp["investment"].sum()
            ret2 = df_exp["return"].sum()
            st.write(f"買い目総数: {n_bets_total:,} / 的中: {n_hits} / 回収率: {ret2/inv2*100:.1f}%")
            st.caption("※確率帯×EV帯の詳細マトリクスは将来の拡張で追加可能です。まず全体の数字を確認してください。")

# ============================================================
# Tab2: 当日予想
# ============================================================
with tab2:
    st.markdown("##### 当日(or 過去)1レースを取得して、v19モデルでEV>閾値の買い目を出します。")
    cA, cB, cC = st.columns(3)
    with cA: d_input = st.date_input("日付", value=datetime.now(JST).date())
    with cB: v_idx = st.selectbox("場", options=list(JCD_NAME.keys()), format_func=lambda x: JCD_NAME[x])
    with cC: r_idx = st.selectbox("R", options=list(range(1, 13)))

    if st.button("🔍 取得して予想", type="primary", use_container_width=True):
        dstr = d_input.strftime("%Y%m%d")
        with st.spinner("取得中..."):
            res = fetch_race_data(dstr, v_idx, r_idx)
            time.sleep(1.0)   # 節度ある待機
        if not res:
            st.error("取得失敗。日付や場、レース番号を確認してください。")
            st.stop()
        racers, lane_to_rank, odds_map, payoff = res

        st.subheader("出走表")
        df_show = pd.DataFrame(racers)[["lane","cls_val","age","weight","avg_st","n_win","m_2ren","tenji","course_in"]]
        df_show.columns = ["枠","級","年齢","体重","平均ST","勝率","M2連率","展示","コースIN"]
        st.dataframe(df_show.set_index("枠"), use_container_width=True)
        if len(odds_map) > 0:
            st.caption(f"オッズ取得: {len(odds_map)}/120 件")
        else:
            st.warning("3連単オッズが取得できませんでした (締切前等)")

        feat_df = make_race_features(racers)
        with st.spinner("予測中..."):
            combo_probs = predict_combo_probs(feat_df, v_idx)

        if not combo_probs:
            st.error("予測失敗。モデルを確認してください。")
            st.stop()

        # EV計算 (オッズがあれば)
        if odds_map:
            chosen = select_bets_by_ev(combo_probs, odds_map, ev_th, min_prob, max_bets)
            st.subheader(f"🎯 採用買い目 (EV≥{ev_th}, 確率≥{min_prob*100:.1f}%, 上限{max_bets}点)")
            if chosen:
                df_b = pd.DataFrame([{
                    "買い目": c["bet"],
                    "予想確率(%)": round(c["prob"]*100, 2),
                    "オッズ": round(c["odds"], 1),
                    "EV": round(c["ev"], 3),
                } for c in chosen])
                st.dataframe(df_b.set_index("買い目"), use_container_width=True)
                st.code(",".join(c["bet"] for c in chosen))
                if payoff and lane_to_rank:
                    r1 = next((l for l,r in lane_to_rank.items() if r==1), None)
                    r2 = next((l for l,r in lane_to_rank.items() if r==2), None)
                    r3 = next((l for l,r in lane_to_rank.items() if r==3), None)
                    if r1 and r2 and r3:
                        result = f"{r1}-{r2}-{r3}"
                        buys = [c["bet"] for c in chosen]
                        hit = result in buys
                        inv = len(buys)*bet_amt
                        ret = payoff*(bet_amt/100.0) if hit else 0
                        st.success(f"結果: {result} ({payoff}円) — "
                                   f"{'🎯 的中' if hit else '❌ 外れ'} "
                                   f"投資 {inv:,}円 / 回収 {int(ret):,}円")
            else:
                st.info("条件を満たす買い目なし → 見送り")

        # 確率上位は常に表示
        st.subheader("📊 確率上位 (参考)")
        top = sorted(combo_probs.items(), key=lambda x: x[1], reverse=True)[:15]
        df_top = pd.DataFrame([{"買い目":k, "予想確率(%)":round(v*100,2),
                                  "オッズ":odds_map.get(k,0), "EV":round(v*odds_map.get(k,0),3)}
                                 for k,v in top])
        st.dataframe(df_top.set_index("買い目"), use_container_width=True)
