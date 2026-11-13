"""國家海洋研究院 觀測站「風力氣溫氣壓資料」分析 - Streamlit 版

把原本三支 requests.get 下載 CSV 的程式(員貝嶼 / 澎南 / 萬里桐)合併成一個 App:
  * 輸入「產品名稱」篩選要分析的觀測站;可隨時在側邊欄新增 / 批次貼上 / 刪除監測站網址(存於 sources.json)
  * 輸入「每頁筆數」與「頁數」決定每個測站取多少筆資料
  * 依各觀測站的數據繪製折線圖、圓餅圖、旭日圖(Plotly)
  * 「同時間比較」分頁:自選項目/測站/時間範圍/對齊方式,比較各站同一時間的風力、氣溫、氣壓,
    並可手動輸入對照資料

關於中文字型:
  原本在 Colab 用 !wget 下載字型是給 matplotlib 用的(它在伺服器端畫圖)。
  Plotly 圖表是在使用者的瀏覽器裡繪製,直接使用瀏覽器內建的中文字型,
  不需要下載或安裝任何字型檔。

執行:
    pip install -r requirements.txt
    streamlit run app.py
"""
import hashlib
import io
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import urllib3

# ----------------------------- 設定 -----------------------------
BASE = "https://www.namr.gov.tw/filedownload?file=opendata/"
DEFAULT_PRODUCT = "風力氣溫氣壓資料"

# 內建資料來源(取自原本的 CODE01~03)。注意:網址裡的檔名含時間戳記,
# 資料集重新發佈後可能失效,屆時請到官網取得新網址並在側邊欄「自訂資料來源」貼上。
BUILTIN_SOURCES = [
    {
        "station": "員貝嶼",
        "product": DEFAULT_PRODUCT,
        "url": BASE + "202607091634290.csv"
        "&filedisplay=%E5%93%A1%E8%B2%9D%E5%B6%BC%E9%A2%A8%E5%8A%9B%E6%B0%A3%E6%BA%AB%E6%B0%A3%E5%A3%93%E8%B3%87%E6%96%99.csv"
        "&flag=doc&dataserno=202410290011",
    },
    {
        "station": "澎南",
        "product": DEFAULT_PRODUCT,
        "url": BASE + "202607091639520.csv"
        "&filedisplay=%E6%BE%8E%E5%8D%97%E9%A2%A8%E5%8A%9B%E6%B0%A3%E6%BA%AB%E6%B0%A3%E5%A3%93%E8%B3%87%E6%96%99.csv"
        "&flag=doc&dataserno=202410290018",
    },
    {
        "station": "萬里桐",
        "product": DEFAULT_PRODUCT,
        "url": BASE + "202607091637320.csv"
        "&filedisplay=%E8%90%AC%E9%87%8C%E6%A1%90%E9%A2%A8%E5%8A%9B%E6%B0%A3%E6%BA%AB%E6%B0%A3%E5%A3%93%E8%B3%87%E6%96%99.csv"
        "&flag=doc&dataserno=202410290015",
    },
]

SOURCES_FILE = Path(__file__).with_name("sources.json")   # 使用者新增的資料來源會存在這裡

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
ENCODINGS = ["utf-8-sig", "cp950", "big5hkscs"]   # 原檔為 Big5,cp950 是它的超集合
TIME_KEYS = ("時間", "日期", "date", "time", "觀測")
COMPASS = ["北", "東北", "東", "東南", "南", "西南", "西", "西北"]
ALL_STATIONS = "全部測站合併"
FREQ_RULES = {"原始資料": None, "小時平均": "h", "日平均": "D", "週平均": "W", "月平均": "MS"}
NONE_OPT = "(不比較)"
# 三個比較項目的欄位名稱關鍵字(用來自動猜測預設欄位,使用者可改)
VAR_KEYS = {
    "風力": ("風速", "風力", "wind"),
    "氣溫": ("氣溫", "溫度", "temp"),
    "氣壓": ("氣壓", "壓力", "pressure"),
}
ALIGN_RULES = {
    "原始時間(完全相同)": None, "10 分鐘": "10min", "30 分鐘": "30min",
    "1 小時": "h", "1 日": "D",
}
FONT_FAMILY = '"Noto Sans TC","Microsoft JhengHei","PingFang TC","Heiti TC",sans-serif'


# ----------------------------- 資料下載與讀取 -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bytes(url: str):
    """下載檔案。先正常驗證 SSL;若憑證有問題才退回 verify=False(原程式的做法)"""
    insecure = False
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
    except requests.exceptions.SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
        insecure = True
    r.raise_for_status()
    return r.content, insecure


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """依序嘗試多種編碼讀取 CSV,並做基本清理"""
    if content[:512].lstrip().lower().startswith(b"<"):
        raise ValueError("下載到的是網頁而不是 CSV(連結可能已過期)。")
    last = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc)
            break
        except UnicodeDecodeError as e:
            last = e
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise ValueError(f"CSV 格式無法解析:{e}")
    else:
        raise ValueError(f"無法判斷檔案編碼:{last}")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")   # 僅刪除完全為空的列與欄
    return df.reset_index(drop=True)


# ----------------------------- 欄位偵測與整理 -----------------------------
def parse_dt(s: pd.Series) -> pd.Series:
    """解析日期時間字串;支援民國年(例如 115/07/09 10:00)"""
    s = s.astype(str).str.strip()
    m = s.str.extract(r"^(\d{2,3})([/\-.])(\d{1,2})\2(\d{1,2})(.*)$")
    if m[0].notna().mean() > 0.8:
        yr = pd.to_numeric(m[0], errors="coerce")
        if yr.median() < 200:   # 民國年轉西元
            s = (yr + 1911).astype("Int64").astype(str) + "-" + m[2] + "-" + m[3] + m[4]
    return pd.to_datetime(s, errors="coerce", format="mixed")


def detect_time(df: pd.DataFrame):
    """回傳 (時間 Series 或 None, 被用來組成時間的欄位清單)"""
    if "日期" in df.columns and "時間" in df.columns:        # 日期、時間分欄
        t = parse_dt(df["日期"].astype(str) + " " + df["時間"].astype(str))
        if t.notna().mean() > 0.8:
            return t, ["日期", "時間"]

    cands = [c for c in df.columns if any(k in c.lower() for k in TIME_KEYS)]
    for c in cands:
        t = parse_dt(df[c])
        if t.notna().mean() > 0.8:
            return t, [c]

    if all(k in df.columns for k in ("年", "月", "日")):      # 年/月/日(/時/分)分欄
        names = {"年": "year", "月": "month", "日": "day", "時": "hour", "分": "minute"}
        cols = [k for k in names if k in df.columns]
        tmp = pd.DataFrame({names[k]: pd.to_numeric(df[k], errors="coerce") for k in cols})
        if tmp["year"].median() < 200:
            tmp["year"] = tmp["year"] + 1911
        return pd.to_datetime(tmp, errors="coerce"), cols

    for c in df.select_dtypes(exclude="number").columns:     # 最後手段:試每個文字欄
        t = parse_dt(df[c])
        if t.notna().mean() > 0.8:
            return t, [c]
    return None, []


def prepare(df_raw: pd.DataFrame, station: str, sentinels: list):
    """整理單一測站:加上測站/時間欄、數值欄轉型、缺失代碼轉 NaN。
    回傳 (整理後 DataFrame, 數值欄位清單, 用於時間的欄位清單)"""
    t, used = detect_time(df_raw)
    out = pd.DataFrame(index=df_raw.index)
    out["測站"] = station
    if t is None:
        out["時間"] = pd.Series(pd.NaT, index=df_raw.index, dtype="datetime64[ns]")
    else:
        out["時間"] = t

    numeric = []
    for c in df_raw.columns:
        if c in used:
            continue
        name = f"{c}_原始" if c in ("測站", "時間") else c
        s = pd.to_numeric(df_raw[c], errors="coerce")
        if s.notna().mean() >= 0.5:                      # 超過半數可轉數字 → 視為數值欄
            out[name] = s.where(~s.isin(sentinels))
            numeric.append(name)
        else:
            out[name] = df_raw[c]
    return out, numeric, used


def take_rows(df: pd.DataFrame, n: int, newest: bool) -> pd.DataFrame:
    """依時間排序後取 n 筆(最新端或最早端);沒有時間欄就依原順序"""
    if df["時間"].notna().any():
        df = df.sort_values("時間", kind="stable")
    return (df.tail(n) if newest else df.head(n)).reset_index(drop=True)


# ----------------------------- 資料來源管理 -----------------------------
def source_full(s: dict) -> str:
    """用來比對「產品名稱」篩選的完整名稱,例如 澎南風力氣溫氣壓資料"""
    return f"{s['station']}{s.get('product', '')}"


def source_labels(sources: list) -> list:
    """每個來源在畫面與資料中的顯示名稱;測站名稱重複時加註產品名稱以區分"""
    counts = {}
    for s in sources:
        counts[s["station"]] = counts.get(s["station"], 0) + 1
    return [
        s["station"] if counts[s["station"]] == 1 else f"{s['station']}({s.get('product') or '未命名'})"
        for s in sources
    ]


def _clean_source(d):
    if not isinstance(d, dict):
        return None
    station, url = str(d.get("station", "")).strip(), str(d.get("url", "")).strip()
    if not station or not url.startswith("http"):
        return None
    return {"station": station, "product": str(d.get("product") or DEFAULT_PRODUCT).strip(), "url": url}


def load_sources():
    """讀取 sources.json;不存在就用內建三站。回傳 (來源清單, 警告訊息或 None)"""
    builtin = [dict(s) for s in BUILTIN_SOURCES]
    try:
        data = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return builtin, None
    except (OSError, ValueError) as e:
        return builtin, f"讀取 {SOURCES_FILE.name} 失敗({e}),已改用內建來源。"
    if not isinstance(data, list):
        return builtin, f"{SOURCES_FILE.name} 格式不符,已改用內建來源。"
    return [c for c in (_clean_source(x) for x in data) if c], None


def save_sources(sources: list):
    """寫入 sources.json。成功回傳 None,失敗回傳說明文字"""
    try:
        SOURCES_FILE.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
        return None
    except OSError as e:
        return f"無法寫入 {SOURCES_FILE.name}({e}),新增的來源只在本次工作階段有效。"


def add_source(sources: list, station: str, product: str, url: str):
    """新增(或更新網址)。回傳 (新清單, 動作, 說明);動作為 added / updated / unchanged / rejected"""
    c = _clean_source({"station": station, "product": product, "url": url})
    if c is None:
        return sources, "rejected", "測站名稱不可空白,網址需以 http:// 或 https:// 開頭。"
    for s in sources:
        if s["url"] == c["url"]:
            if s["station"] == c["station"] and s["product"] == c["product"]:
                return sources, "unchanged", f"「{c['station']}」已存在且網址相同,沒有變更。"
            return sources, "rejected", f"這個網址已經用在「{s['station']}({s['product']})」。"
    new = [dict(s) for s in sources]
    for s in new:
        if s["station"] == c["station"] and s["product"] == c["product"]:
            s["url"] = c["url"]
            return new, "updated", f"已更新「{c['station']}」的網址。"
    new.append(c)
    return new, "added", f"已新增「{c['station']}」。"


def parse_bulk(text: str):
    """批次貼上:每行「測站名稱|網址」或「測站名稱|網址|產品名稱」。回傳 (項目清單, 格式錯誤訊息清單)"""
    items, bad = [], []
    for n, line in enumerate((text or "").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1].startswith("http"):
            bad.append(f"第 {n} 行格式不符:{line[:40]}")
            continue
        items.append({
            "station": parts[0], "url": parts[1],
            "product": parts[2] if len(parts) > 2 and parts[2] else DEFAULT_PRODUCT,
        })
    return items, bad


def validate_source(url: str) -> dict:
    """下載並解析一次,確認這個網址真的是可用的 CSV。失敗會丟出例外"""
    content, insecure = fetch_bytes(url)
    raw = read_csv_bytes(content)
    _, used = detect_time(raw)
    _, numeric, _ = prepare(raw, "_", [])
    return {"rows": len(raw), "time_cols": used, "numeric": numeric, "insecure": insecure}


# ----------------------------- 分析用的資料整理 -----------------------------
def to_compass(s: pd.Series) -> pd.Series:
    """風向角度(0~360)轉八方位"""
    idx = np.floor(((s % 360) + 22.5) / 45) % 8
    return idx.map(lambda i: COMPASS[int(i)] if pd.notna(i) else None)


def bin_series(s: pd.Series, var: str, n_bins: int) -> pd.Series:
    """把數值分成區間標籤(風向則轉八方位)。所有測站共用同一組區間,才能互相比較"""
    if "風向" in var:
        return to_compass(s)
    valid = s.dropna()
    if valid.empty:
        return pd.Series([None] * len(s), index=s.index, dtype="object")
    lo, hi = float(valid.min()), float(valid.max())
    if lo == hi:
        return s.map(lambda v: f"{lo:g}" if pd.notna(v) else None)
    edges = np.linspace(lo, hi, n_bins + 1)
    fmt = ".1f" if hi - lo >= 10 else ".2f"
    labels = [f"{edges[i]:{fmt}}~{edges[i + 1]:{fmt}}" for i in range(n_bins)]
    codes = pd.cut(s, bins=edges, labels=False, include_lowest=True)
    return codes.map(lambda i: labels[int(i)] if pd.notna(i) else None)


def line_data(df: pd.DataFrame, var: str, freq_label: str):
    """折線圖資料。回傳 (DataFrame, x 軸欄位名稱)"""
    d = df.dropna(subset=[var])
    if d["時間"].notna().any():
        d = d.dropna(subset=["時間"])
        rule = FREQ_RULES[freq_label]
        if rule:
            d = (d.set_index("時間").groupby("測站")[var]
                  .resample(rule).mean().dropna().reset_index())
        else:
            d = d[["測站", "時間", var]]
        return d, "時間"
    d = d[["測站", var]].copy()
    d["序號"] = d.groupby("測站").cumcount() + 1
    return d, "序號"


def station_summary(df: pd.DataFrame, var: str) -> pd.DataFrame:
    out = df.groupby("測站")[var].agg(
        筆數="count", 平均="mean", 最小="min", 最大="max", 標準差="std"
    ).round(3)                                   # 先只對數值欄四捨五入,再加日期欄
    if df["時間"].notna().any():
        t = df.dropna(subset=[var]).groupby("測站")["時間"]
        out["起"] = t.min()
        out["訖"] = t.max()
    return out.reset_index()


def pie_bins_data(df: pd.DataFrame, binned: pd.Series, station: str) -> pd.DataFrame:
    s = binned if station == ALL_STATIONS else binned[df["測站"] == station]
    c = s.dropna().value_counts()
    return pd.DataFrame({"區間": c.index.astype(str), "筆數": c.values})


def pie_count_data(df: pd.DataFrame, var: str) -> pd.DataFrame:
    return df.groupby("測站")[var].count().reset_index(name="有效筆數")


SUNBURST_PATHS = {
    "測站 → 月份 → 區間": ["測站", "月份", "區間"],
    "月份 → 測站 → 區間": ["月份", "測站", "區間"],
    "測站 → 區間": ["測站", "區間"],
}


def sunburst_data(df: pd.DataFrame, binned: pd.Series, path: list) -> pd.DataFrame:
    tmp = df[["測站", "時間"]].copy()
    tmp["區間"] = binned.astype("object")
    tmp = tmp.dropna(subset=["區間"])
    tmp["區間"] = tmp["區間"].astype(str)
    tmp["月份"] = tmp["時間"].dt.strftime("%Y-%m").fillna("無時間")
    return tmp.groupby(path).size().reset_index(name="筆數")


# ----------------------------- 同時間比較:資料整理 -----------------------------
def guess_column(numeric_cols: list, keys: tuple):
    """依關鍵字猜測欄位(例如「風力」→ 含「風速」的欄位),猜不到回傳 None"""
    for c in numeric_cols:
        if any(k in c.lower() for k in keys):
            return c
    return None


def align_wides(df, variables, stations, rule, t_start, t_end, strict):
    """把各測站資料對齊到同一時間軸。
    回傳 {項目: 寬表(列=時間, 欄=測站)};strict=True 時只保留「所有項目、所有測站」都有資料的時間點"""
    d = df[df["測站"].isin(stations)].dropna(subset=["時間"])
    d = d[(d["時間"] >= pd.Timestamp(t_start)) & (d["時間"] < pd.Timestamp(t_end) + pd.Timedelta(days=1))]
    empty_idx = pd.DatetimeIndex([], name="時間")
    wides = {}
    for var in variables:
        sub = d[["時間", "測站", var]].dropna(subset=[var])
        if sub.empty:
            wides[var] = pd.DataFrame(columns=stations, index=empty_idx, dtype=float)
            continue
        if rule:   # 例如 10 分鐘 / 1 小時 取平均,讓不同步的測站可以對齊
            sub = (sub.set_index("時間").groupby("測站")[var]
                      .resample(rule).mean().dropna().reset_index())
        w = sub.pivot_table(index="時間", columns="測站", values=var, aggfunc="mean")
        wides[var] = w.reindex(columns=stations).sort_index()
    if strict and wides:
        common = None
        for w in wides.values():
            idx = w.dropna(how="any").index
            common = idx if common is None else common.intersection(idx)
        common = common.sort_values()
        wides = {v: w.reindex(common) for v, w in wides.items()}
    return wides


def all_times(wides: dict) -> pd.DatetimeIndex:
    idx = None
    for w in wides.values():
        idx = w.index if idx is None else idx.union(w.index)
    return idx if idx is not None else pd.DatetimeIndex([])


def to_long(wides: dict) -> pd.DataFrame:
    frames = []
    for var, w in wides.items():
        if w.empty:
            continue
        l = (w.rename_axis(index="時間", columns=None).reset_index()
              .melt(id_vars="時間", var_name="測站", value_name="值")
              .dropna(subset=["值"]))
        l["項目"] = var
        frames.append(l)
    if not frames:
        return pd.DataFrame(columns=["時間", "測站", "值", "項目"])
    return pd.concat(frames, ignore_index=True)


def means_table(wides: dict) -> pd.DataFrame:
    """各測站在「比較時間點」上的平均(列=測站,欄=項目)"""
    return pd.DataFrame({v: w.mean() for v, w in wides.items()}).round(3).rename_axis("測站").reset_index()


def pair_stats(wides: dict, stations: list) -> pd.DataFrame:
    """兩兩測站在同時間的差異統計:相關係數、平均差(A−B)、平均絕對差、RMSE"""
    rows = []
    for var, w in wides.items():
        for a, b in itertools.combinations(stations, 2):
            x = w[[a, b]].dropna()
            if len(x) < 2:
                rows.append({"項目": var, "測站A": a, "測站B": b, "共同筆數": len(x)})
                continue
            diff = x[a] - x[b]
            rows.append({
                "項目": var, "測站A": a, "測站B": b, "共同筆數": len(x),
                "相關係數": x[a].corr(x[b]),
                "平均差(A−B)": diff.mean(),
                "平均絕對差": diff.abs().mean(),
                "RMSE": float(np.sqrt((diff ** 2).mean())),
            })
    return pd.DataFrame(rows).round(3)


def clean_manual(manual, variables: list) -> pd.DataFrame:
    """整理使用者手動輸入的對照列:名稱不可空白,數值欄轉數字"""
    cols = ["名稱"] + list(variables)
    if manual is None or len(manual) == 0:
        return pd.DataFrame(columns=cols)
    m = manual.copy()
    m = m[m["名稱"].notna()]
    m["名稱"] = m["名稱"].astype(str).str.strip()
    m = m[~m["名稱"].isin(["", "nan", "None"])]
    for v in variables:
        m[v] = pd.to_numeric(m[v], errors="coerce") if v in m.columns else np.nan
    return m[cols].reset_index(drop=True)


def snapshot_long(wides: dict, t, manual=None) -> pd.DataFrame:
    """指定時間點(取最接近的時間)各測站的數值;可附上手動輸入的對照列"""
    rows = []
    for var, w in wides.items():
        if not w.empty:
            i = w.index.get_indexer([t], method="nearest")[0]
            for station, v in w.iloc[i].items():
                if pd.notna(v):
                    rows.append({"項目": var, "測站": station, "值": float(v), "時間": w.index[i]})
        if manual is not None and len(manual):
            for _, m in manual.iterrows():
                if pd.notna(m[var]):
                    rows.append({"項目": var, "測站": f"{m['名稱']}(手動)", "值": float(m[var]), "時間": pd.NaT})
    return pd.DataFrame(rows, columns=["項目", "測站", "值", "時間"])


# ----------------------------- 圖表 -----------------------------
def style(fig, title: str):
    fig.update_layout(
        title=title,
        font=dict(family=FONT_FAMILY, size=14),   # 用瀏覽器的中文字型,不需下載字型檔
        template="plotly_white",
        margin=dict(t=60, l=20, r=20, b=20),
    )
    return fig


def make_line(df, var, freq_label):
    d, x = line_data(df, var, freq_label)
    if d.empty:
        return None
    fig = px.line(d, x=x, y=var, color="測站", markers=len(d) <= 200)
    return style(fig, f"各測站「{var}」折線圖({freq_label})")


def make_pie_bins(df, binned, var, station):
    d = pie_bins_data(df, binned, station)
    if d.empty:
        return None
    fig = px.pie(d, names="區間", values="筆數")
    fig.update_traces(textinfo="percent+label")
    return style(fig, f"「{var}」區間分布:{station}")


def make_pie_count(df, var):
    d = pie_count_data(df, var)
    if d["有效筆數"].sum() == 0:
        return None
    fig = px.pie(d, names="測站", values="有效筆數")
    fig.update_traces(textinfo="percent+label")
    return style(fig, f"各測站「{var}」有效資料筆數占比")


def make_sunburst(df, binned, var, path):
    d = sunburst_data(df, binned, path)
    if d.empty:
        return None
    fig = px.sunburst(d, path=path, values="筆數")
    fig.update_traces(textinfo="label+percent parent")
    return style(fig, f"「{var}」旭日圖({' → '.join(path)})")


def tidy_facets(fig):
    """分面圖:各自獨立的 Y 軸(單位不同不能共用),並把標題簡化成項目名稱"""
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def make_compare_lines(long_df):
    if long_df.empty:
        return None
    n = long_df["項目"].nunique()
    fig = px.line(long_df, x="時間", y="值", color="測站", facet_row="項目",
                  markers=len(long_df) <= 300, facet_row_spacing=0.08)
    fig.update_layout(height=280 * n + 100)
    return style(tidy_facets(fig), "各測站同時間對照")


def make_corr_heatmap(w, var):
    c = w.corr(min_periods=2)
    if c.isna().all().all():
        return None
    fig = px.imshow(c, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto")
    return style(fig, f"「{var}」站間相關係數")


def make_diff_line(w, var, a, b):
    d = (w[a] - w[b]).dropna()
    if d.empty:
        return None
    fig = px.line(x=d.index, y=d.values, labels={"x": "時間", "y": f"{a} − {b}"},
                  markers=len(d) <= 300)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return style(fig, f"「{var}」{a} − {b} 同時間差值")


def make_snapshot_bar(snap, t):
    if snap.empty:
        return None
    fig = px.bar(snap, x="測站", y="值", color="測站", facet_col="項目", text_auto=".2f")
    fig.update_layout(showlegend=False)
    return style(tidy_facets(fig), f"{t:%Y-%m-%d %H:%M} 附近各測站同時間比較")


# ----------------------------- 介面 -----------------------------
def render_compare(df: pd.DataFrame, numeric_cols: list):
    """「三站同時間比較」分頁:輸入比較條件 → 對齊時間 → 圖表與統計"""
    if df["時間"].notna().sum() == 0:
        st.warning("沒有偵測到時間欄,無法做同時間比較。")
        return
    stations_all = sorted(df["測站"].unique())
    if len(stations_all) < 2:
        st.info("需要至少兩個測站的資料才能比較,請在側邊欄多選幾個觀測站後重新載入。")
        return

    tmin, tmax = df["時間"].min().date(), df["時間"].max().date()
    # 元件的 key 帶上資料特徵:重新載入資料、選項改變時,舊的選取狀態不會與新選項衝突
    sig = hashlib.md5(repr((stations_all, numeric_cols, str(tmin), str(tmax))).encode()).hexdigest()[:8]

    # ---------- 輸入比較條件 ----------
    st.markdown("#### 輸入比較條件")
    st.caption("每個項目請選對應的欄位(已依欄位名稱自動猜測,可自行更改)。")
    opts = [NONE_OPT] + numeric_cols
    slot = {}
    for col, (label, keys) in zip(st.columns(3), VAR_KEYS.items()):
        guess = guess_column(numeric_cols, keys)
        slot[label] = col.selectbox(
            f"{label}欄位", opts, index=opts.index(guess) if guess else 0, key=f"cmp_{label}_{sig}"
        )
    variables = list(dict.fromkeys(v for v in slot.values() if v != NONE_OPT))
    stations = st.multiselect("比較的測站", stations_all, default=stations_all, key=f"cmp_stations_{sig}")

    d1, d2, d3 = st.columns(3)
    rng = d1.date_input("時間範圍", value=(tmin, tmax), min_value=tmin, max_value=tmax, key=f"cmp_range_{sig}")
    align = d2.selectbox("時間對齊方式", list(ALIGN_RULES), key="cmp_align",
                         help="各測站的取樣時間若不同步,選較粗的間隔(取平均)才能對齊。")
    strict = d3.checkbox("只保留所有測站、所有項目都有資料的時間點", value=True, key="cmp_strict")

    if not variables:
        st.info("請至少選擇一個比較項目。")
        return
    if len(stations) < 2:
        st.info("請至少選擇兩個測站。")
        return
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        st.info("時間範圍請選擇「起、迄」兩個日期。")
        return

    wides = align_wides(df, variables, stations, ALIGN_RULES[align], rng[0], rng[1], strict)
    times = all_times(wides)
    if len(times) == 0:
        st.warning(
            "在這些條件下找不到任何「同時間」的資料。可以試試:改用較粗的時間對齊方式(例如 1 小時)、"
            "取消「只保留所有測站都有資料」、放寬時間範圍,或在側邊欄加大「頁數」讓各站的時間範圍有重疊。"
        )
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("同時間的時間點數", f"{len(times):,}")
    m2.metric("比較測站", len(stations))
    m3.metric("比較項目", "、".join(variables))
    st.divider()

    # ---------- 1. 同時間對照折線 ----------
    st.subheader("1. 各測站同時間對照")
    show(make_compare_lines(to_long(wides)), "cmp_line")

    # ---------- 2. 統計比較 ----------
    st.subheader("2. 統計比較")
    st.markdown("**各測站平均(僅計入比較的時間點)**")
    st.dataframe(means_table(wides), hide_index=True)
    st.markdown("**兩兩測站差異**(平均差 = 測站A − 測站B)")
    ps = pair_stats(wides, stations)
    st.dataframe(ps, hide_index=True)
    st.download_button("下載差異統計 CSV", ps.to_csv(index=False).encode("utf-8-sig"),
                       file_name="station_pair_stats.csv", mime="text/csv")
    st.markdown("**站間相關係數**")
    for col, var in zip(st.columns(len(variables)), variables):
        with col:
            show(make_corr_heatmap(wides[var], var), f"cmp_corr_{var}")

    # ---------- 3. 兩站差值 ----------
    st.subheader("3. 兩站差值走勢")
    e1, e2, e3 = st.columns(3)
    vkey, skey = "|".join(variables), "|".join(stations)
    dvar = e1.selectbox("項目", variables, key=f"cmp_dvar_{vkey}")
    sa = e2.selectbox("測站 A", stations, index=0, key=f"cmp_a_{skey}")
    sb = e3.selectbox("測站 B", stations, index=1, key=f"cmp_b_{skey}")
    if sa == sb:
        st.info("測站 A 與 B 請選不同的測站。")
    else:
        show(make_diff_line(wides[dvar], dvar, sa, sb), "cmp_diff")

    # ---------- 4. 指定時間點 + 手動輸入對照 ----------
    st.subheader("4. 指定時間點比較")
    typed = st.text_input("輸入時間(例如 2026-06-01 12:00,會取最接近的時間點;留空則用下方拉桿)",
                          key="cmp_time_text")
    t = None
    if typed.strip():
        parsed = pd.to_datetime(typed.strip(), errors="coerce")
        if pd.isna(parsed):
            st.warning("時間格式無法辨識,改用拉桿。")
        else:
            t = times[times.get_indexer([parsed], method="nearest")[0]]
    if t is None:
        if len(times) > 1:
            i = st.slider("時間點", 1, len(times), max(1, len(times) // 2),
                          key=f"cmp_idx_{len(times)}_{sig}")
            t = times[i - 1]
        else:
            t = times[0]
    st.caption(f"比較時間點:{t:%Y-%m-%d %H:%M:%S}")

    with st.expander("手動輸入對照資料(選用):加入你自己的數值一起比較"):
        st.caption("每列一筆對照資料(例如另一個來源的觀測值)。名稱不可空白;數值欄可留空。")
        base = pd.DataFrame({"名稱": pd.Series(dtype="str"),
                             **{v: pd.Series(dtype="float") for v in variables}})
        edited = st.data_editor(
            base, num_rows="dynamic", key="cmp_manual_" + "|".join(variables),
            column_config={v: st.column_config.NumberColumn(v, format="%.2f") for v in variables},
        )
    manual = clean_manual(edited, variables)
    show(make_snapshot_bar(snapshot_long(wides, t, manual), t), "cmp_snap")


def render_source_manager():
    """側邊欄:新增 / 批次貼上 / 刪除資料來源,變更會存到 sources.json"""
    sources = st.session_state["sources"]
    with st.expander(f"管理資料來源({len(sources)} 個)"):
        st.caption(f"新增的來源會存到 {SOURCES_FILE.name}(與 app.py 同資料夾),下次開啟仍在。"
                   "雲端部署環境的檔案可能在重啟後消失。")

        # ---- 新增單一測站 ----
        st.markdown("**新增測站**")
        with st.form("add_source", clear_on_submit=True):
            name = st.text_input("測站名稱", placeholder="例如:七美")
            prod = st.text_input("資料集產品名稱", value=DEFAULT_PRODUCT)
            url = st.text_input("CSV 網址", placeholder="https://www.namr.gov.tw/filedownload?...")
            check = st.checkbox("新增前先驗證(下載並解析)", value=True)
            submitted = st.form_submit_button("新增")
        if submitted:
            new, action, text = add_source(sources, name, prod, url)
            if check and action in ("added", "updated"):
                try:
                    info = validate_source(url.strip())
                    text += (f"驗證成功:{info['rows']:,} 筆,時間欄「{'、'.join(info['time_cols']) or '未偵測到'}」,"
                             f"{len(info['numeric'])} 個數值欄。")
                except Exception as e:
                    action, text = "rejected", f"驗證失敗,未新增:{e}"
            if action in ("added", "updated"):
                st.session_state["sources"] = sources = new
                err = save_sources(new)
                text += f"(提醒:{err})" if err else ""
                filt = st.session_state.get("product_filter", "").strip().lower()
                entry = next(x for x in new if x["url"] == url.strip())
                if filt and filt not in source_full(entry).lower():
                    text += f"\n\n注意:目前上方「產品名稱」篩選是「{filt}」,這個資料集不符合,所以不會出現在測站清單,請調整篩選。"
            {"added": st.success, "updated": st.success, "unchanged": st.info}.get(action, st.error)(text)

        # ---- 批次貼上 ----
        st.markdown("**批次貼上**")
        bulk = st.text_area("每行一筆:測站名稱|網址(或 測站名稱|網址|產品名稱)", height=90, key="bulk_text")
        if st.button("批次新增", key="bulk_btn"):
            items, bad = parse_bulk(bulk)
            new, counts, skipped = sources, {"added": 0, "updated": 0}, []
            for it in items:
                new, action, text = add_source(new, it["station"], it["product"], it["url"])
                if action in counts:
                    counts[action] += 1
                else:
                    skipped.append(text)
            if counts["added"] or counts["updated"]:
                st.session_state["sources"] = sources = new
                err = save_sources(new)
                if err:
                    st.warning(err)
            st.success(f"新增 {counts['added']} 個、更新 {counts['updated']} 個。批次新增不會先驗證,"
                       "載入時若有問題會逐站顯示錯誤。")
            for m in bad + skipped:
                st.warning(m)

        # ---- 目前的來源(可刪除) ----
        st.markdown("**目前的資料來源**")
        if not sources:
            st.info("目前沒有任何資料來源,請先新增。")
        for s_, lab in zip(sources, source_labels(sources)):
            c1, c2 = st.columns([5, 1])
            c1.caption(f"**{lab}** · {s_['product']}  \n{s_['url'][:45]}...")
            key = "del_" + hashlib.md5(s_["url"].encode()).hexdigest()[:8]
            if c2.button("🗑", key=key, help=f"刪除「{lab}」"):
                st.session_state["sources"] = [x for x in sources if x["url"] != s_["url"]]
                save_sources(st.session_state["sources"])
                st.rerun()
        if st.button("還原為內建三站", key="reset_sources"):
            st.session_state["sources"] = [dict(x) for x in BUILTIN_SOURCES]
            save_sources(st.session_state["sources"])
            st.rerun()


def show(fig, key: str):
    if fig is None:
        st.info("沒有可繪製的資料。")
    else:
        st.plotly_chart(fig, key=key)


def main():
    st.set_page_config(page_title="觀測站風力氣溫氣壓分析", page_icon="🌊", layout="wide")
    if "sources" not in st.session_state:
        st.session_state["sources"], warn = load_sources()
        if warn:
            st.warning(warn)
    st.title("🌊 觀測站 風力・氣溫・氣壓 資料分析")
    st.caption("資料來源:國家海洋研究院開放資料 (namr.gov.tw)")

    # ---- 側邊欄:資料來源與取樣 ----
    with st.sidebar:
        st.header("1. 資料來源")
        product = st.text_input(
            "產品名稱", value=DEFAULT_PRODUCT, key="product_filter",
            help="用來篩選資料集,例如「風力氣溫氣壓資料」會選到全部測站;"
                 "輸入「澎南」只會選到澎南。留空代表全部。",
        )
        render_source_manager()
        sources = st.session_state["sources"]
        labels = source_labels(sources)
        key = product.strip().lower()
        matched = [(lab, s_) for lab, s_ in zip(labels, sources) if key in source_full(s_).lower()]
        if not matched:
            st.warning("沒有符合的資料集,請調整產品名稱或新增資料來源。")
        picked = st.multiselect(
            "要分析的觀測站", options=[lab for lab, _ in matched],
            default=[lab for lab, _ in matched],
        )

        st.header("2. 取樣")
        page_size = st.number_input("每頁筆數", min_value=50, max_value=100000, value=500, step=50)
        pages = st.number_input("頁數", min_value=1, max_value=1000, value=5, step=1)
        newest = st.radio("取樣位置", ["最新資料", "最早資料"], horizontal=True) == "最新資料"
        st.caption(f"每個測站取 {int(page_size * pages):,} 筆(不足則全取)")
        sentinel_text = st.text_input(
            "視為缺失值的代碼(逗號分隔)", value="-9999,-999",
            help="這些數值會被當成缺失,不納入統計與繪圖。",
        )
        load = st.button("載入並分析", type="primary")

    # ---- 載入資料 ----
    if load:
        if not picked:
            st.sidebar.error("請至少選擇一個觀測站。")
        else:
            try:
                sentinels = [float(x) for x in sentinel_text.split(",") if x.strip()]
            except ValueError:
                st.sidebar.error("缺失值代碼必須是數字,以逗號分隔。")
                sentinels = None
            if sentinels is not None:
                frames, numeric_cols, notes, errors, detect = [], [], [], [], []
                bar = st.progress(0.0, text="下載中...")
                chosen = [(lab, s_) for lab, s_ in matched if lab in picked]
                for i, (lab, s_) in enumerate(chosen, start=1):
                    bar.progress((i - 1) / len(chosen), text=f"下載 {lab} ...")
                    try:
                        content, insecure = fetch_bytes(s_["url"])
                        if insecure:
                            notes.append(f"{lab}:網站憑證驗證失敗,已略過 SSL 驗證下載。")
                        raw = read_csv_bytes(content)
                        prep, num, used = prepare(raw, lab, sentinels)
                        total = len(prep)
                        prep = take_rows(prep, int(page_size * pages), newest)
                        frames.append(prep)
                        numeric_cols += [c for c in num if c not in numeric_cols]
                        detect.append({
                            "測站": lab, "原始筆數": total, "取用筆數": len(prep),
                            "時間欄位": "、".join(used) or "(未偵測到)",
                            "數值欄位": "、".join(num),
                        })
                    except Exception as e:   # 單一測站失敗不影響其他測站
                        errors.append(f"{lab}:{e}")
                bar.empty()
                if not frames:
                    st.error("全部下載失敗:\n\n" + "\n\n".join(errors))
                    st.session_state.pop("data", None)
                else:
                    st.session_state["data"] = {
                        "df": pd.concat(frames, ignore_index=True),
                        "numeric": numeric_cols, "notes": notes,
                        "errors": errors, "detect": pd.DataFrame(detect),
                    }

    data = st.session_state.get("data")
    if not data:
        st.info("請在左側設定產品名稱與取樣範圍,然後按「載入並分析」。")
        return

    df, numeric_cols = data["df"], data["numeric"]
    for n in data["notes"]:
        st.warning(n)
    for e in data["errors"]:
        st.error(f"載入失敗 → {e}")
    if not numeric_cols:
        st.error("沒有偵測到數值欄位,請展開下方「欄位偵測結果」檢查檔案內容。")
        st.dataframe(df.head(20))
        return

    has_time = df["時間"].notna().any()
    c1, c2, c3 = st.columns(3)
    c1.metric("測站數", df["測站"].nunique())
    c2.metric("資料筆數", f"{len(df):,}")
    c3.metric("時間範圍", f"{df['時間'].min():%Y-%m-%d} ~ {df['時間'].max():%Y-%m-%d}" if has_time else "未偵測到時間欄")
    with st.expander("欄位偵測結果"):
        st.dataframe(data["detect"], hide_index=True)

    # ---- 分析設定 ----
    s1, s2 = st.columns([2, 1])
    var = s1.selectbox("分析項目(數值欄位)", numeric_cols)
    n_bins = s2.slider("區間數(圓餅圖/旭日圖)", 3, 12, 6, disabled="風向" in var)
    if "風向" in var:
        st.caption("風向欄位視為角度(0~360),圓餅圖與旭日圖會改用八方位分組。")
    binned = bin_series(df[var], var, n_bins)

    st.subheader(f"各測站「{var}」統計")
    st.dataframe(station_summary(df, var), hide_index=True)

    tab_line, tab_pie, tab_sun, tab_cmp, tab_data = st.tabs(
        ["📈 折線圖", "🥧 圓餅圖", "☀️ 旭日圖", "🔍 同時間比較", "📋 資料表"])

    with tab_line:
        freq = st.selectbox("時間彙總", list(FREQ_RULES), index=0, disabled=not has_time)
        fig = make_line(df, var, freq)
        show(fig, "line")

    with tab_pie:
        mode = st.radio("圓餅圖類型", ["數值區間分布", "各測站資料筆數占比"], horizontal=True)
        if mode == "數值區間分布":
            station = st.selectbox("測站", [ALL_STATIONS] + sorted(df["測站"].unique()))
            fig = make_pie_bins(df, binned, var, station)
        else:
            fig = make_pie_count(df, var)
        show(fig, "pie")

    with tab_sun:
        paths = SUNBURST_PATHS if has_time else {"測站 → 區間": SUNBURST_PATHS["測站 → 區間"]}
        choice = st.selectbox("階層", list(paths))
        fig = make_sunburst(df, binned, var, paths[choice])
        show(fig, "sun")
        st.caption("扇形大小 = 該分組的資料筆數;點擊扇形可展開/收合該層。")

    with tab_cmp:
        render_compare(df, numeric_cols)

    with tab_data:
        st.dataframe(df)
        st.download_button(
            "下載 CSV(含所有測站)",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="namr_stations.csv", mime="text/csv",
        )


if __name__ == "__main__":
    main()
