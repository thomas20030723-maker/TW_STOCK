"""
台股動能/RSI 輪動策略 v2（核心 0050 + 動能衛星）：回測 + 選股
規格參照貼文：訓練期找參數、驗證期分開報告、與 0050 買進持有比較，
參數最多 3 個、每個最多 5 種值，先列偷看未來風險再跑結果。

用法：
    pip install yfinance pandas numpy
    python tw_momentum.py            # 下載資料 → 訓練期網格搜尋 → 驗證期報告 → 今日選股
    python tw_momentum.py --offline  # 只用 ./data 快取

======================== 可能偷看未來的地方 & 避免方式 ========================
1. 當天收盤算訊號、當天收盤成交 → 訊號用 t 日收盤，部位從 t+1 日收盤才建立（shift 1 天）。
2. 指標用到未來資料（如 centered rolling、全期標準化）→ 只用 rolling 往回看的視窗，不做全期 z-score。
3. 參數在全期挑選 → 網格搜尋只用訓練期報酬；驗證期只跑一次、不回頭改參數。
4. 驗證期第一天的指標需要訓練期資料暖機 → 允許用過去資料暖機（這是合法的「過去」），
   但驗證期績效只從 VALID_START 起算，且起算前一天的部位就由當時訊號決定。
5. 倖存者偏差：股票池用「今天」的權值股 → 會高估績效，無法用免費資料完全避免；
   緩解：股票池放大到 ~50 檔且含已衰退的老權值股，報告時註明此偏差。
6. 除權息/分割：用 yfinance auto_adjust 還原價，避免除息日假跌觸發訊號；
   還原價本身以「事後」配息回推，但只影響價格水準，不影響當時可知的報酬率序列。
7. 新上市股：上市前無資料（NaN）→ 排名時直接排除，不以 0 或回填值代替。
8. 大盤濾網用 0050 自身 200MA → 同樣 shift 1 天，且此規則固定、不參與調參。
9. 交易成本：手續費 0.1425% 雙邊 + 證交稅 0.3% 賣出，避免高換手策略被高估。
==============================================================================
"""
import argparse, itertools, os
import numpy as np, pandas as pd

TRAIN_START, TRAIN_END = "2008-01-01", "2021-12-31"   # 台股免費資料約 2000s 起，無法回到 1980
VALID_START = "2022-01-01"
BENCH = "0050.TW"
UNIVERSE = """2330 2317 2454 2308 2382 2881 2882 2891 2303 3711 2412 2886 2884 1216 2885
2357 3231 2892 2345 2880 5880 2883 2887 2890 3008 2327 2379 3034 3037 2395 1303 1301 2002
1326 6505 2207 2912 4904 3045 2603 2609 2615 5871 2301 6669 3661 2408 2344 1101 2105 9910""".split()
TICKERS = [t + ".TW" for t in UNIVERSE]

# ---- 參數網格（3 個參數 × 5 值）----
GRID = {
    "lookback": [60, 90, 120, 180, 250],   # 動能回看天數（跳過最近 21 天）
    "top_k":    [3, 5, 8, 10, 15],         # 衛星持有檔數
    "core_w":   [0.0, 0.25, 0.5, 0.75, 1.0],  # 多頭時 0050 核心部位比例（1.0 = 純 0050 擇時）
}
# ---- 固定規則（事前決定、不調參）----
REBAL = 21          # 每 21 交易日換股
MA_FILTER = 200     # 0050 < 200MA → 全部空手
SKIP = 21           # 動能跳過最近一個月（避開短期反轉）
VOL_N = 60          # 動能除以 60 日波動度（風險調整動能）
RSI_MAX = 80        # 排除極端過熱
BUFFER = 2          # 已持有者只要仍在前 BUFFER*top_k 名就續抱（降換手）
BUY_COST, SELL_COST = 0.001425, 0.001425 + 0.003


def load_prices(offline=False, cache="data"):
    os.makedirs(cache, exist_ok=True)
    f = os.path.join(cache, "close.csv")
    if offline:
        if os.path.exists(f):
            return pd.read_csv(f, index_col=0, parse_dates=True)
        raise SystemExit("找不到 data/close.csv")
    import yfinance as yf
    df = yf.download(TICKERS + [BENCH], start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    df = df.dropna(how="all")
    df.to_csv(f)
    return df


def rsi(px, n=14):
    d = px.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def scores(px, lookback):
    stocks = px.drop(columns=[BENCH])
    mom = stocks.shift(SKIP) / stocks.shift(lookback) - 1
    vol = stocks.pct_change().rolling(VOL_N).std() * np.sqrt(252)
    return mom, mom / vol, rsi(stocks)


def target_weights(px, lookback, top_k, core_w):
    stocks = px.drop(columns=[BENCH])
    mom, score, r = scores(px, lookback)
    bench = px[BENCH]
    risk_on = bench > bench.rolling(MA_FILTER).mean()
    w = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    held = []
    for i in range(max(lookback, MA_FILTER, VOL_N), len(px), REBAL):
        row = pd.Series(0.0, index=px.columns)
        if risk_on.iloc[i]:
            s = score.iloc[i].where((r.iloc[i] < RSI_MAX) & (mom.iloc[i] > 0)).dropna().sort_values(ascending=False)
            keep = [t for t in held if t in s.index[:BUFFER * top_k]]
            picks = (keep + [t for t in s.index if t not in keep])[:top_k]
            row[BENCH] = core_w
            if picks:
                row[picks] = (1 - core_w) / top_k
            held = picks
        else:
            held = []
        w.iloc[i] = row
    return w.ffill().fillna(0.0)


def backtest(px, w, start, end=None):
    ret = px[w.columns].pct_change().fillna(0.0)
    pos = w.shift(1).fillna(0.0)                      # t 日訊號 → t+1 日持有（防偷看 #1）
    drift = pos.diff().fillna(pos)
    cost = drift.clip(lower=0).sum(1) * BUY_COST + (-drift.clip(upper=0)).sum(1) * SELL_COST
    daily = (pos * ret).sum(1) - cost
    sl = slice(start, end)
    return daily.loc[sl], pos.loc[sl], ret.loc[sl]


def trades(pos, ret):
    out = []
    for c in pos.columns:
        held = pos[c] > 0
        if not held.any():
            continue
        grp = (held != held.shift()).cumsum()[held]
        for _, idx in grp.groupby(grp).groups.items():
            out.append(((1 + ret.loc[idx, c]).prod() - 1 - BUY_COST - SELL_COST, len(idx)))
    return pd.DataFrame(out, columns=["ret", "days"])


def metrics(daily, pos=None, ret=None):
    eq = (1 + daily).cumprod()
    yrs = len(daily) / 252
    m = {
        "年化報酬": eq.iloc[-1] ** (1 / yrs) - 1,
        "最大回撤": (eq / eq.cummax() - 1).min(),
        "夏普值": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
    }
    m["Calmar"] = m["年化報酬"] / -m["最大回撤"] if m["最大回撤"] < 0 else np.nan
    if pos is not None:
        t = trades(pos, ret)
        win, loss = t.ret[t.ret > 0], t.ret[t.ret <= 0]
        m.update({
            "勝率": len(win) / len(t) if len(t) else np.nan,
            "盈虧比": win.mean() / -loss.mean() if len(loss) and len(win) else np.nan,
            "交易次數": len(t),
            "平均持有天數": t.days.mean() if len(t) else np.nan,
            "暴露時間": (pos.sum(1) > 0).mean(),
        })
    return m


def fmt(m):
    pct = {"年化報酬", "最大回撤", "勝率", "暴露時間"}
    return {k: (f"{v:.1%}" if k in pct else f"{v:.2f}" if isinstance(v, float) else v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    px = load_prices(a.offline)

    # ---- 1) 只用訓練期挑參數（目標：夏普值）----
    rows = []
    for lb, k, rm in itertools.product(*GRID.values()):
        w = target_weights(px, lb, k, rm)
        d, _, _ = backtest(px, w, TRAIN_START, TRAIN_END)
        rows.append({"lookback": lb, "top_k": k, "core_w": rm, **metrics(d)})
    grid = pd.DataFrame(rows).sort_values("Calmar", ascending=False)   # 目標：報酬/回撤
    grid.to_csv("grid_train.csv", index=False)
    best = grid.iloc[0]
    p = dict(lookback=int(best.lookback), top_k=int(best.top_k), core_w=float(best.core_w))
    print("最佳參數（訓練期 Calmar）:", p)
    import json; json.dump(p, open("params.json", "w"))
    print("\n訓練期前 10 名組合：\n", grid.head(10).round(3).to_string(index=False))

    # ---- 2) 訓練期 / 驗證期分開報告，並與 0050 比較 ----
    w = target_weights(px, **p)
    bret = px[BENCH].pct_change().fillna(0.0)
    report = {}
    for name, s, e in [("訓練期", TRAIN_START, TRAIN_END), ("驗證期", VALID_START, None)]:
        d, pos, r = backtest(px, w, s, e)
        report[f"策略-{name}"] = fmt(metrics(d, pos, r))
        report[f"0050-{name}"] = fmt(metrics(bret.loc[s:e]))
    rep = pd.DataFrame(report)
    print("\n", rep.to_string())
    rep.to_csv("report.csv")

    # ---- 3) 驗證期有沒有變差 ----
    tr, va = report["策略-訓練期"], report["策略-驗證期"]
    print(f"\n驗證期 vs 訓練期：夏普 {tr['夏普值']} → {va['夏普值']}，"
          f"年化 {tr['年化報酬']} → {va['年化報酬']}，回撤 {tr['最大回撤']} → {va['最大回撤']}")
    # 參數穩健度：最佳參數鄰近組合的夏普中位數，太低代表過擬合
    print("訓練期所有組合 Calmar 中位數:", round(grid["Calmar"].median(), 2))
    print(f"驗證期超額年化 vs 0050：{metrics(backtest(px, w, VALID_START)[0])['年化報酬'] - metrics(bret.loc[VALID_START:])['年化報酬']:+.1%}")

    # ---- 4) 今日選股（用最新收盤資料，明天執行）----
    mom, score, r = scores(px, p["lookback"])
    risk_on = px[BENCH].iloc[-1] > px[BENCH].rolling(MA_FILTER).mean().iloc[-1]
    scr = pd.DataFrame({"動能": mom.iloc[-1], "風險調整動能": score.iloc[-1], "RSI14": r.iloc[-1]}).dropna()
    scr = scr[(scr.RSI14 < RSI_MAX) & (scr["動能"] > 0)].sort_values("風險調整動能", ascending=False)
    print(f"\n資料日 {px.index[-1].date()}｜大盤濾網：{'多頭' if risk_on else '空頭，建議空手'}｜0050 核心 {p['core_w']:.0%}")
    print(scr.head(p["top_k"]).round(3).to_string())
    print("\n目標權重：\n", w.iloc[-1][w.iloc[-1] > 0].round(3).to_string())
    scr.to_csv("screen_today.csv")


if __name__ == "__main__":
    main()
