"""
隔日當沖作多候選：收盤後選股 → 隔天開盤買、收盤前賣（盤中觸停損就出）
回測規格同 tw_momentum：訓練期調參、驗證期分開報告、3 參數 × 5 值、先列偷看未來風險。

======================== 可能偷看未來的地方 & 避免方式 ========================
1. 用當天 K 棒選當天的股票 → 訊號全部用 t-1 日（含）以前的資料，交易在 t 日。
2. 開盤價決定要不要進場 → 只用「跳空幅度」過濾，開盤當下就能知道，不算偷看。
3. 日 K 看不到盤中順序：同一天 high、low 都碰到時不知道先碰哪個
   → 一律假設先碰停損（保守），不做停利。
4. 停損用 low 判斷成交 → 假設停損價成交，但實際跳空/急殺會更差，另扣滑價 0.1%。
5. 參數在全期挑 → 只用訓練期，驗證期跑一次不回頭改。
6. 倖存者偏差：股票池用今天的名單 → 會高估，註明。
7. 成本：手續費 0.1425% ×2（可設折扣）＋當沖證交稅 0.15%＋滑價 0.1%。
==============================================================================
"""
import itertools, json, os, sys, html
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
import tw_momentum as tm

GRID = {
    "vol_ratio": [1.5, 2.0, 2.5, 3.0, 4.0],      # 前一日量 / 20 日均量 下限
    "close_pos": [0.6, 0.7, 0.8, 0.9, 0.95],     # 前一日收盤在當日高低區間的位置下限（1 = 收最高）
    "stop":      [0.01, 0.015, 0.02, 0.03, 1.0], # 盤中停損（1.0 = 不停損，收盤出）
}
TOP_N = 3            # 每天最多做 3 檔（固定）
MAX_GAP = 0.04       # 開盤跳空 > 4% 不追（固定）
FINMIND = "https://api.finmindtrade.com/api/v4/data"
MIN_TURNOVER = 1e8   # 20 日平均成交值 ≥ 1 億元（當時可知的流動性，固定）
MIN_PRICE = 10       # 股價 ≥ 10 元
DT_TRAIN_START = "2011-01-01"   # 中小型股資料從 2010 起下載
FEE_DISCOUNT = 1.0   # 券商手續費折扣，例如 2.8 折填 0.28
COST = 0.001425 * FEE_DISCOUNT * 2 + 0.0015 + 0.001   # 手續費雙邊 + 當沖稅 + 滑價


def load_ohlcv(offline=False, cache="data"):
    f = os.path.join(cache, "ohlcv.pkl")
    if offline:
        return pd.read_pickle(f)
    import yfinance as yf
    raw = yf.download(tm.TICKERS, start="2003-07-01", auto_adjust=False, progress=False)
    d = {k: raw[k].dropna(how="all") for k in ["Open", "High", "Low", "Close", "Volume"]}
    now = datetime.now(timezone(timedelta(hours=8)))
    if d["Close"].index[-1].date() == now.date() and (now.hour, now.minute) < (14, 30):
        d = {k: v.iloc[:-1] for k, v in d.items()}
    os.makedirs(cache, exist_ok=True)
    pd.to_pickle(d, f)
    return d


def get_universe(cache="data"):
    """全部上市櫃普通股，排除 0050 權值股（=中小型）。名單用 FinMind TaiwanStockInfo，失敗就用快取。"""
    f = os.path.join(cache, "universe_small.csv")
    try:
        import requests
        js = requests.get(FINMIND, params=dict(dataset="TaiwanStockInfo", token=os.environ.get("FINMIND_TOKEN", "")), timeout=60).json()
        u = pd.DataFrame(js["data"])
        u = u[u.stock_id.str.fullmatch(r"[1-9]\d{3}") & u.type.isin(["twse", "tpex"])]
        u = u[~u.industry_category.astype(str).str.contains("ETF|受益|存託|指數")]
        u = u[~u.stock_id.isin(tm.UNIVERSE)].drop_duplicates("stock_id")
        u["ticker"] = u.stock_id + np.where(u.type == "twse", ".TW", ".TWO")
        u = u[["ticker", "stock_id", "stock_name"]]
        os.makedirs(cache, exist_ok=True)
        u.to_csv(f, index=False)
    except Exception as e:
        print("股票清單下載失敗，用快取：", e)
        u = pd.read_csv(f, dtype=str)
    return u


def load_small(full, offline=False, cache="data"):
    f = os.path.join(cache, "ohlcv_small.pkl")
    if offline:
        return pd.read_pickle(f)
    import yfinance as yf
    u = get_universe(cache)
    start = "2010-01-01" if full else (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
    parts = {k: [] for k in ["Open", "High", "Low", "Close", "Volume"]}
    tick = u.ticker.tolist()
    for i in range(0, len(tick), 200):
        raw = yf.download(tick[i:i + 200], start=start, auto_adjust=False, progress=False, threads=True)
        for k in parts:
            if k in raw:
                parts[k].append(raw[k].astype("float32"))
    d = {k: pd.concat(v, axis=1).sort_index().dropna(how="all", axis=1) for k, v in parts.items()}
    cols = d["Close"].columns
    d = {k: v.reindex(columns=cols) for k, v in d.items()}
    b = yf.download("0050.TW", start=start, auto_adjust=True, progress=False)["Close"]
    d["bench"] = (b.iloc[:, 0] if isinstance(b, pd.DataFrame) else b).reindex(d["Close"].index).ffill()
    now = datetime.now(timezone(timedelta(hours=8)))
    if d["Close"].index[-1].date() == now.date() and (now.hour, now.minute) < (14, 30):
        d = {k: v.iloc[:-1] for k, v in d.items()}
    d["names"] = dict(zip(u.ticker, u.stock_name))
    os.makedirs(cache, exist_ok=True)
    pd.to_pickle(d, f)
    return d


# ---- 加強條件候選（門檻事前固定，不調參；只在訓練期逐一測試，有改善才採用）----
OPEN_FILTER = "進場日開盤≥平盤"
FILTERS = {
    "前日漲幅 2%~9%（不追漲停）": lambda f: (f["ret1"] >= 0.02) & (f["ret1"] < 0.09),
    "大盤(0050)站上 20MA":       lambda f: f["mkt"],
    "收盤創 20 日新高":           lambda f: f["C"] >= f["hh20"],
    "前日振幅 3%~10%":            lambda f: (f["rng"] >= 0.03) & (f["rng"] <= 0.10),
    "未連漲 4 天以上":            lambda f: f["streak"] < 4,
    OPEN_FILTER:                  None,     # 開盤當下才知道，在 backtest 進場時判斷
}
MIN_IMPROVE = 0.0005   # 訓練期平均每筆至少改善 0.05% 才採用


def features(d):
    C, H, L = d["Close"], d["High"], d["Low"]
    up = (C > C.shift(1)).astype(int)
    cs = up.cumsum()
    streak = cs - cs.where(up == 0).ffill().fillna(0)      # 連續收紅天數
    b = d.get("bench")
    mkt = (b > b.rolling(20).mean()).reindex(C.index).ffill().fillna(False) if b is not None else pd.Series(True, index=C.index)
    return {"C": C, "ret1": C / C.shift(1) - 1, "hh20": C.rolling(20).max(), "rng": (H - L) / C.shift(1),
            "streak": streak, "mkt": pd.DataFrame(np.repeat(mkt.values[:, None], C.shape[1], 1), index=C.index, columns=C.columns)}


def signals(d, vol_ratio, close_pos, filters=()):
    """t 日收盤後算出的候選（隔天 t+1 交易）"""
    O, H, L, C, V = (d[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    vr = V / V.rolling(20).mean().shift(1)
    cp = (C - L) / (H - L).replace(0, np.nan)
    trend = C > C.rolling(20).mean()
    up = C > C.shift(1)
    liq = (C * V).rolling(20).mean() >= MIN_TURNOVER
    ok = (vr >= vol_ratio) & (cp >= close_pos) & trend & up & (V > 500_000) & liq & (C >= MIN_PRICE)
    if filters:
        f = features(d)
        for name in filters:
            if FILTERS.get(name) is not None:
                ok &= FILTERS[name](f)
    score = vr.where(ok)
    return score, vr, cp


def backtest(d, vol_ratio, close_pos, stop, start, end=None, filters=()):
    O, L, C = d["Open"], d["Low"], d["Close"]
    score, _, _ = signals(d, vol_ratio, close_pos, filters)
    score = score.shift(1)                                   # 昨天的訊號 → 今天交易（防偷看 #1）
    gap = O / C.shift(1) - 1
    stop_px = O * (1 - stop)
    hit = L <= stop_px                                       # 盤中觸停損（假設先碰停損 #3）
    r = np.where(hit, -stop, C / O - 1) - COST
    r = pd.DataFrame(r, index=O.index, columns=O.columns)
    score = score.where(gap.abs() <= MAX_GAP)
    if OPEN_FILTER in filters:
        score = score.where(gap >= 0)
    rank = score.rank(axis=1, ascending=False, method="first")
    pick = rank <= TOP_N
    sl = slice(start, end)
    pick, r = pick.loc[sl], r.loc[sl]
    trades = r.where(pick).stack().dropna()
    # 每天資金平均分給 TOP_N 檔，不足的放現金
    daily = (r.where(pick).fillna(0).sum(axis=1)) / TOP_N
    return daily, trades


def metrics(daily, trades):
    eq = (1 + daily).cumprod()
    yrs = len(daily) / 252
    win, loss = trades[trades > 0], trades[trades <= 0]
    return {
        "年化報酬": eq.iloc[-1] ** (1 / yrs) - 1,
        "最大回撤": (eq / eq.cummax() - 1).min(),
        "夏普值": daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan,
        "勝率": len(win) / len(trades) if len(trades) else np.nan,
        "盈虧比": win.mean() / -loss.mean() if len(win) and len(loss) else np.nan,
        "交易次數": len(trades),
        "平均每筆": trades.mean() if len(trades) else np.nan,
        "有交易天數比例": (daily != 0).mean(),
    }


def main():
    offline = "--offline" in sys.argv
    pf, rf = "params_dt.json", "dt_results.json"
    p = json.load(open(pf)) if os.path.exists(pf) else {}
    retune = p.get("version") != 2 or not os.path.exists(rf)
    d = load_small(full=retune, offline=offline)
    if retune:
        rows = []
        for vr, cp, st in itertools.product(*GRID.values()):
            dl, tr = backtest(d, vr, cp, st, DT_TRAIN_START, tm.TRAIN_END)
            if len(tr) >= 100:
                rows.append({"vol_ratio": vr, "close_pos": cp, "stop": st, **metrics(dl, tr)})
        grid = pd.DataFrame(rows).sort_values("夏普值", ascending=False)
        grid.to_csv("grid_daytrade.csv", index=False)
        b = grid.iloc[0]
        p = dict(vol_ratio=float(b.vol_ratio), close_pos=float(b.close_pos), stop=float(b.stop), universe="small", version=2)
        # ---- 加強條件：只用訓練期，逐一加上去看平均每筆有沒有變好 ----
        args = (p["vol_ratio"], p["close_pos"], p["stop"], DT_TRAIN_START, tm.TRAIN_END)
        base = metrics(*backtest(d, *args))
        abl = [{"條件": "（基本條件）", "筆數": base["交易次數"], "勝率": base["勝率"], "平均每筆": base["平均每筆"], "採用": "—"}]
        good = []
        for name in FILTERS:
            m = metrics(*backtest(d, *args, filters=(name,)))
            ok = m["交易次數"] >= 100 and m["平均每筆"] - base["平均每筆"] >= MIN_IMPROVE
            abl.append({"條件": name, "筆數": m["交易次數"], "勝率": m["勝率"], "平均每筆": m["平均每筆"], "採用": "✓" if ok else "×"})
            if ok:
                good.append((m["平均每筆"], name))
        chosen = [n for _, n in sorted(good, reverse=True)]
        while chosen:                      # 合併後筆數太少就拿掉最弱的
            m = metrics(*backtest(d, *args, filters=tuple(chosen)))
            if m["交易次數"] >= 100:
                break
            chosen.pop()
        p["filters"] = chosen
        json.dump(p, open(pf, "w"), ensure_ascii=False)
        res = {"消融": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()} for r in abl]}
        for name, s0, e in [("訓練期", DT_TRAIN_START, tm.TRAIN_END), ("驗證期", tm.VALID_START, None)]:
            res[name] = {k: float(v) for k, v in metrics(*backtest(d, p["vol_ratio"], p["close_pos"], p["stop"], s0, e, filters=tuple(chosen))).items()}
        res["驗證期"]["截至"] = str(d["Close"].index[-1].date())
        json.dump(res, open(rf, "w"), ensure_ascii=False)
    else:
        res = json.load(open(rf))       # 每日只更新候選；回測結果沿用調參當時（避免每天下載 15 年資料）
    print("當沖參數:", p)
    for k in ["訓練期", "驗證期"]:
        print(k, res[k])
    names = d.pop("names", {})
    flt = tuple(p.get("filters", []))
    global NAME_MAP; NAME_MAP = names

    # ---- 明日候選 ----
    score, vr, cp = signals(d, p["vol_ratio"], p["close_pos"], flt)
    last = score.index[-1]
    s = score.iloc[-1].dropna().sort_values(ascending=False)
    C = d["Close"].iloc[-1]
    cands = [(t, C[t], vr.iloc[-1][t], cp.iloc[-1][t]) for t in s.index[:TOP_N]]
    watch = [(t, C[t], vr.iloc[-1][t], cp.iloc[-1][t]) for t in s.index[TOP_N:TOP_N + 5]]
    # ---- 今日診斷：每個條件各有幾檔通過、最接近的股票卡在哪 ----
    O, H, L, Cc, V = (d[k].iloc[-1] for k in ["Open", "High", "Low", "Close", "Volume"])
    ma20 = d["Close"].rolling(20).mean().iloc[-1]
    prev = d["Close"].iloc[-2]
    conds = pd.DataFrame({
        f"量比≥{p['vol_ratio']}": vr.iloc[-1] >= p["vol_ratio"],
        f"收盤位置≥{p['close_pos']:.0%}": cp.iloc[-1] >= p["close_pos"],
        "站上20MA": Cc > ma20,
        "收紅": Cc > prev,
        "量≥500張": V > 500_000,
        "成交值≥1億": (d["Close"] * d["Volume"]).rolling(20).mean().iloc[-1] >= MIN_TURNOVER,
        "股價≥10": Cc >= MIN_PRICE,
    })
    if flt:
        f = features(d)
        for n in flt:
            if FILTERS.get(n) is not None:
                conds[n] = FILTERS[n](f).iloc[-1]
    conds = conds[Cc.notna()]
    diag = {"counts": conds.sum().to_dict(), "all": int(conds.all(axis=1).sum()), "n": len(conds)}
    near = conds.sum(axis=1).sort_values(ascending=False).head(8).index
    diag["near"] = [(t, vr.iloc[-1][t], cp.iloc[-1][t], Cc[t] / prev[t] - 1,
                     [c for c in conds.columns if not conds.loc[t, c]]) for t in near]
    render(p, last, cands, watch, res, diag)


NAME_MAP = {}


def diag_html(diag, nm):
    if not diag:
        return ""
    cnt = "".join(f"<tr><td>{html.escape(k)}</td><td class=num>{v} / {diag['n']}</td></tr>" for k, v in diag["counts"].items())
    cnt += f"<tr><td><b>全部同時符合</b></td><td class=num><b>{diag['all']}</b></td></tr>"
    near = "".join(
        f"<tr><td>{html.escape(nm(t))}</td><td class=num>{v:.1f}x</td><td class=num>{q:.0%}</td><td class=num>{r:+.1%}</td>"
        f"<td>{html.escape('、'.join(f) or '全部符合')}</td></tr>" for t, v, q, r, f in diag["near"])
    return (f"<section class=card><h2>今日診斷：為什麼選到 / 沒選到</h2><div class=wrap><table>"
            f"<tr><th>條件</th><th class=num>通過檔數</th></tr>{cnt}</table></div>"
            f"<h2 style='margin-top:14px'>最接近的股票</h2><div class=wrap><table>"
            f"<tr><th>股票</th><th class=num>量比</th><th class=num>收盤位置</th><th class=num>漲跌</th><th>沒通過</th></tr>{near}</table></div></section>")


def abl_html(abl, p):
    if not abl:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(r['條件'])}</td><td class=num>{int(r['筆數'])}</td><td class=num>{r['勝率']:.0%}</td>"
        f"<td class=num>{r['平均每筆']:+.2%}</td><td>{r['採用']}</td></tr>" for r in abl)
    used = "、".join(p.get("filters", [])) or "無（加任何條件都沒有明顯改善）"
    return (f"<h2 style='margin-top:16px'>加強條件測試（只用訓練期）</h2><div class=wrap><table>"
            f"<tr><th>條件</th><th class=num>筆數</th><th class=num>勝率</th><th class=num>平均每筆</th><th>採用</th></tr>{rows}</table></div>"
            f"<p class=hint>每個條件單獨加在基本條件上測試，平均每筆改善 ≥ {MIN_IMPROVE:.2%} 且筆數 ≥ 100 才採用。目前採用：{html.escape(used)}。上方驗證期是加上採用條件後的樣本外結果。</p>")


def render(p, last, cands, watch, res, diag=None):
    nm = lambda t: f"{t.split('.')[0]} {NAME_MAP.get(t, '')}"
    stop_txt = "不停損，收盤前出場" if p["stop"] >= 1 else f"開盤價 −{p['stop']:.1%}"
    def rows(lst):
        if not lst:
            return "<tr><td colspan=6 class=empty>明天沒有符合條件的股票，休息</td></tr>"
        out = ""
        for t, c, v, q in lst:
            lim = c * (1 + MAX_GAP)
            sp = "—" if p["stop"] >= 1 else f"{c * (1 - p['stop']):.2f}*"
            out += (f"<tr><td>{html.escape(nm(t))}</td><td class=num>{c:.2f}</td><td class=num>{v:.1f}x</td>"
                    f"<td class=num>{q:.0%}</td><td class=num>≤ {lim:.2f}</td><td class=num>{sp}</td></tr>")
        return out
    def kpi(k):
        m = res[k]
        return (f"<tr><td>{k}</td><td class=num>{m['年化報酬']:.1%}</td><td class=num>{m['最大回撤']:.1%}</td>"
                f"<td class=num>{m['夏普值']:.2f}</td><td class=num>{m['勝率']:.0%}</td><td class=num>{m['盈虧比']:.2f}</td>"
                f"<td class=num>{m['平均每筆']:+.2%}</td><td class=num>{int(m['交易次數'])}</td></tr>")
    tw = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    th = "<tr><th>股票</th><th class=num>收盤</th><th class=num>量比</th><th class=num>收盤位置</th><th class=num>開盤不追高於</th><th class=num>停損</th></tr>"
    out = f"""<!doctype html><html lang=zh-Hant><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>中小型當沖候選</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#77756f;--line:#e4e2dc;--acc:#2f5d8a;--warn:#a05a00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--fg:#ecebe6;--mut:#9a988f;--line:#33322e;--acc:#7fa9d6;--warn:#e0a050}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"PingFang TC","Noto Sans TC",sans-serif}}
main{{max-width:760px;margin:0 auto;padding:20px 16px 48px}}h1{{font-size:22px;margin:0}}h2{{font-size:15px;margin:0 0 10px}}
.sub,.hint{{color:var(--mut);font-size:13px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:7px 6px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--mut);font-weight:500;font-size:12px}}.num{{text-align:right;font-variant-numeric:tabular-nums}}.wrap{{overflow-x:auto}}
.empty{{color:var(--mut);text-align:center}}.rule{{font-size:13px;line-height:1.8}}.warn{{color:var(--warn);font-size:13px}}a{{color:var(--acc)}}
</style></head><body><main>
<h1>中小型股・隔日當沖多單</h1>
<div class=sub>資料日 {last.date()}｜更新於 {tw}（台北）｜<a href="index.html">波段選股</a>｜<a href="short.html">籌碼空單</a></div>
<section class=card><h2>明天做多候選</h2><div class=wrap><table>{th}{rows(cands)}</table></div>
<div class=rule><b>操作規則</b><br>
① 開盤價買進；開盤跳空超過 ±{MAX_GAP:.0%} 就放棄這檔{'；<b>開盤低於平盤也放棄</b>' if OPEN_FILTER in p.get('filters', []) else ''}<br>
② 停損：{stop_txt}　③ 沒停損就在收盤前（13:25 前）賣出，不留倉<br>
<span class=hint>* 停損價以今日收盤估算，實際以明天開盤價 ×(1−{p['stop'] if p['stop'] < 1 else 0:.1%}) 為準。資金平均分 {TOP_N} 份。</span></div></section>
<section class=card><h2>備選（前 3 檔放棄時遞補）</h2><div class=wrap><table>{th}{rows(watch)}</table></div></section>
{diag_html(diag, nm)}
<section class=card><h2>回測（每筆已扣手續費、當沖稅、滑價）</h2><div class=wrap><table>
<tr><th></th><th class=num>年化</th><th class=num>最大回撤</th><th class=num>夏普</th><th class=num>勝率</th><th class=num>盈虧比</th><th class=num>平均每筆</th><th class=num>筆數</th></tr>
{kpi('訓練期')}{kpi('驗證期')}</table></div>
{abl_html(res.get('消融'), p)}
<p class=hint>參數：量比 ≥ {p['vol_ratio']}、收盤位置 ≥ {p['close_pos']:.0%}、停損 {stop_txt}。訓練期 {DT_TRAIN_START[:4]}–{tm.TRAIN_END[:4]} 挑參數，驗證期 {tm.VALID_START[:4]}–{res['驗證期'].get('截至', '')[:7]} 樣本外（回測只在重新調參時更新）。<br>
股票池：上市櫃普通股排除 0050 成分股，且當時 20 日均成交值 ≥ 1 億、股價 ≥ 10 元。</p><p class=hint></p>
<p class=warn>日 K 回測看不到盤中走勢，實際滑價與停損成交通常更差；驗證期「平均每筆」若接近 0 或為負，代表扣完成本後沒有優勢，不建議實單。</p></section>
<p class=sub>僅供研究，非投資建議。當沖須先開通資格並簽署風險預告書。</p>
</main></body></html>"""
    os.makedirs("docs", exist_ok=True)
    open("docs/daytrade.html", "w", encoding="utf-8").write(out)
    print("written docs/daytrade.html")


if __name__ == "__main__":
    main()
