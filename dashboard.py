"""
每日選股看板：下載最新收盤 → 用固定參數 (params.json) 算持股 → 輸出 docs/index.html
參數只在跑 tw_momentum.py 時用訓練期決定，這裡不重新調參（避免每天偷偷過擬合）。
"""
import json, os, sys, html
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd
import tw_momentum as tm

NAMES = {"2330":"台積電","2317":"鴻海","2454":"聯發科","2308":"台達電","2382":"廣達","2881":"富邦金","2882":"國泰金",
"2891":"中信金","2303":"聯電","3711":"日月光投控","2412":"中華電","2886":"兆豐金","2884":"玉山金","1216":"統一",
"2885":"元大金","2357":"華碩","3231":"緯創","2892":"第一金","2345":"智邦","2880":"華南金","5880":"合庫金",
"2883":"凱基金","2887":"台新金","2890":"永豐金","3008":"大立光","2327":"國巨","2379":"瑞昱","3034":"聯詠",
"3037":"欣興","2395":"研華","1303":"南亞","1301":"台塑","2002":"中鋼","1326":"台化","6505":"台塑化","2207":"和泰車",
"2912":"統一超","4904":"遠傳","3045":"台灣大","2603":"長榮","2609":"陽明","2615":"萬海","5871":"中租-KY",
"2301":"光寶科","6669":"緯穎","3661":"世芯-KY","2408":"南亞科","2344":"華邦電","1101":"台泥","2105":"正新","9910":"豐泰"}
NAMES["0050"]="元大台灣50（核心）"
nm = lambda t: f"{t.replace('.TW','')} {NAMES.get(t.replace('.TW',''), '')}"
pct = lambda v: "—" if pd.isna(v) else f"{v:+.1%}"


def svg_curve(a, b, w=680, h=240, padl=34, padb=20):
    """對數座標淨值曲線，標註年份與樣本外分界"""
    la, lb = np.log(a), np.log(b)
    lo, hi = min(la.min(), lb.min()), max(la.max(), lb.max())
    n = len(a); X = lambda i: padl + i / (n - 1) * (w - padl - 4)
    Y = lambda v: (h - padb) - (v - lo) / (hi - lo) * (h - padb - 8)
    path = lambda s: "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(s.values))
    g = []
    years = a.index.year
    for yr in sorted(set(years))[1:]:
        i = int(np.argmax(years == yr)); x = X(i)
        g.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="4" y2="{h-padb}" class=grid />')
        if yr % 2 == 0:
            g.append(f'<text x="{x:.1f}" y="{h-5}" class=ax text-anchor=middle>{str(yr)[2:]}</text>')
    for mult in [1, 2, 5, 10, 20, 50]:
        v = np.log(mult)
        if lo <= v <= hi:
            g.append(f'<text x="{padl-4}" y="{Y(v)+4:.1f}" class=ax text-anchor=end>{mult}x</text>')
    vi = int(np.argmax(a.index >= pd.Timestamp(tm.VALID_START)))
    xv = X(vi)
    g.append(f'<rect x="{xv:.1f}" y="4" width="{w-4-xv:.1f}" height="{h-padb-4}" class=oos />'
             f'<text x="{xv+4:.1f}" y="16" class=ax>樣本外</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart">{"".join(g)}'
            f'<path d="{path(lb)}" class="l-bench"/><path d="{path(la)}" class="l-strat"/></svg>')


def yearly(d, bd):
    ys = (1 + d).groupby(d.index.year).prod() - 1
    yb = (1 + bd).groupby(bd.index.year).prod() - 1
    rows = ""
    for y in ys.index:
        diff = ys[y] - yb.get(y, np.nan)
        cls = "pos" if diff > 0 else "neg"
        tag = " <span class=oos-tag>樣本外</span>" if y >= int(tm.VALID_START[:4]) else ""
        rows += (f"<tr><td>{y}{tag}</td><td class=num>{ys[y]:+.1%}</td><td class=num>{yb.get(y, np.nan):+.1%}</td>"
                 f"<td class='num {cls}'>{diff:+.1%}</td></tr>")
    return rows


def main():
    offline = "--offline" in sys.argv
    os.makedirs("docs", exist_ok=True)
    p = json.load(open("params.json")) if os.path.exists("params.json") else dict(lookback=120, top_k=8, core_w=0.5)
    px = tm.load_prices(offline)
    w = tm.target_weights(px, **p)
    stocks = px.drop(columns=[tm.BENCH])
    last = px.index[-1]

    # 換股日：與回測相同的 21 日節奏
    start = max(p["lookback"], tm.MA_FILTER)
    n = len(px) - 1
    last_reb = start + ((n - start) // tm.REBAL) * tm.REBAL
    days_to_next = last_reb + tm.REBAL - n
    is_reb = last_reb == n

    hold = w.iloc[-1][w.iloc[-1] > 0]
    prev = w.iloc[last_reb - 1] if last_reb > 0 else w.iloc[-1] * 0
    prev = prev[prev > 0]
    added, removed = sorted(set(hold.index) - set(prev.index)), sorted(set(prev.index) - set(hold.index))

    bench = px[tm.BENCH]; ma = bench.rolling(tm.MA_FILTER, min_periods=150).mean()
    risk_on = bench.iloc[-1] > ma.iloc[-1]
    momdf, scdf, rdf = tm.scores(px, p["lookback"])
    mom, sc, r = momdf.iloc[-1], scdf.iloc[-1], rdf.iloc[-1]
    d1 = px.pct_change().iloc[-1]
    rank = pd.DataFrame({"mom": mom, "sc": sc, "rsi": r}).dropna().sort_values("sc", ascending=False)
    rank["ok"] = (rank.rsi < tm.RSI_MAX) & (rank.mom > 0)

    # 全期（訓練 + 驗證）與驗證期績效
    bret = bench.pct_change().fillna(0.0)
    dall, _, _ = tm.backtest(px, w, tm.TRAIN_START)
    d, pos, ret = tm.backtest(px, w, tm.VALID_START)
    dtr, _, _ = tm.backtest(px, w, tm.TRAIN_START, tm.TRAIN_END)
    m, bm = tm.metrics(d), tm.metrics(bret.loc[tm.VALID_START:])
    mt, bmt = tm.metrics(dtr), tm.metrics(bret.loc[tm.TRAIN_START:tm.TRAIN_END])
    ball = bret.loc[dall.index]
    eq, beq = (1 + dall).cumprod(), (1 + ball).cumprod()
    yr_rows = yearly(dall, ball)

    # 逐筆交易（實際收盤價）
    raw = tm.load_raw()
    raw = px if raw is None else raw.reindex(px.index).ffill()
    _, pos_all, _ = tm.backtest(px, w, tm.TRAIN_START)
    tl = tm.trade_log(pos_all, px, raw)
    tl.to_csv("docs/trades.csv", index=False, encoding="utf-8-sig") if len(tl) else None
    opn = tl[tl["持有中"]].set_index("股票") if len(tl) else pd.DataFrame()
    pend_sell = [t for t in opn.index if t not in hold.index]
    f2 = lambda v: "—" if v is None or pd.isna(v) else f"{v:,.2f}"

    def hold_row(t, wt):
        if t in opn.index:
            o = opn.loc[t]
            info = (f"<td>{o['進場日']}</td><td class=num>{f2(o['進場價'])}</td><td class=num>{f2(o['現價'])}</td>"
                    f"<td class='num {'pos' if o['報酬'] > 0 else 'neg'}'>{o['報酬']:+.1%}</td>")
        else:
            info = f"<td colspan=2 class=hint>明日收盤買進</td><td class=num>{f2(raw[t].iloc[-1])}</td><td class=num>—</td>"
        return (f"<tr><td>{html.escape(nm(t))}{' <span class=new>新</span>' if t not in opn.index else ''}</td>"
                f"<td class=num>{wt:.0%}</td>{info}</tr>")
    hold_rows = "".join(hold_row(t, wt) for t, wt in hold.sort_values(ascending=False).items())
    hold_rows += "".join(
        f"<tr class=sell><td>{html.escape(nm(t))} <span class=out>賣</span></td><td class=num>0%</td>"
        f"<td>{opn.loc[t, '進場日']}</td><td class=num>{f2(opn.loc[t, '進場價'])}</td><td class=num>{f2(opn.loc[t, '現價'])}</td>"
        f"<td class=num>{opn.loc[t, '報酬']:+.1%}</td></tr>" for t in pend_sell)
    if not len(hold) and not pend_sell:
        hold_rows = "<tr><td colspan=6 class=empty>目前空手（大盤濾網或無合格標的）</td></tr>"
    closed = tl[~tl["持有中"]].sort_values("出場日", ascending=False).head(30) if len(tl) else pd.DataFrame()
    trade_rows = "".join(
        f"<tr><td>{html.escape(nm(row['股票']))}</td><td>{row['進場日']}</td><td class=num>{f2(row['進場價'])}</td>"
        f"<td>{row['出場日']}</td><td class=num>{f2(row['出場價'])}</td>"
        f"<td class='num {'pos' if row['報酬'] > 0 else 'neg'}'>{row['報酬']:+.1%}</td><td class=num>{row['天數']}</td></tr>"
        for _, row in closed.iterrows())
    # 出場條件
    raw_b = raw[tm.BENCH].iloc[-1]
    trig = raw_b * ma.iloc[-1] / bench.iloc[-1]
    gap = bench.iloc[-1] / ma.iloc[-1] - 1
    buf_n = tm.BUFFER * p["top_k"]
    rank_rows = "".join(
        f"<tr class='{'' if row.ok else 'dim'}'><td>{i}</td><td>{html.escape(nm(t))}</td>"
        f"<td class=num>{pct(row.mom)}</td><td class=num>{row.sc:.2f}</td><td class=num>{row.rsi:.0f}</td><td>{'✓' if row.ok else '×'}</td></tr>"
        for i, (t, row) in enumerate(rank.head(20).iterrows(), 1))

    tw = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    action = ("大盤在 200MA 之下：全部空手，等站回再進場" if not risk_on else
              "今天是換股日：明天收盤依下表調整" if is_reb else f"非換股日，續抱。下次換股還有 {days_to_next} 個交易日")
    chg = ""
    if is_reb and (added or removed):
        chg = (f"<p class=chg>買進：{'、'.join(map(nm, added)) or '無'}<br>賣出：{'、'.join(map(nm, removed)) or '無'}</p>")

    out = f"""<!doctype html><html lang=zh-Hant><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>台股動能選股</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#77756f;--line:#e4e2dc;--up:#c0392b;--dn:#1e8449;--acc:#2f5d8a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--fg:#ecebe6;--mut:#9a988f;--line:#33322e;--acc:#7fa9d6}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"PingFang TC","Noto Sans TC",sans-serif}}
main{{max-width:760px;margin:0 auto;padding:20px 16px 48px}}h1{{font-size:22px;margin:0}}
.sub{{color:var(--mut);font-size:13px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-top:14px}}
h2{{font-size:15px;margin:0 0 10px}}.status{{display:flex;gap:10px;flex-wrap:wrap}}
.pill{{padding:4px 10px;border-radius:99px;font-size:13px;font-weight:600}}.on{{background:#e8f3ec;color:var(--dn)}}.off{{background:#fbeaea;color:var(--up)}}
@media (prefers-color-scheme:dark){{.on{{background:#1d3326}}.off{{background:#3a2020}}}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:7px 6px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--mut);font-weight:500;font-size:12px}}.num{{text-align:right;font-variant-numeric:tabular-nums}}
.dim td{{color:var(--mut)}}.new{{font-size:11px;background:var(--acc);color:#fff;border-radius:4px;padding:1px 5px}}
.empty{{color:var(--mut);text-align:center}}.chg{{margin:10px 0 0;font-size:14px}}
.kpi{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.kpi div{{font-size:12px;color:var(--mut)}}.kpi b{{display:block;font-size:17px;color:var(--fg)}}
.chart{{width:100%;height:auto;margin-top:10px}}.l-strat{{fill:none;stroke:var(--acc);stroke-width:2}}.l-bench{{fill:none;stroke:var(--mut);stroke-width:1.5;stroke-dasharray:4 3}}
.grid{{stroke:var(--line);stroke-width:1}}.ax{{font-size:10px;fill:var(--mut)}}.oos{{fill:var(--acc);opacity:.07}}
.pos{{color:var(--up)}}.neg{{color:var(--dn)}}.oos-tag{{font-size:10px;color:var(--acc);border:1px solid var(--acc);border-radius:4px;padding:0 4px}}
.exit{{margin-top:12px;font-size:13px;line-height:1.7;background:var(--bg);border-radius:8px;padding:10px 12px}}.hint{{color:var(--mut);font-size:12px}}
.out{{font-size:11px;background:var(--dn);color:#fff;border-radius:4px;padding:1px 5px}}.sell td{{color:var(--mut)}}a{{color:var(--acc)}}
.lg{{font-size:12px;color:var(--mut)}}.wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>台股動能選股</h1>
<div class=sub>資料日 {last.date()}｜更新於 {tw}（台北）｜參數 lookback={p['lookback']} · 衛星 {p['top_k']} 檔 · 0050 核心 {p['core_w']:.0%}</div>
<section class=card><div class=status>
<span class="pill {'on' if risk_on else 'off'}">大盤濾網：{'多頭 可持股' if risk_on else '空頭 空手'}</span>
<span class=pill style="background:var(--bg)">0050 {raw_b:.2f}｜200MA 約 {trig:.2f}（{gap:+.1%}）</span></div>
<p style="margin:10px 0 0">{action}</p>{chg}</section>
<section class=card><h2>目前持股</h2><div class=wrap><table>
<tr><th>股票</th><th class=num>權重</th><th>進場日</th><th class=num>進場價</th><th class=num>現價</th><th class=num>損益</th></tr>{hold_rows}</table></div>
<div class=exit><b>出場條件</b>（成交都在訊號隔天收盤）<br>
① 0050 收盤跌破 200MA（約 {trig:.2f}，目前距離 {gap:+.1%}）→ 全部賣出<br>
② 換股日時跌出前 {buf_n} 名、動能轉負或 RSI≥{tm.RSI_MAX} → 賣出該檔<br>
<span class=hint>策略沒有個股停損價，進出場價為實際收盤價（未還原），損益已含手續費與證交稅、以還原價計算。</span></div></section>
<section class=card><h2>風險調整動能排行 Top 20</h2><div class=wrap><table>
<tr><th>#</th><th>股票</th><th class=num>動能</th><th class=num>風險調整</th><th class=num>RSI</th><th>合格</th></tr>{rank_rows}</table></div></section>
<section class=card><h2>回測績效 {dall.index[0].year}–{last.year}</h2>
<div class=wrap><table><tr><th></th><th class=num>年化</th><th class=num>最大回撤</th><th class=num>夏普</th></tr>
<tr><td>策略・訓練期 {tm.TRAIN_START[:4]}–{tm.TRAIN_END[:4]}</td><td class=num>{mt['年化報酬']:.1%}</td><td class=num>{mt['最大回撤']:.1%}</td><td class=num>{mt['夏普值']:.2f}</td></tr>
<tr class=dim><td>0050・訓練期</td><td class=num>{bmt['年化報酬']:.1%}</td><td class=num>{bmt['最大回撤']:.1%}</td><td class=num>{bmt['夏普值']:.2f}</td></tr>
<tr><td>策略・驗證期 {tm.VALID_START[:4]}–</td><td class=num>{m['年化報酬']:.1%}</td><td class=num>{m['最大回撤']:.1%}</td><td class=num>{m['夏普值']:.2f}</td></tr>
<tr class=dim><td>0050・驗證期</td><td class=num>{bm['年化報酬']:.1%}</td><td class=num>{bm['最大回撤']:.1%}</td><td class=num>{bm['夏普值']:.2f}</td></tr></table></div>
{svg_curve(eq, beq)}<div class=lg>— 策略　- - 0050 買進持有（對數座標，x 軸為西元年後兩位）</div></section>
<section class=card><h2>近 30 筆已平倉交易</h2><div class=wrap><table>
<tr><th>股票</th><th>進場日</th><th class=num>進場價</th><th>出場日</th><th class=num>出場價</th><th class=num>報酬</th><th class=num>天數</th></tr>{trade_rows}</table></div>
<p class=hint style="margin:8px 0 0"><a href="trades.csv" download>下載全部交易紀錄 CSV</a></p></section>
<section class=card><h2>逐年報酬</h2><div class=wrap><table>
<tr><th>年</th><th class=num>策略</th><th class=num>0050</th><th class=num>差額</th></tr>{yr_rows}</table></div></section>
<p class=sub>僅供研究，非投資建議。股票池為現今權值股，回測含倖存者偏差。</p>
</main></body></html>"""
    os.makedirs("docs", exist_ok=True)
    open("docs/index.html", "w", encoding="utf-8").write(out)
    print("written docs/index.html", last.date())


if __name__ == "__main__":
    main()
