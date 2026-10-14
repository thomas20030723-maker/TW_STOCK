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


def svg_curve(a, b, w=640, h=200):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    def path(s):
        xs = np.linspace(0, w, len(s)); ys = h - (s.values - lo) / (hi - lo) * (h - 10) - 5
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" class="chart">'
            f'<path d="{path(b)}" class="l-bench"/><path d="{path(a)}" class="l-strat"/></svg>')


def main():
    offline = "--offline" in sys.argv
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

    bench = px[tm.BENCH]; ma = bench.rolling(tm.MA_FILTER).mean()
    risk_on = bench.iloc[-1] > ma.iloc[-1]
    momdf, scdf, rdf = tm.scores(px, p["lookback"])
    mom, sc, r = momdf.iloc[-1], scdf.iloc[-1], rdf.iloc[-1]
    d1 = px.pct_change().iloc[-1]
    rank = pd.DataFrame({"mom": mom, "sc": sc, "rsi": r}).dropna().sort_values("sc", ascending=False)
    rank["ok"] = (rank.rsi < tm.RSI_MAX) & (rank.mom > 0)

    # 驗證期績效
    d, pos, ret = tm.backtest(px, w, tm.VALID_START)
    m = tm.metrics(d, pos, ret)
    bm = tm.metrics(bench.pct_change().loc[tm.VALID_START:])
    eq, beq = (1 + d).cumprod(), (1 + bench.pct_change().loc[tm.VALID_START:].fillna(0)).cumprod()

    hold_rows = "".join(
        f"<tr><td>{html.escape(nm(t))}{' <span class=new>新</span>' if t in added else ''}</td>"
        f"<td class=num>{wt:.0%}</td><td class=num>{pct(mom.get(t))}</td><td class=num>{'—' if t == tm.BENCH else f'{r[t]:.0f}'}</td>"
        f"<td class=num>{pct(d1[t])}</td></tr>" for t, wt in hold.sort_values(ascending=False).items())
    if not len(hold):
        hold_rows = "<tr><td colspan=5 class=empty>目前空手（大盤濾網或無合格標的）</td></tr>"
    rank_rows = "".join(
        f"<tr class='{'' if row.ok else 'dim'}'><td>{i}</td><td>{html.escape(nm(t))}</td>"
        f"<td class=num>{pct(row.mom)}</td><td class=num>{row.sc:.2f}</td><td class=num>{row.rsi:.0f}</td><td>{'✓' if row.ok else '×'}</td></tr>"
        for i, (t, row) in enumerate(rank.head(20).iterrows(), 1))

    tw = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    action = ("今天是換股日：明天收盤依下表調整" if is_reb else f"非換股日，續抱。下次換股還有 {days_to_next} 個交易日")
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
.chart{{width:100%;height:180px;margin-top:10px}}.l-strat{{fill:none;stroke:var(--acc);stroke-width:2}}.l-bench{{fill:none;stroke:var(--mut);stroke-width:1.5;stroke-dasharray:4 3}}
.lg{{font-size:12px;color:var(--mut)}}.wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>台股動能選股</h1>
<div class=sub>資料日 {last.date()}｜更新於 {tw}（台北）｜參數 lookback={p['lookback']} · 衛星 {p['top_k']} 檔 · 0050 核心 {p['core_w']:.0%}</div>
<section class=card><div class=status>
<span class="pill {'on' if risk_on else 'off'}">大盤濾網：{'多頭 可持股' if risk_on else '空頭 空手'}</span>
<span class=pill style="background:var(--bg)">0050 {bench.iloc[-1]:.2f} / 200MA {ma.iloc[-1]:.2f}</span></div>
<p style="margin:10px 0 0">{action}</p>{chg}</section>
<section class=card><h2>目前持股</h2><div class=wrap><table>
<tr><th>股票</th><th class=num>權重</th><th class=num>動能</th><th class=num>RSI</th><th class=num>今日</th></tr>{hold_rows}</table></div></section>
<section class=card><h2>風險調整動能排行 Top 20</h2><div class=wrap><table>
<tr><th>#</th><th>股票</th><th class=num>動能</th><th class=num>風險調整</th><th class=num>RSI</th><th>合格</th></tr>{rank_rows}</table></div></section>
<section class=card><h2>驗證期績效（{tm.VALID_START[:4]} 起，樣本外）</h2>
<div class=kpi><div>年化報酬<b>{m['年化報酬']:.1%}</b>0050 {bm['年化報酬']:.1%}</div>
<div>最大回撤<b>{m['最大回撤']:.1%}</b>0050 {bm['最大回撤']:.1%}</div>
<div>夏普值<b>{m['夏普值']:.2f}</b>0050 {bm['夏普值']:.2f}</div></div>
{svg_curve(eq, beq)}<div class=lg>— 策略　- - 0050 買進持有</div></section>
<p class=sub>僅供研究，非投資建議。股票池為現今權值股，回測含倖存者偏差。</p>
</main></body></html>"""
    os.makedirs("docs", exist_ok=True)
    open("docs/index.html", "w", encoding="utf-8").write(out)
    print("written docs/index.html", last.date())


if __name__ == "__main__":
    main()
