# Research notes — Kalshi 15-minute & hourly Bitcoin markets

Everything below was measured on Kalshi's public data: every trade
(microsecond timestamps) and Kalshi's published 1-second settlement-index
series, with realistic fills and fees (see *Method*). Code: `kalshi_bot/research/`.

## 1. What the contract really is (verified, not assumed)

| Fact | Evidence |
|---|---|
| YES ⇔ mean of the 60 BRTI prints stamped **T−60 … T−1** ≥ strike | ~7,400 official settlements, every week Jul–Oct 2026: exact to the cent in 98.6–99.7%, one cent off otherwise; re-confirmed on live-recorded prints. Kalshi's docs describe a window one second later. |
| Strike = previous contract's settlement average | every contract; the bot also computes it locally and checks it against the published value |
| Hourly `KXBTCD` strikes settle on the **same** average at the top of the hour | 19:00 ET example: KXBTCD settlement 84,194.99 = KXBTC15M 19:00 settlement |
| Taker fee `0.07·p·(1−p)` (rounded up to $0.0001); no maker fee on KXBTC15M | real demo fills: 102.5 contracts at 81¢ → fee $1.1043 |
| Crypto trades on exchange shard 2 | collateral must be moved there first (the bot does it) |
| Liquidity | ~2–3 M contracts, 30–40k trades per 15-minute market, and rising: contracts traded per second in the final minute went from ~1,700 (late July) to ~7,100 (early October) |
| Other coins use the same rule on their own index | ETH `ETHUSD_RTI` (2 decimals), SOL `SOLUSD_RTI` (4), XRP `XRPUSD_RTI` (4), DOGE `DOGEUSD_RTI` (7): each settles on the 60-second average, rounded to that precision |
| The **demo** exchange is not a simulator of production | demo settled `KXBTCD-26SEP2813-T83599.99` NO; production settled it YES (83,682.46 ≥ 83,600). Demo books are filled by test bots at arbitrary prices. Demo P&L is meaningless. |
| This PC's clock drifts (0.3 s ahead in September, 1.6 s in October) | measured from feed timestamps; the bot times everything on the exchange clock |

## 2. Hypotheses tested

| Idea | Result | Verdict |
|---|---|---|
| Original v1 rule (5-min candles + 20-EMA at minute 10, hold) | 2,439 markets: 80.5% win rate at an average price of 80.8¢ → **−1.23¢/contract** after fees | loses |
| Model-vs-market trading mid-window | Kalshi's price beats a Brownian index model on log-loss at every minute 1–14 | no edge |
| Favourite–longshot bias (buy 85–95¢ favourites) | ≈ +1¢/contract, not significant, sign flips by month | too weak |
| Resting (maker) bids, zero maker fee | adversely selected; walk-forward t ≈ 1.1 with losing folds | rejected |
| Lock-in **before** the averaging window | 61–90 s out loses even when the model shows z ≥ 3 (−6.6¢/contract, 133 markets); 61–120 s: −5.5¢ | rejected |
| Relaxing the confidence gates (z < 1.5 or model < 90%) | more markets but drawdown doubles and the t-stat falls from 5.9 to 2.4–4.4 | rejected |
| Lock-in on **ETH** 15-minute markets | 16.6 days, 1,568 markets: first version −$21 (82 markets), current default +$50 (96 markets), two losses over $150 — indistinguishable from zero | **not traded** |
| Lock-in on **XRP** 15-minute markets | 12.7 days, 1,201 markets: first version +$203 (46 markets) but current default only +$51 (57 markets, one −$103 loss) — the answer flips with the rule, so it is not established | **not traded** |
| **Lock-in inside the averaging window (last 60 s), BTC 15-min** | §4 | **trade** |
| **Same, hourly BTC strike ladder (KXBTCD)** | §4 | **trade** |

## 3. Why the edge exists — and where the model is weakest

Settlement is an **average** of the last minute. Once k of the 60 prints are
known only the remaining m = 60 − k matter, and the variance of their mean is
`σ²·m(m+1)(2m+1)/(6·60²)` — far smaller than the spot-to-close variance most
traders price with. Strong favourites are therefore sometimes offered below
value in the final seconds.

Two places where the model is over-confident on the trades it takes (not on
all seconds — it is well calibrated overall):

* **Early in the window (45–60 s left).** Favourites the model put at 96.3%
  won 93.2%; at 99.6% they won 98.0%. Trades there earned 1.5¢/contract
  against 3¢+ later. The shipped rule therefore inflates volatility linearly
  from ×1.0 at 30 s left to ×1.25 at 60 s left ("early-window caution").
* **When the market disagrees a lot.** Where the offer was 10¢+ below the
  model, the favourite won 91.8% against a modelled 97.7% — other traders
  sometimes know something the index has not printed yet. These trades are
  still the most profitable per contract (+12¢), but they are why position
  size is capped per contract rather than scaled up without limit.

## 4. Results

*The tables in this section are as of 10 Oct 2026. The report regenerated on
11 Oct (the one behind the README charts and the dashboard's Backtest tab)
adds one more day, which includes the three losing markets of the first live
day (§7): walk-forward +$6,257 over 1,312 markets (11 of 11 weeks), hourly
strikes +$1,641 over 240 markets, the first version on unseen data +$717 over
124 markets (91.9% won), and a stress-test baseline of +$6,149 for the current
rule.*

### 4a. Unseen data (the test that matters)

The first version was finished on 27 Sep 2026. Everything that closed after
that played no part in building or tuning it. Rules were written down before
this data was examined (the first three days were looked at once, early, while
choosing the candidates — that is the only contamination).

**BTC 15-min, 28 Sep – 10 Oct, 1,185 markets, 12.6 days**

| Rule | Markets | per day | Won | Edge/contract | Max drawdown | P&L |
|---|---|---|---|---|---|---|
| First version exactly as shipped (edge ≥ 2¢) | 110 | 8.7 | 93.6% | +7.07¢ | $115 | **+$806** |
| **Current default** (edge ≥ 1¢, early-window caution) | 139 | 11.0 | 96.4% | +5.51¢ | $107 | **+$847** |
| Edge ≥ 1¢ without the caution | 176 | 13.9 | 95.5% | +3.99¢ | $227 | +$824 |
| Conservative (volatility ×1.2, edge ≥ 2¢) | 65 | 5.1 | 93.8% | +8.12¢ | $60 | +$542 |
| Strict (z ≥ 2, edge ≥ 2¢) | 78 | 6.2 | 96.2% | +6.59¢ | $93 | +$513 |

The first version was profitable on 11 of the 13 calendar days (the other two
−$16 and −$32). Opportunities were somewhat rarer than in the tuning period
(8.7 vs 11.6 markets a day) while the edge per contract held up.

**Hourly BTC strikes, 28 Sep – 10 Oct, 619 markets with real volume, 12.6 days**

| Rule | Markets | per day | Won | Edge/contract | Worst market | P&L |
|---|---|---|---|---|---|---|
| First version exactly as shipped | 44 | 3.5 | 97.7% | +7.96¢ | −$0.93 | **+$275** |
| **Current default** | 45 | 3.6 | 100% | +5.38¢ | +$0.21 | **+$252** |
| Edge ≥ 1¢ without the caution | 64 | 5.1 | 98.4% | +5.60¢ | −$0.93 | +$386 |

Every one of the 11 days with a trade was positive. Forty-five markets is a
small sample: "100% won" will not last, and a loss here is a full stake like
anywhere else.

On BTC 15-min, one day (3 Oct) produced about 40% of the unseen-period profit;
without it the current default made +$493 over the other 12 days.

### 4b. Walk-forward, 31 Jul – 10 Oct

Before each test week the volatility model is re-chosen by log-loss and the
thresholds by risk-penalised P&L, **using only earlier data**. Sizing: 50
contracts per order (up to 3× on large edges), ≤ 300 contracts per market.

| | BTC 15-min (KXBTC15M) | Hourly strikes (KXBTCD) |
|---|---|---|
| Out-of-sample period | 31 Jul – 10 Oct (71.4 days) | 31 Aug – 10 Oct (40 days) |
| Test folds profitable | 11 / 11 | 6 / 6 |
| Calendar weeks profitable | 11 / 11 | 6 / 6 |
| Net P&L | **+$6,312** | **+$1,638** |
| Contracts / markets traded | 180,216 / 1,286 (18 markets a day) | 29,299 / 239 (6 a day) |
| Edge per contract (after fees) | +3.5¢ | +5.6¢ |
| Markets won | 95.1% | 97.1% |
| t-stat across markets | 6.0 | 6.3 |
| Max drawdown | $486 | $139 |
| Worst single market | −$272 | −$139 |

The procedure chose "last 60 s, z ≥ 1.5, edge ≥ 1¢, volatility ×1.0" in every
fold. The shipped default adds the early-window caution on top: on unseen data
it traded fewer markets for the same profit with half the drawdown.

**Stress tests** (whole period, shipped parameters, +$6,547 baseline):
1 s latency +$6,303 · every fill 2¢ worse +$3,546 · 25% participation +$4,403 ·
edge ≥ 2¢ +$5,450 · edge ≥ 3¢ & z ≥ 2 +$3,193 · volatility ×1.2 +$4,655 ·
volatility ×1.5 +$2,873 · starting 90 s out +$4,028 (worse — confirms the
window boundary).

### 4c. What this means for a $100 account

`python -m kalshi_bot.research.account` replays the traded markets for an
account that starts at $100 with the live limits (6% per contract, 10% per
settlement moment, whole contracts). On the 12.6 **unseen** days:

| Markets traded | Trades | per day | Losing | $100 became | Worst drawdown |
|---|---|---|---|---|---|
| BTC 15-min only | 137 | 11 | 5 | $144 | 9% |
| BTC 15-min + hourly strikes | 182 | 15 | 5 | $167 | 9% |

The shape of every trade is the same: a typical win is about **$0.47** and a
loss is about **$4.90** (the whole stake), so roughly ten wins pay for one
loss. The rule stays profitable while fewer than about 1 trade in 11 loses;
on unseen data it was 1 in ~30, on the whole history 1 in ~27.

Read that as "the signal survived contact with new data", not as a promise.
The replay assumes the bot gets a share of the fills other traders got within
two seconds of its decision, and that the mispricing keeps occurring. Live
results can be worse on both counts; that is what the first weeks with a small
account are for, and why the bot stops itself after a 30% drawdown.

## 5. Method

* **Execution:** a decision at second *t* fills only at the price of the next
  public taker trade on that side after *t* + 0.25 s (within 2 s), for ≤ 50% of
  that trade's size. Validated on 726 recorded production order-book snapshots:
  trade-derived prices are unbiased vs the live best ask (mean +0.02¢).
* **Research ↔ live parity:** `tests/test_parity.py` feeds the backtest's own
  trades through the live strategy class (≥ 99% must match) and fails if the
  shipped defaults differ from the parameters reported here.
* **Data quality:** 3 archived markets without a published strike are excluded.
* **Tests:** `python -m pytest -q tests` — pricing vs Monte Carlo, index
  reconstruction, book conventions, fees, tick grids, accounting, account-
  relative limits, cash-aware sizing, drawdown stop, per-coin rounding,
  multi-coin engine, exchange-clock offset and joint risk caps.

## 6. Caveats

Backtests are not guarantees. The edge comes from other traders mispricing
the settlement average; volume in these markets quadrupled in ten weeks, and
as more traders price the average correctly the edge will shrink. The bot's
live fills may be worse than the tape-based simulation. Capacity is limited
by the liquidity offered at mispriced levels. Losses happen (about 4–5% of
traded markets) and each one is the whole stake in that contract — which is
why the limits are a small share of the account and why the bot switches
itself off after a 30% drawdown instead of trading on.

## 7. First live day (10–11 Oct 2026) and the change that followed

The bot went live on 10 Oct at 21:50 UTC with $105 and 10% per trade. First
18 hours: 24 trades, 21 won, 3 lost, **−$10.06** (average win +$0.57, average
loss −$7.36).

**What the three losses had in common**

| Time (UTC) | Bought | Seconds left | Lead over strike | What happened |
|---|---|---|---|---|
| 12:44:02 | NO at 90¢ | 58 | $11 | index +$24 in one second, 3 s later |
| 13:29:00 | YES at 40¢ | 60 | $13 | index −$23 one second later |
| 14:14:00 | NO at 69¢ | 59 | $10 | index +$26 in one second, 5 s later |

* A very quiet market: volatility in the lowest 9% of the 11-week history.
  The model scales confidence to recent volatility, so it rated a $10–13 lead
  95–99% safe, while one-second moves of $17+ still occurred in about one
  minute in ten.
* Entering at the first possible moment: 12 of the 24 trades were entered with
  more than 55 s left and produced all three losses (−$15.90); the other 12
  had none (+$5.84).
* The market leads the index: each time Kalshi prices collapsed about a second
  before the BRTI print showed the move. At 13:29 the price had already fallen
  from 84¢ to 40¢ 0.15 s before the bot's order.

**Live matched the backtest.** Replaying the same hours gives the same three
losing markets (−$8.20 on 13 trades); live took 9 extra small winners (+$3.77)
that the tape-based fill rule does not count. First fill arrived a median
12 ms after the order. No errors, disconnects or rejected orders.

**Rules tested on the 79 days before going live** (constant $100 account, 10%
per trade, whole contracts sized by the limit price, tape-based fills):

| Rule | Trades/day | Lost | Profit | Worst drawdown |
|---|---|---|---|---|
| Final 60 s, no lead rule (previous default) | 13.6 | 3.6% | +$456 | $27.70 |
| Final 55 s | 12.7 | 3.2% | +$472 | $17.30 |
| ≥ 2 bp lead while > 45 s remain | 13.2 | 3.1% | +$478 | $20.20 |
| **Both (shipped for 15-minute markets)** | 12.4 | 2.8% | +$476 | $14.40 |
| Final 50 s | 11.9 | 3.1% | +$434 | $14.40 |
| Lead must survive a 3 bp jump for the time left | 8.4 | 1.1% | +$274 | $16.40 |

The shipped combination keeps profit about the same (the difference is not
statistically meaningful) and halves the drawdown. With the project's research
sizing it trades 9% fewer markets for 7% less profit, a higher t-stat (9.1 vs
8.4) and a 29% smaller drawdown. In the backtest 24% of entries fell in the
first five seconds of the window; they lost 5.3% of the time and earned $0.16
a trade against $0.45–0.52 later in the minute.

The hourly ladder is unchanged: it loses about 1.3% of the time and either
rule would only cost it about 8% of its profit.

**Tested and not adopted:** pausing after a loss (losses barely cluster: 5.5%
loss rate within two hours of a loss vs 3.9% overall); a volatility floor or
sitting out quiet markets; z ≥ 2 everywhere; 2¢ minimum edge; skipping offers
more than 20¢ below the model (costs about 10% of backtest profit — though the
only two live trades of that kind both lost, so it stays on the watch list).

**Also measured:** index moves of 2 bp or more in one second are 1.3–1.4 times
more frequent in the first five seconds after a minute boundary than on
average, and when one side leads by a small margin a move that size goes
against the leader about 60% of the time. The backtest loss rate over the
last four weeks has been about 5%, against 3.6% for the whole history, so the
margin of safety is thinner than the long-run average suggests.
