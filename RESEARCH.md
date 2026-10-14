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

*(Read this table with §8: the per-second replay behind it misses about a
third of what the live bot does and understates both the loss rate and the
drawdown.)*

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

## 8. The market moves first (13 Oct 2026): a trade-by-trade replay and two more checks

**What happened.** From the v3 change (11 Oct 17:03 UTC) to the evening of
13 Oct the bot made 27 trades: 24 won, 3 lost, **−$5.74** (hourly strikes: 13
trades, 12 won, +$4.78; 15-minute markets: 14 trades, 12 won, −$10.52).

| Time (UTC, 13 Oct) | Market | Bought | Seconds left | Model | Result | The tape around the order |
|---|---|---|---|---|---|---|
| 07:59:17 | hourly | YES 14 @ 28¢ | 42 | 97.1% | −$4.11 | 5,352 contracts sold from 98¢ down to 20¢ in the quarter second before; the next index print, 0.4 s after the order, was $22 lower and under the line |
| 17:44:28 | 15-minute | YES 14 @ 81¢ | 32 | 95.7% | −$11.49 | 27,703 sold from 98¢ down to 53¢ in the quarter second before; the index print 0.9 s after the order was $23 lower |
| 20:14:27 | 15-minute | YES 12 @ 93¢ | 32 | 98.4% | −$11.26 | 11,285 sold from 99¢ down to 91¢ in the half second before; the next print, 0.1 s after the order, was $7 lower, at the line; YES traded at 98¢ again a second later, then Bitcoin fell another $29 in four seconds |

In the first two the bot bought a favourite whose price had already
collapsed. Kalshi's traders had reacted to a jump in Bitcoin before the index
value the bot holds showed it; the model still called the old favourite, and
28¢ for something it valued at 97¢ looked like a bargain. This is the "market
leads the index" effect of §7, met from the other side. The third is a
milder case, discussed below.

**Why the backtest had not shown this.** The per-second backtest (§4–§7)
looks at each market once a second and fills at the next trade at least 0.25 s
later. The live bot looks ten times a second and its orders arrive within
about 12 ms, so inside a second it buys at prices the once-a-second view never
sees — including the prices of a collapse in progress.

**The trade-by-trade replay** (`kalshi_bot/research/tape_replay.py`) runs
`strategy/lockin.py`, the class the engine runs, at every public trade in
which another taker bought the favourite (a real trade at a price proves an
offer was there). It hands the strategy the index print the bot would have
held at that instant (a print is usable 50 ms after its timestamp) and a
`TradeTape` with every earlier trade. A fill is booked at that trade's price,
sized by the live limits (whole contracts, a share of the account per market
and per closing moment) and never for more than a quarter of what traded at or
under the bot's limit in the following second.

Checked against live trading: of the bot's 27 real trades since v3 the replay
(checks off, as the live bot was) takes 26, all on the same side, including
the three losses. It also takes 4 small trades the live bot did not. P&L on
the 26 shared markets: live −$6.17, replay −$4.90.

**What it says about risk.** All recorded history (15-minute markets from
24 Jul, hourly strikes from 24 Aug, to 13 Oct 20:15 UTC; 19,609 markets), a
$100 account that never grows, at most 10% per market and 15% per closing
moment:

| v3 rule | Trades a day | Lost | Profit | Worst dip | Worst day |
|---|---|---|---|---|---|
| Per-second backtest (§7; 15-minute only, 79 days) | 12.4 | 2.8% | +$476 | $14.40 | — |
| Trade by trade, 15-minute markets (82 days) | 18.1 | 4.3% | +$481 | — | — |
| Trade by trade, hourly strikes (51 days) | 8.7 | 3.6% | +$148 | — | — |
| Trade by trade, both | 1,920 trades | 4.1% | +$629 | $34.10 | −$25.70 |

The earlier figures understated the risk. About a third of what the live bot
does on 15-minute markets is invisible to the per-second backtest, and those
extra trades roughly break even while carrying the same full-stake losses.
For v3 the honest numbers are about 4% losing trades and a worst dip near
$34 on a $100 account, not 2.8% and $14.

**Two checks (v4, live since 13 Oct 21:45 UTC).** Before buying, the strategy
asks the trade tape two questions about the outcome it wants:

1. *Price fall* — did it trade, or was it bid, more than 10¢ above the price
   now offered within the last second? (`fall_guard`, `fall_window_s`)
2. *Rush of selling* — were more contracts of it sold by takers in the last
   half second than this market normally trades in a whole second (both
   sides, average of the ten seconds that ended a second ago, never less than
   50 contracts)? (`sell_guard_x`, `sell_window_s`, `sell_base_s`,
   `sell_base_floor`)

If either answer is yes it does not buy, and it stays out for one more second
after the last warning (`guard_hold_s`) so that the next index print can
arrive. Then it decides afresh. The engine feeds the tape from Kalshi's
WebSocket `trade` channel and also remembers the best bid it saw on each pass,
so the price-fall check does not depend on a trade message arriving in time.
Without a confirmed trade subscription the engine does not buy at all, and the
first wait in each market is written to the log (`WAIT <ticker>: <reason>`).

| Rule | Trades | Lost | Profit | Worst dip | Worst day | Losing days |
|---|---|---|---|---|---|---|
| No checks (v3) | 1,920 | 79 (4.1%) | +$629 | $34.10 | −$25.70 | 17 of 82 |
| Price-fall check only | 1,900 | 59 (3.1%) | +$692 | $23.70 | −$14.60 | 13 |
| Selling check only | 1,726 | 50 (2.9%) | +$776 | $23.60 | −$14.60 | 13 |
| **Both, stay out 1 s (shipped)** | 1,720 | 48 (2.8%) | +$733 | $23.60 | −$14.60 | 12 |

The shipped rule keeps 90% of the trades (15-minute 16.8 a day, 3.1% lost;
hourly 6.9 a day, 1.7% lost) and beats "no checks" in each part of the history
taken alone: 24 Aug–18 Sep +$399 against +$371, 19 Sep–13 Oct +$215 against
+$147, and the 15-minute-only weeks before 24 Aug +$119 against +$111. Over
the last four weeks: 2.8% lost and +$230, against 4.4% and +$170.

Where the +$104 comes from:

* 1,447 markets are bought at the same moment either way.
* 202 are skipped altogether. All but one were winners: −$102.
* 241 are bought later on the same side (3.3 s at the median), at 94¢ on
  average instead of 91¢: −$106. Eight of them lose either way.
* In 30 the bot buys later (7.3 s at the median) and on the *other* side.
  Without the checks all 30 lost, $243 together; with them all 30 win, $60
  together: **+$303**.

So the checks cost a little in some 440 markets and pay for it in the 30
where the old rule bought into a collapse: 31 of the 79 losing trades go.

Sensitivity, everything else as shipped (shipped setting in bold):

| Stay out after a warning | 0 s | 0.5 s | **1 s** | 1.5 s | 2 s | 3 s |
|---|---|---|---|---|---|---|
| Trades | 1,778 | 1,738 | **1,720** | 1,715 | 1,706 | 1,682 |
| Lost | 60 | 50 | **48** | 48 | 48 | 48 |
| Profit | +$675 | +$739 | **+$733** | +$723 | +$712 | +$691 |
| Worst dip | $33.70 | $23.70 | **$23.60** | $23.70 | $23.70 | $23.70 |

Without the stay-out each check clears before the index has caught up and the
bot buys the bounce: half the benefit is lost. The first implementation,
checked only on the two losing moments, had no stay-out; the full replay
caught it before it shipped.

| Selling threshold (× a normal second) | 2 | 1.5 | **1** | 0.75 | 0.5 | 0.35 |
|---|---|---|---|---|---|---|
| Trades | 1,796 | 1,769 | **1,720** | 1,681 | 1,621 | 1,515 |
| Lost | 57 | 54 | **48** | 44 | 41 | 38 |
| Profit | +$673 | +$693 | **+$733** | +$751 | +$730 | +$691 |
| Worst dip | $25.40 | $24.80 | **$23.60** | $23.60 | $23.90 | $18.60 |

| Price-fall threshold | 5¢ | **10¢** | 15¢ | 20¢ |
|---|---|---|---|---|
| Trades | 1,683 | **1,720** | 1,723 | 1,723 |
| Lost | 46 | **48** | 48 | 49 |
| Profit | +$694 | **+$733** | +$755 | +$754 |
| Worst dip | $19.20 | **$23.60** | $23.60 | $23.60 |

Differences of $20–40 between neighbouring settings are a handful of trades
and mean nothing; the tables are there to show that the result does not hinge
on the exact numbers.

**The third loss: a milder case that the shipped thresholds miss.** At that
order the highest YES trade of the last second was 99.4¢ against 93.4¢ paid —
a 6¢ fall, under the 10¢ threshold — and 11,285 contracts had been sold in the
last half second against a "normal second" of 13,509, inflated by a
53,000-contract burst four seconds earlier. Neither check fires: the replay of
the shipped rule buys the same market at 94.9¢ and loses $9.52. A selling
threshold of 0.75× or lower keeps the bot out through the next seconds of
heavy selling (the checks fire on 424 of the replay's looks at this market)
until the model has turned, and then buys NO at 79¢. 0.75× is no worse than 1×
in any column above, but the gap is four losing trades in 1,700 and one of
them is the trade that prompted the comparison, so it was not changed on that
evidence alone. Measuring the "normal second" by the median of the ten
one-second totals, which a single burst cannot inflate, was also tried: at the
same number of trades it was no better than the plain average (1,676 trades,
45 lost, +$721 at 1×). It is also a weaker case than the first two: a second
after the bot paid 93¢ the market itself, with the new index print in hand,
was paying 98¢ for YES. What lost the trade was Bitcoin falling another $29
over the following eight seconds, which is the strategy's ordinary risk (§6).

**First hours live** (13 Oct 21:45 UTC to 14 Oct 02:30 UTC). Five markets
traded, none lost, +$6.05; the price-fall check fired once and the selling
check twice (`WAIT` lines in the log). 22:59:30 UTC, 30 s before the 23:00
close: the index print fell $20 in
one second to $11 under the line and the model turned to NO at 95%. By the
time that print reached the bot the market had already been there and was on
its way back: NO had traded at 81¢ and was offered at about 60¢. The
price-fall check held the bot out; a second later the next print ($6 under)
left the model unsure, and at 15 s left, with Bitcoin $9 above the line again,
the bot bought YES at 73¢ and won $2.82. The replay of that market buys NO at
63¢ and loses ($9.62 without the checks, $7.11 with them): it acts at the
first print after the index value, a few tenths of a second before the live
bot, when NO had only traded at 71¢. Live can therefore catch cases the replay
does not, and the reverse will happen too. The two selling-rush firings
(01:59:43 and 02:29:22 UTC) came seconds after the bot had bought, when it
already held nearly its full position in that market, and changed nothing.

**Tested and not adopted** (same data, 13 Oct): switching the hourly ladder
off (about a fifth of the profit gone, dips no smaller; the 13 live
hourly trades since v3 are +$4.78); z ≥ 2 everywhere; a 2¢ minimum edge;
shorter windows; a 97–98¢ price cap (fewer trades for no profit); skipping
offers far below the model's value; requiring the last index move to favour
the trade; giving the hourly ladder the 15-minute 55 s / early-lead rules.
The losses that remain show no warning on the tape beforehand.

**Caveats.** The replay knows trades, not the order book: it assumes an offer
wherever another taker bought, and that the bot would have had a quarter of
what traded at or under its limit in the next second. It cannot know how other
traders would have reacted to the bot's own orders. In paper mode the checks
read the public tape by polling, so they are coarser than live. Eighty-two
days is one sample, and the last two ran worse than the average (3 losses in
27 live trades).

**Reproduce.**

```
python -m kalshi_bot.research.history --days 3 --tail-seconds 130
python -m kalshi_bot.research.history --series KXBTCD --days 3 --min-volume 20000 --tail-seconds 130
python -m kalshi_bot.research.tape_replay --per-market 0.10 --per-close 0.15
```

Tests: `tests/test_tape.py` (the tape against brute-force definitions),
`tests/test_guards.py` (both checks, the stay-out, the engine wiring, the log
line, and the two real losing buys replayed from `tests/data/real_losses.json`),
`tests/test_tape_replay.py` (what the replay books on a made-up market).
