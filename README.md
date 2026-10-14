<h1 align="center">
  <img src="docs/img/hero.svg" width="100%" alt="Lock-In: a trading bot for Kalshi's 15-minute Bitcoin markets. It waits for the final minute and buys the near-certain side.">
</h1>

<p align="center">
  <a href="#the-idea-in-30-seconds">The idea</a> ·
  <a href="#when-does-it-buy">The rule</a> ·
  <a href="#does-it-work">Does it work?</a> ·
  <a href="#safety-rails">Safety</a> ·
  <a href="#the-dashboard">Dashboard</a> ·
  <a href="#try-it">Try it</a> ·
  <a href="RESEARCH.md">Research notes</a>
</p>

**Lock-In is a trading bot for [Kalshi](https://kalshi.com), an exchange where people trade on yes-or-no questions.**

It trades one kind of question: *"Will Bitcoin be above this price 15 minutes from now?"* It does not try to predict Bitcoin. It waits for the last minute of each market, when the answer is almost settled, and buys the likely side only if that side is still cheaper than it should be.

> [!WARNING]
> This bot can trade real money, and it can lose it. Three days in (13 Oct 2026) its own trades with real money were about $8 down ([day one](#3-the-first-day-with-real-money), [day three](#4-day-three-the-market-moved-first)). Nothing here is financial advice.

---

## The idea in 30 seconds

<img src="docs/img/market.svg" width="100%" alt="A Kalshi market is a yes-or-no question, for example: will Bitcoin be above $83,581.29 at 3:15 PM? YES costs 94 cents and NO costs 7 cents. The side that turns out right pays one dollar.">

Every 15 minutes Kalshi opens a new Bitcoin market. You can buy **YES** or **NO** for somewhere between 1¢ and 99¢. When the market closes, the right side pays **$1** and the wrong side pays nothing.

The interesting part is how Kalshi decides the result:

<img src="docs/img/final-minute.svg" width="100%" alt="A 15-minute market drawn as 15 blocks. The last block, the final minute, is magnified into 60 one-second prices. With 25 seconds left, 35 of the 60 prices are already known and cannot change.">

It does not use the last price. It uses the **average of 60 prices**, one per second, from the final minute. So as that minute ticks by, more and more of the answer is already fixed.

<img src="docs/img/certainty.svg" width="100%" alt="Chart: room left for the result to move as the close approaches. With 30 seconds left Kalshi's average can move 21 percent as far as it could a minute out, against 71 percent for a market that only used the final price. With 10 seconds left: 4 percent against 41 percent.">

That is the whole idea. Late in the final minute one side is often close to certain, but its price has not always caught up. Lock-In reads each of those 60 prices as it arrives, works out the real odds, and buys when the near-certain side is on offer for less than it is worth. It holds for under a minute, then the market pays out.

## When does it buy?

<img src="docs/img/checks.svg" width="100%" alt="Six questions before any order: is it the final 55 seconds, is the model at least 90 percent sure, is the price safely clear of the line, is the contract at least one cent cheaper than it is worth after fees, is the market calm right now, and is the bet inside the risk limits. Only six yes answers lead to a purchase.">

Most markets fail at least one check, and then the bot does nothing. In the replay below it traded about **one market in six**.

<details>
<summary><b>The exact rule, for the curious</b></summary>

<br>

- **Markets:** Kalshi's 15-minute Bitcoin markets (`KXBTC15M`) and the hourly Bitcoin price ladder (`KXBTCD`). Both settle on the average of the 60 one-second CF Benchmarks BRTI prints before the close.
- **Model:** the average is priced exactly. Prints already in are fixed, and only the remaining ones can move it. Volatility is measured live, and fat tails are handled with a Student-t distribution. The maths is checked against simulation in `tests/`.
- **Buy when:** the favourite is at least 90% likely, at least 1.5 standard deviations clear of the strike, and offered at least 1¢ below fair value after Kalshi's fee.
- **15-minute markets only:** act in the final 55 seconds, and while more than 45 seconds remain the expected result must be at least 2 basis points (about $17) from the strike. The hourly markets use the full 60 seconds.
- **Wait when the market moved first:** Kalshi's traders react to a jump in Bitcoin up to a second before the index shows it. If the side the bot wants traded, or was bid, more than 10¢ higher within the last second, or more of it was sold in the last half second than normally trades in a whole second, the bot holds off and looks again a second later.
- **Orders:** immediate-or-cancel at a capped price, then held to settlement.
- **Other coins:** the engine can run ETH, SOL, XRP and DOGE, but only Bitcoin passed the tests, so the others are off by default.

Every number and how it was measured: [RESEARCH.md](RESEARCH.md).

</details>

## Does it work?

**On past markets: yes. With real money: too early to say.** Here is the evidence for both, including the bad days.

### 1. Replayed on past markets

<img src="docs/img/backtest.svg" width="100%" alt="Backtest from 31 July to 11 October 2026: profit of 6,257 dollars at test size over 1,312 markets, 95 percent of them won, 11 of 11 test weeks profitable, 58 days up and 15 days down.">

| | 15-minute markets | Hourly markets |
|---|---|---|
| Period | 31 Jul – 11 Oct 2026 | 31 Aug – 11 Oct 2026 |
| Markets traded | 1,312 | 240 |
| Markets won | 95.0% | 97.1% |
| Profit after fees, at test size | **+$6,257** | **+$1,641** |
| Profit per contract | +3.4¢ | +5.6¢ |
| Weeks in profit | 11 of 11 | 6 of 6 |
| Days in profit | 58 of 73 | 37 of 40 |
| Deepest dip along the way | $486 | $139 |

What keeps the replay honest:

- **Real trades only.** An order counts as filled only if a real trade happened at that price right afterwards, and only for half of that trade's size.
- **Delays and fees are in.** A quarter of a second passes between deciding and buying, and Kalshi's fee is charged on every purchase.
- **No hindsight.** The settings for each week were chosen from earlier weeks only.
- **Tested on unseen data.** The first version was frozen on 27 Sep 2026, then run unchanged on the two weeks that followed: 124 markets traded, 91.9% won, **+$717**.

> [!NOTE]
> "Test size" means up to 300 contracts per market, roughly $280 at a time. A $100 account trades the same signals at about $6 each, so its dollar results are far smaller. A replay is also kinder than real life: it cannot know how other traders would have reacted to the bot's orders.

### 2. The catch: small wins, big losses

<img src="docs/img/payoff.svg" width="100%" alt="A contract bought at 93 cents gains about 6.6 cents if right and loses about 93.4 cents if wrong, so one loss wipes out about 14 wins. Break-even is being right 93.4 percent of the time. The backtest was right 96.8 percent of the time; the first live day 87.5 percent.">

Buying a near-certain thing for 93¢ earns about 6½¢ when it works and loses about 93¢ when it does not. **One loss wipes out about 14 wins.** The bot has to be right more than 93.4% of the time just to break even. In the replay it was right 96.8% of the time. That gap is the entire profit, and it is thin. (The sharper replay in [section 4](#4-day-three-the-market-moved-first) puts it nearer 96% for the old rule and 97% with the newest check.)

### 3. The first day with real money

<img src="docs/img/first-live-day.svg" width="100%" alt="First live day, 10 to 11 October 2026: 24 trades, 21 won and 3 lost, net result minus 10 dollars and 6 cents. All three losses were bought 58 to 60 seconds before the close with a lead of 10 to 13 dollars, and a one-second jump of 23 to 26 dollars flipped the result.">

The bot went live on 10 Oct 2026 with about $100. In its first 18 hours it won 21 of 24 trades and was still **down $10.06**, because three losses cost more than 21 wins earned.

All three losses looked the same: bought at the earliest moment allowed, with a small lead, and then Bitcoin jumped about $25 in a single second. A replay of those same hours shows the same three losing markets, so this was the strategy's known weak spot and not a malfunction.

The rule was tightened the same day: wait five more seconds, and demand a bigger lead early in the minute. Over the 79 days before going live, that change kept the profit and roughly halved the worst dip.

One day proves nothing either way. It is the reason the bets are small.

### 4. Day three: the market moved first

<img src="docs/img/moved-first.svg" width="100%" alt="A real losing trade on 13 October 2026, quarter second by quarter second. On Kalshi the price of YES fell from 98 cents to as low as 53 cents while the bot's Bitcoin price still showed 9 to 12 dollars above the line. The bot bought at 81 cents with its model 96 percent sure. Just under a second later its price feed showed Bitcoin 14 dollars below the line.">

After that change the bot made 27 trades in two days: 24 right, 3 wrong, **$5.74 down**. Two of the three losses had the same cause, and it was not the one from day one.

When Bitcoin jumps, traders on Kalshi react within a fraction of a second. The official price the bot reads shows the jump up to a second later. In that gap the bot still believed in the old favourite, saw its price collapse, and bought what looked like a bargain. The picture above is one of those trades.

It also exposed a blind spot in the replay from section 1. That replay looks at each market once a second. The live bot looks ten times a second, so about a third of what it really does never showed up there. A second replay now runs the live code over every recorded trade. It reproduces 26 of the bot's 27 real trades since the rule change, including all three losses, and its verdict on the old rule is less flattering: **4.1% of trades lost, not 2.8%**, and a worst dip of **$34 on a $100 account, not $14**.

<img src="docs/img/two-checks.svg" width="100%" alt="The live code replayed trade by trade over 82 days on a 100 dollar account. Without the two checks: 1,920 trades, 4.1 percent lost, profit 629 dollars, worst dip 34 dollars, worst day minus 26 dollars. With them: 1,720 trades, 2.8 percent lost, profit 733 dollars, worst dip 24 dollars, worst day minus 15 dollars.">

So the bot got a sixth question: *is the market calm right now?* If the side it wants has just dropped more than 10¢, or sellers have just rushed in, it waits a second and looks again. In the sharper replay that keeps 90% of the trades and removes about four in ten of the losing ones. Nearly all of the gain comes from some 30 markets where the old rule bought the collapsing side and the new rule ends up on the side that won.

The first evening showed it at work. At 6:59 PM ET Bitcoin dropped $20 in one second, to just under the line. The old rule would have bought NO at about 60¢ and lost. The bot waited, Bitcoin bounced back, and fifteen seconds later it bought YES at 73¢ and won $2.82.

The third loss was a milder case of the same thing, and the new check, as set, would not have stopped it: sellers had rushed in half a second earlier, but just under the threshold. A second later the market itself was paying 98¢ for YES again, and then Bitcoin fell another $29. A slightly stricter setting would have kept the bot out, though over the whole history the two settings are hard to tell apart. Some of this is simply the risk from section 2.

### 5. Stress tests and calibration

<img src="docs/img/stress-tests.svg" width="100%" alt="Stress tests of the backtest. As tested: 6,149 dollars. Bot four times slower: 6,092. Smaller orders: 4,902. Half as much available to buy: 4,106. Every fill two cents worse: 3,438. Model 50 percent more cautious: 2,903.">

The replay made harder on purpose: slower reactions, worse prices, less to buy. Profit shrinks and stays positive.

<img src="docs/img/calibration.svg" width="100%" alt="Calibration chart. When the model said 67 percent the favourite won 70 percent of the time; 84 gave 88; 92 gave 94; 96 gave 97; 98.4 gave 98.7; and at the top the two match.">

Can the model's confidence be trusted? When it said 96%, the favourite won about 97% of the time. It leans slightly cautious across the board.

### What can still go wrong

- **Sudden jumps.** A sharp move in the last few seconds turns a "sure thing" into a full loss. This is the main risk.
- **The market can know first.** Other traders see a move before the bot's price feed does. The newest check catches the obvious cases, not all of them.
- **The edge can fade.** It exists because some traders misprice the average. As more of them get it right, there is less to earn.
- **Real fills can be worse** than the replay assumes, especially for bigger accounts.
- **Replays can be too kind.** The first one was: it missed a third of the bot's trades. The sharper one still has to assume how much the bot could have bought, and nobody can check that in advance.
- **Recent days were harder.** Three losses in the last 27 real trades is about 11%, against roughly 4% in the replay of the same rule.
- **Software and connections fail.** The safety rails below limit the damage. They do not remove it.

## Safety rails

<img src="docs/img/safety.svg" width="100%" alt="Limits for a 100 dollar account: at most 6 dollars in one contract, at most 10 dollars riding on one closing moment, trading stops for the day after a 15 dollar loss, and a 30 percent fall from the peak switches the bot off.">

Every limit is a setting, and each has a hard dollar cap on top. The bot starts in **Demo** mode with fake money. Real trading stays locked until you unlock it on purpose by typing a confirmation phrase.

<details>
<summary><b>See the settings screen</b></summary>

<br>

<p align="center">
  <img src="docs/img/dashboard-settings.png" width="470" alt="The settings screen: risk limits as percentages of the account with hard dollar caps, and the strategy settings.">
</p>

</details>

## The dashboard

A black-and-white control room that runs in your browser.

<img src="docs/img/dashboard-live.png" width="100%" alt="The Live tab of the dashboard: time to settlement, the model's probability next to the market price, the current position, the risk limits, a chart of the Bitcoin index against the strike, and the order book.">

<p align="center"><sub>The Live tab in Paper mode: real Kalshi prices, a pretend $100 account.</sub></p>

<img src="docs/img/dashboard-backtest.png" width="100%" alt="The Backtest tab of the dashboard: profit, win rate, drawdown, the equity curve and the unseen-data check.">

<p align="center"><sub>The Backtest tab: the replay results above, straight from the research code.</sub></p>

- **Live:** the countdown, the model's odds next to the market's price, your position, your limits, and whether the trade feed is live.
- **Performance:** every trade and the running profit or loss.
- **Backtest:** the research results, so you can see what the rules did on past markets.
- **Settings:** mode, API key, limits and strategy numbers, plus a readiness check that never places an order.
- **Start, Stop, Kill:** Stop cancels open orders. Kill cancels everything and blocks trading until you release it.

## Try it

You need Python 3.12 or newer and a Kalshi account to create an API key.

```bash
git clone https://github.com/yusufbadar/KalshiBTCPredictionModel15.git
cd KalshiBTCPredictionModel15
pip install -r requirements.txt
python -m kalshi_bot
```

Then open **http://127.0.0.1:8000** and:

1. Go to **Settings**, choose a mode, paste your API key ID and private key, and press **Save**.
2. Press **Run readiness check**. It tests the login, the price feed and your limits.
3. Press **Start**.

| Mode | Money | Prices | Good for |
|---|---|---|---|
| **Paper** | pretend ($100 by default) | real Kalshi prices | watching the strategy work. Start here. |
| **Demo** | Kalshi's fake-money exchange | made-up test prices | checking that orders and limits behave |
| **Live** | real | real | only after you unlock it on purpose |

> [!NOTE]
> Profit or loss in **Demo** means nothing. Kalshi's demo exchange is filled by test bots at random prices, and its results do not match the real exchange. Use Paper to judge the strategy.

<details>
<summary><b>Going live with real money</b></summary>

<br>

1. Create an API key on Kalshi's real exchange and fund the account.
2. In **Settings** choose **Live**, type the phrase `TRADE REAL MONEY`, paste the production key ID and private key, and press **Save**. Demo and production keys are stored separately.
3. Press **Run readiness check**. Every line should say OK.
4. Press **Start**. Run only one copy of the bot per Kalshi account.

Start small. With the default limits a $100 account puts at most $6 into one contract.

</details>

<details>
<summary><b>Without the dashboard</b></summary>

<br>

```bash
python -m kalshi_bot.cli --env paper     # run in a terminal, Ctrl+C to stop
python -m kalshi_bot.preflight           # readiness check only
python -m pytest -q tests                # 119 tests
```

</details>

### Running it around the clock

<img src="docs/img/server.svg" width="100%" alt="The bot runs as a service on a small cloud server. The dashboard listens only inside the server and is reached from a laptop through an encrypted SSH tunnel, behind a password. The server talks to Kalshi in a few milliseconds.">

The bot has to stay switched on, so it lives best on a small Linux server for a few dollars a month. Serverless hosts such as Vercel cannot keep a program running, so they do not work. One script installs everything, and the dashboard stays private behind an SSH tunnel and a password.

<p align="center">
  <img src="docs/img/dashboard-login.png" width="440" alt="The dashboard sign-in page.">
</p>

Step-by-step guide: [deploy/README.md](deploy/README.md).

## How it is built

<img src="docs/img/architecture.svg" width="100%" alt="How the bot is built. Bitcoin's price and every Kalshi trade come in to the price feed. A model works out the odds that each side wins. Six checks decide whether to buy. Risk limits size the order. The order goes back to Kalshi. Every step is recorded in a trade log and shown on the dashboard.">

The research code and the live bot share one rule. A test replays the backtest's own trades through the live strategy and fails unless at least 99% of them match.

<details>
<summary><b>Project layout</b></summary>

<br>

```
kalshi_bot/
  api/          request signing, REST and WebSocket clients
  core/         order book, price index and clock, public trade tape, markets, orders,
                risk limits, trade ledger, paper broker, one-bot-per-account lock
  model/        fair value of the 60-second average, live volatility
  strategy/     the lock-in rule
  research/     data download, backtester, walk-forward report, unseen-data check,
                small-account simulation, trade-by-trade replay of the live code
  web/          dashboard server, login, the two HTML pages
  engine.py     the live trading loop        cli.py   terminal runner
  preflight.py  read-only readiness check
deploy/         server install: service file, setup script, bundle builder, guide
docs/           pictures in this README and the script that draws them
tests/          pricing, accounting, risk, login, dashboard and engine tests
legacy/         the original candle-pattern bot, kept for reference
```

</details>

<details>
<summary><b>Reproduce the numbers</b></summary>

<br>

Everything is measured on Kalshi's public trade data, which the first command downloads.

```bash
python -m kalshi_bot.research.history --days 14 --tail-seconds 130            # newest 15-minute markets
python -m kalshi_bot.research.history --series KXBTCD --days 14 --min-volume 20000 --tail-seconds 130
python -m kalshi_bot.research.report --series KXBTC15M     # walk-forward + unseen-data check
python -m kalshi_bot.research.report --series KXBTCD
python -m kalshi_bot.research.holdout                      # fixed rules on markets after the freeze
python -m kalshi_bot.research.account                      # what it means for a $100 account
python -m kalshi_bot.research.tape_replay --per-market 0.10 --per-close 0.15   # the live code, trade by trade
python docs/make_figures.py                                # redraw the charts in this README
```

</details>

## The story so far

<img src="docs/img/story.svg" width="100%" alt="Timeline. March 2026: a candle-pattern bot, scrapped after it lost in a replay on real fills. 27 September: lock-in rules written down and frozen. 28 September to 10 October: tested on unseen data, still profitable. 10 October: live with 100 dollars on a small cloud server. 11 October: the first day lost 10 dollars and the rule was tightened. 13 October: two more losses traced to the market moving before the price feed, and the bot now waits a second when that happens.">

Ideas that were tested and thrown out:

| Idea | What happened |
|---|---|
| Predicting the next move from price candles (the original bot) | Lost 1.2¢ per contract once real prices and fees were counted |
| Buying before the final minute | Lost about 6¢ per contract, even when the model was very confident |
| Leaving orders in the book and waiting to be filled | Filled mostly when it was wrong |
| Betting on favourites in general | Too weak to tell apart from luck |
| The same rule on Ethereum and XRP | Break-even or inconsistent, so they stay off |
| Looser entry rules, to trade more often | More trades, and dips twice as deep |
| Switching off the hourly markets after a loss there | Gives up about a fifth of the profit, and the dips are no smaller |
| Stricter entry: a surer model, a bigger discount, a lower price cap | Fewer trades, and about as many losses |

## Words used here

| Word | Meaning |
|---|---|
| **Contract** | One YES or NO ticket. It costs 1¢ to 99¢ and pays $1 if it is right. |
| **The line** (strike) | The price Bitcoin has to finish above for YES to win. |
| **Settlement** | The moment Kalshi decides the result and pays out. |
| **Index** | The Bitcoin price Kalshi settles on (CF Benchmarks BRTI), published once a second. |
| **Trade tape** | The public list of every trade in a market, as it happens. |
| **Edge** | How much cheaper a contract is than it should be, after fees. |
| **Replay** (backtest) | Running the rules over past markets to see what would have happened. |
| **Paper trading** | Pretend trades against real prices. |
| **Dip** (drawdown) | The fall from the account's highest point to a later low. |
| **Kill switch** | A stop that cancels everything and stays on until a person releases it. |

## Disclaimer

This is software for research and education. Trading on prediction markets carries a real risk of loss, including the whole stake on any contract. Backtests and simulations are not guarantees, and live results can be worse. Nothing here is investment advice. Only trade money you can afford to lose.
