# Running the bot on a server

The bot is one long-running program: it keeps live connections to Kalshi open
and reacts inside the last minute of every market. It needs a machine that is
always on. A small Linux server does that for a few dollars a month; your
laptop can then be closed.

**Not Vercel / Netlify / other "serverless" hosts.** Those run short functions
on demand and shut them down. They cannot hold a 24/7 process, a WebSocket
feed or the bot's trade database.

## What to rent

| | |
|---|---|
| Type | a plain virtual server (VPS) |
| Size | 1 CPU, 1 GB RAM, 10+ GB disk is enough |
| System | Ubuntu 24.04 LTS (or newer) |
| Region | **US East (Ohio)** if offered — Kalshi's API runs in AWS `us-east-2`, so the connection to it is as short as it gets. Any US East / US Central region is fine. |
| Login | add your SSH key when you create it (no password logins) |

AWS Lightsail in Ohio is the closest match; DigitalOcean, Vultr, Hetzner (US)
and similar all work.

## Install (about five minutes)

On your computer, in the project folder:

```bash
python deploy/make_bundle.py
```

```bash
scp lockin-server.zip root@YOUR_SERVER:/root/
```

The bundle is code only (about 150 KB). It contains no API keys, settings or
trade history.

On AWS Lightsail and most Ubuntu images the login user is `ubuntu`, not
`root`: use `ubuntu@YOUR_SERVER` everywhere below and put `sudo` in front of
the server commands.

Then on the server (`ssh root@YOUR_SERVER`):

```bash
apt-get update && apt-get install -y unzip && mkdir -p /opt/lockin && unzip -o /root/lockin-server.zip -d /opt/lockin
```

```bash
bash /opt/lockin/deploy/setup.sh
```

`setup.sh` installs Python packages into a private environment, creates a
`lockin` service account that can write only its own data and key folders,
installs the service so it starts on boot and restarts if it ever crashes, and
turns on the firewall with only SSH open.

Set the dashboard password (you type it twice; it is stored only as a hash):

```bash
cd /opt/lockin && sudo -u lockin .venv/bin/python -m kalshi_bot --set-password
```

## Open the dashboard — privately

The dashboard is **not** on the internet. It listens only inside the server,
and you reach it through an encrypted SSH tunnel. On your computer:

```bash
ssh -N -L 8000:127.0.0.1:8000 root@YOUR_SERVER
```

Leave that window open and browse to **http://localhost:8000**. Sign in with
the dashboard password, then:

1. **Settings** → choose the exchange your key belongs to — **Demo** for a
   demo key, **Live** for a production key (type the unlock phrase) → paste
   the API key ID and private key → **Save**. Saving never starts trading.
   The key travels through the SSH tunnel and is stored on the server in
   `/opt/lockin/secrets` (readable only by the service). To paper-trade with
   a production key, save it under **Live** first, then select **Paper** and
   save again (a key pasted while Paper is selected is filed as a demo key).
2. **Run readiness check** — every line should say OK.
3. **Start**.

Close the tunnel and the laptop whenever you like; the bot keeps running on
the server. Open the tunnel again to look in on it.

Run it in **Paper** on the server for a couple of days first. It trades the
real production prices with a simulated $100, so you see how often it trades
and whether the server is stable before any money is involved.

## Two locks, on purpose

| Lock | What it stops |
|---|---|
| SSH tunnel + firewall | Nobody on the internet can even reach the dashboard; only someone with your SSH key can. |
| Dashboard password | Anyone who does reach it still has to sign in. Wrong passwords lock the login (1 minute, doubling to 15). Sessions end after 12 hours. |

The dashboard also refuses requests that come from other websites, and it will
not start on a public address unless a password is set.

## What happens when things go wrong

* **Server reboots or the program crashes:** the service restarts and, if the
  bot was running, it resumes by itself (`--autostart`). Open positions are
  restored from the trade database.
* **You pressed Stop:** it stays stopped, even after a reboot.
* **Kill switch engaged (by you or by the 30% drawdown stop):** it stays
  engaged across restarts until you release it in the dashboard.
* **Kalshi maintenance / lost connection:** the bot stops sending orders,
  reconnects and carries on.

## Day to day

```bash
journalctl -u lockin -f          # live log
systemctl status lockin          # is it running?
systemctl restart lockin         # restart
```

Update to a newer version: build and upload a new bundle, unzip it over
`/opt/lockin` and run `setup.sh` again. Settings, keys and history are kept.

Back up `/opt/lockin/storage/bot.sqlite3` (the trade history) now and then.

## Rules that keep the money safe

* **One bot per Kalshi account.** Do not run Live on the server and on your
  laptop at the same time — each would take full-size positions.
* Use a **separate API key** for the server and delete it on Kalshi if the
  server is ever lost or retired.
* Keep the server updated: `apt-get update && apt-get upgrade`.

## Optional: use it from your phone

Install [Tailscale](https://tailscale.com) on the server and on your phone and
laptop. It gives the server a private address (100.x.y.z) that only your own
devices can reach — still nothing public. Then tell the service to listen
there (a password must already be set):

```bash
systemctl edit lockin
```

add

```
[Service]
Environment=LOCKIN_HOST=100.x.y.z
```

then `systemctl restart lockin` and browse to `http://100.x.y.z:8000` from any
of your devices. With `ufw` enabled also run `ufw allow in on tailscale0`.

Putting the dashboard on a public web address is possible (a reverse proxy
with HTTPS in front, password set) but there is no reason to: the tunnel or
Tailscale gives you the same dashboard with nothing exposed.
