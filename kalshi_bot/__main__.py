"""``python -m kalshi_bot`` starts the dashboard (the bot is started from the UI).

    python -m kalshi_bot                      # http://127.0.0.1:8000, this machine only
    python -m kalshi_bot --set-password       # set / change the dashboard password
    python -m kalshi_bot --autostart          # resume the bot by itself after a restart (servers)
    python -m kalshi_bot --host 0.0.0.0       # listen on the network - needs a password
"""
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lock-In trading bot dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--autostart", action="store_true",
                    help="if the bot was running when the program last stopped, start it again")
    ap.add_argument("--set-password", action="store_true", help="set the dashboard password and exit")
    ap.add_argument("--remove-password", action="store_true", help="remove the dashboard password and exit")
    a = ap.parse_args()
    if a.set_password:
        from kalshi_bot.web.auth import set_password_interactive
        set_password_interactive()
    elif a.remove_password:
        from kalshi_bot.web.auth import Auth
        Auth().clear()
        print("Dashboard password removed. Without one the dashboard only works on 127.0.0.1.")
    else:
        from kalshi_bot.web.server import main
        main(a.host, a.port, a.autostart)
