"""
Dashboard login.

The dashboard can start, stop and reconfigure a bot that trades real money,
so anything beyond this machine's own loopback address must be locked.

* One password, stored only as a salted scrypt hash in ``storage/auth.json``.
  Set it with ``python -m kalshi_bot --set-password``.
* A successful login gets a signed session cookie (HttpOnly, SameSite=Strict,
  Secure over HTTPS) that expires after ``SESSION_HOURS``.
* Wrong passwords are throttled: after 5 misses the login locks for a minute,
  doubling up to 15 minutes.
* Requests whose ``Origin`` is another site are refused (a web page you visit
  cannot press Start on your bot), and without a password only loopback host
  names are served (blocks DNS-rebinding tricks against a local dashboard).

With no password set the dashboard only works on 127.0.0.1 - ``main()``
refuses to listen on any other address.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from kalshi_bot.config import STORAGE_DIR

AUTH_PATH = STORAGE_DIR / "auth.json"
COOKIE = "lockin_session"
SESSION_HOURS = 12.0
MIN_PASSWORD = 10
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
_SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)


class Auth:
    def __init__(self, path: Path | None = None):
        self.path = path or AUTH_PATH
        self._salt = self._hash = self._secret = None
        self._fails = 0
        self._locked_until = 0.0
        self._mtime = None
        self._checked = 0.0
        self.load()

    # ------------------------------------------------------------- storage
    def load(self) -> None:
        self._salt = self._hash = self._secret = None
        self._mtime = self._stat()
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            self._salt, self._hash, self._secret = _unb64(d["salt"]), _unb64(d["hash"]), _unb64(d["secret"])
        except (OSError, ValueError, KeyError):
            pass

    def _stat(self):
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return None

    def maybe_reload(self) -> None:
        """Pick up a password set (or removed) from the command line while the
        dashboard is running. Checks the file at most every two seconds."""
        now = time.time()
        if now - self._checked < 2.0:
            return
        self._checked = now
        if self._stat() != self._mtime:
            self.load()

    @property
    def enabled(self) -> bool:
        return self._hash is not None

    def set_password(self, password: str) -> None:
        """Store a new password. Signing key is replaced too, which logs
        every existing session out."""
        if len(password) < MIN_PASSWORD:
            raise ValueError(f"Use at least {MIN_PASSWORD} characters.")
        salt, secret = secrets.token_bytes(16), secrets.token_bytes(32)
        data = {"salt": _b64(salt), "hash": _b64(hash_password(password, salt)), "secret": _b64(secret)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(self.path)
        self.load()

    def clear(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass
        self.load()

    # --------------------------------------------------------------- login
    def locked_for(self) -> float:
        return max(0.0, self._locked_until - time.time())

    def check_password(self, password: str) -> bool:
        """Constant-time check with throttling. Raises PermissionError while locked."""
        if not self.enabled:
            return False
        wait = self.locked_for()
        if wait > 0:
            raise PermissionError(f"Too many wrong passwords. Try again in {int(wait) + 1} s.")
        ok = hmac.compare_digest(hash_password(password, self._salt), self._hash)
        if ok:
            self._fails = 0
            return True
        self._fails += 1
        if self._fails >= 5:
            self._locked_until = time.time() + min(900.0, 60.0 * 2 ** (self._fails - 5))
        return False

    # ------------------------------------------------------------- sessions
    def new_session(self, hours: float = SESSION_HOURS) -> str:
        body = f"{int(time.time() + hours * 3600)}.{secrets.token_hex(8)}"
        sig = hmac.new(self._secret, body.encode(), hashlib.sha256).digest()
        return f"{body}.{_b64(sig)}"

    def valid_session(self, token: str | None) -> bool:
        if not self.enabled or not token:
            return False
        try:
            exp, nonce, sig = token.split(".")
            body = f"{exp}.{nonce}"
            good = hmac.new(self._secret, body.encode(), hashlib.sha256).digest()
            return hmac.compare_digest(_unb64(sig), good) and int(exp) > time.time()
        except (ValueError, TypeError):
            return False


def host_only(value: str | None) -> str:
    """'example.com:8000' / '[::1]:8000' -> host part, lower case."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        return v[: v.find("]") + 1] if "]" in v else v
    return v.rsplit(":", 1)[0] if ":" in v else v


def same_origin(origin: str | None, host_header: str | None) -> bool:
    """True when a request's Origin (if any) is this dashboard itself."""
    if not origin or origin == "null":
        return origin is None or origin == ""       # no Origin header: not a cross-site browser request
    o = origin.split("://", 1)[-1].lower()
    return o == (host_header or "").strip().lower()


def is_loopback(host: str) -> bool:
    return host_only(host) in LOOPBACK_HOSTS or host in LOOPBACK_HOSTS


def set_password_interactive() -> None:
    import getpass
    a = Auth()
    print("Set the dashboard password (nothing is shown while you type).")
    p1 = getpass.getpass("New password: ")
    p2 = getpass.getpass("Repeat it:    ")
    if p1 != p2:
        raise SystemExit("The two entries differ - nothing was changed.")
    try:
        a.set_password(p1)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    print(f"Password saved (as a salted hash) in {a.path}. A running dashboard picks it up within seconds "
          f"and signs everyone out.")
