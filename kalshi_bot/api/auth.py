"""
Kalshi request signing (Ed25519 or RSA keys).

Every authenticated REST call and the WebSocket handshake carry three headers:

    KALSHI-ACCESS-KEY        API key id
    KALSHI-ACCESS-TIMESTAMP  epoch milliseconds
    KALSHI-ACCESS-SIGNATURE  base64(sign(timestamp + METHOD + path))

``path`` is the full path from the API root (``/trade-api/v2/...``) with the
query string removed. Ed25519 keys sign the text directly; RSA keys use
RSA-PSS / SHA-256 with MGF1(SHA-256) and digest-length salt.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def load_private_key(pem_or_path: str):
    text = pem_or_path.strip()
    raw = text.encode() if text.startswith("-----BEGIN") else Path(text).expanduser().read_bytes()
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, (Ed25519PrivateKey, rsa.RSAPrivateKey)):
        raise ValueError(f"Unsupported Kalshi key type: {type(key).__name__}")
    return key


class Signer:
    def __init__(self, key_id: str, private_key_pem_or_path: str):
        if not key_id:
            raise ValueError("Missing Kalshi API key id")
        self.key_id = key_id.strip()
        self._key = load_private_key(private_key_pem_or_path)

    @property
    def key_type(self) -> str:
        return "ed25519" if isinstance(self._key, Ed25519PrivateKey) else "rsa"

    def sign(self, text: str) -> str:
        msg = text.encode("utf-8")
        if isinstance(self._key, Ed25519PrivateKey):
            sig = self._key.sign(msg)
        else:
            sig = self._key.sign(
                msg,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                hashes.SHA256(),
            )
        return base64.b64encode(sig).decode("ascii")

    def headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        path = path.split("?", 1)[0]
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self.sign(ts + method.upper() + path),
        }
