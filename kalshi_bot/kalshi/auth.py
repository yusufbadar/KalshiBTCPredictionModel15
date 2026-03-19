"""
Kalshi API authentication using RSA-PSS signatures.

Every authenticated request requires three headers:
  KALSHI-ACCESS-KEY       – API key ID
  KALSHI-ACCESS-TIMESTAMP – current epoch milliseconds
  KALSHI-ACCESS-SIGNATURE – RSA-PSS(SHA-256) of  timestamp+METHOD+path
"""
import base64
import datetime
from pathlib import Path
from urllib.parse import urlparse

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from loguru import logger


class KalshiAuth:
    """Handles loading the RSA private key and signing requests."""

    def __init__(self, api_key_id: str, private_key_path: str):
        self.api_key_id = api_key_id
        self._private_key = self._load_key(private_key_path)
        logger.info("Kalshi auth initialised (key_id={}…)", api_key_id[:12])

    @staticmethod
    def _load_key(path: str):
        raw = Path(path).read_bytes()
        return serialization.load_pem_private_key(
            raw, password=None, backend=default_backend()
        )

    def sign(self, method: str, path: str) -> dict:
        """Return the three auth headers for *method* + *path*.

        *path* must be the full path starting with ``/trade-api/v2/…``.
        Query parameters are stripped before signing.
        """
        timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        path_no_qs = path.split("?")[0]
        message = f"{timestamp}{method.upper()}{path_no_qs}".encode()

        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }
