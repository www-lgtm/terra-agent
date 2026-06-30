"""WeChat iLink Bot adapter for Terra Agent.

Minimal implementation derived from Hermes weixin.py.
Handles QR login, long-poll receive, message send.

Usage:
    python -m src.cli weixin    # Interactive QR login + start bot
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import ssl
import struct
import threading
import time
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from src.concierge.router import MessageRouter
from config.settings import config

logger = logging.getLogger(__name__)


def _create_ssl_context() -> ssl.SSLContext:
    """Build SSL context from GatewayConfig.

    - ssl_verify=True, no custom cert → system CA bundle (secure default)
    - ssl_verify=True, custom cert → load from ssl_cert_path
    - ssl_verify=False → emit warning, use CERT_NONE (insecure — dev only)
    """
    gw = config.gateway
    ctx = ssl.create_default_context()

    if not gw.ssl_verify:
        logger.warning(
            "SSL verification DISABLED via GATEWAY_SSL_VERIFY=false — "
            "all iLink traffic is vulnerable to MITM attacks"
        )
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    if gw.ssl_cert_path:
        logger.info("Loading custom SSL cert from %s", gw.ssl_cert_path)
        ctx.load_verify_locations(gw.ssl_cert_path)
        return ctx

    # Default: system CA bundle (secure)
    return ctx

ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
ILINK_APP_ID = "bot"
CHANNEL_VERSION = "2.2.0"
ILINK_APP_CLIENT_VERSION = (2 << 16) | (2 << 8) | 0

EP_GET_UPDATES = "ilink/bot/getupdates"
EP_SEND_MESSAGE = "ilink/bot/sendmessage"
EP_GET_BOT_QR = "ilink/bot/get_bot_qrcode"
EP_GET_QR_STATUS = "ilink/bot/get_qrcode_status"
EP_GET_UPLOAD_URL = "ilink/bot/getuploadurl"

LONG_POLL_TIMEOUT_MS = 25_000
QR_TIMEOUT_MS = 35_000
RETRY_DELAY_SECONDS = 2.0

# iLink media types
MEDIA_IMAGE = 1
ITEM_IMAGE = 2  # type value in item_list for image messages


def _random_wechat_uin() -> str:
    value = struct.unpack(">I", secrets.token_bytes(4))[0]
    return __import__("base64").b64encode(str(value).encode()).decode()


def _base_info() -> dict[str, Any]:
    return {"channel_version": CHANNEL_VERSION}


# ── iLink media upload helpers (ported from Hermes weixin.py) ──

def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)


def _aes128_ecb_encrypt(plaintext: bytes, key: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    return encryptor.update(_pkcs7_pad(plaintext)) + encryptor.finalize()


def _aes_padded_size(size: int) -> int:
    return ((size + 1 + 15) // 16) * 16


def _cdn_upload_url(cdn_base_url: str, upload_param: str, filekey: str) -> str:
    return (
        f"{cdn_base_url.rstrip('/')}/upload"
        f"?encrypted_query_param={quote(upload_param, safe='')}"
        f"&filekey={quote(filekey, safe='')}"
    )


async def _get_upload_url(
    session,
    *,
    base_url: str,
    token: str,
    to_user_id: str,
    media_type: int,
    filekey: str,
    rawsize: int,
    rawfilemd5: str,
    filesize: int,
    aeskey_hex: str,
) -> dict[str, Any]:
    return await _api_post(
        session,
        endpoint=EP_GET_UPLOAD_URL,
        payload={
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": to_user_id,
            "rawsize": rawsize,
            "rawfilemd5": rawfilemd5,
            "filesize": filesize,
            "no_need_thumb": True,
            "aeskey": aeskey_hex,
        },
        token=token,
        base_url=base_url,
        timeout_ms=15_000,
    )


async def _upload_ciphertext(
    session,
    *,
    ciphertext: bytes,
    upload_url: str,
) -> str:
    """Upload encrypted media to the CDN. Returns x-encrypted-param."""
    import aiohttp

    async def _do_upload() -> str:
        async with session.post(
            upload_url,
            data=ciphertext,
            headers={"Content-Type": "application/octet-stream"},
        ) as response:
            if response.status == 200:
                encrypted_param = response.headers.get("x-encrypted-param")
                if encrypted_param:
                    await response.read()
                    return encrypted_param
                raw = await response.text()
                raise RuntimeError(
                    f"CDN upload missing x-encrypted-param header: {raw[:200]}"
                )
            raw = await response.text()
            raise RuntimeError(f"CDN upload HTTP {response.status}: {raw[:200]}")

    return await asyncio.wait_for(_do_upload(), timeout=120)


async def _send_media_message(
    session,
    *,
    base_url: str,
    token: str,
    to_user_id: str,
    image_b64: str,
    caption: str = "",
    context_token: str = "",
) -> str | None:
    """Upload image via iLink CDN and send as native image message.

    Returns the message_id on success, or None if the upload/send fails.
    """
    import uuid

    plaintext = base64.b64decode(image_b64)
    filekey = secrets.token_hex(16)
    aes_key = secrets.token_bytes(16)
    rawsize = len(plaintext)
    rawfilemd5 = hashlib.md5(plaintext).hexdigest()

    # Get CDN upload URL
    upload_response = await _get_upload_url(
        session,
        base_url=base_url,
        token=token,
        to_user_id=to_user_id,
        media_type=MEDIA_IMAGE,
        filekey=filekey,
        rawsize=rawsize,
        rawfilemd5=rawfilemd5,
        filesize=_aes_padded_size(rawsize),
        aeskey_hex=aes_key.hex(),
    )
    upload_param = str(upload_response.get("upload_param") or "")
    upload_full_url = str(upload_response.get("upload_full_url") or "")

    # Encrypt and upload
    ciphertext = _aes128_ecb_encrypt(plaintext, aes_key)

    if upload_full_url:
        upload_url = upload_full_url
    elif upload_param:
        cdn_base = str(upload_response.get("cdn_base_url") or base_url)
        upload_url = _cdn_upload_url(cdn_base, upload_param, filekey)
    else:
        raise RuntimeError(
            f"getUploadUrl returned neither upload_param nor upload_full_url: {upload_response}"
        )

    encrypted_query_param = await _upload_ciphertext(
        session,
        ciphertext=ciphertext,
        upload_url=upload_url,
    )

    # The iLink API expects aes_key as base64(hex_string), NOT base64(raw_bytes)
    aes_key_for_api = base64.b64encode(aes_key.hex().encode("ascii")).decode("ascii")

    # Build image message
    image_msg: dict[str, Any] = {
        "from_user_id": "",
        "to_user_id": to_user_id,
        "client_id": f"terra-{uuid.uuid4().hex}",
        "message_type": 2,
        "message_state": 2,
        "item_list": [
            {
                "type": ITEM_IMAGE,
                "image_item": {
                    "media": {
                        "encrypt_query_param": encrypted_query_param,
                        "aes_key": aes_key_for_api,
                        "encrypt_type": 1,
                    },
                    "mid_size": len(ciphertext),
                },
            },
        ],
    }
    if context_token:
        image_msg["context_token"] = context_token

    await _api_post(
        session,
        endpoint=EP_SEND_MESSAGE,
        payload={"msg": image_msg},
        token=token,
        base_url=base_url,
        timeout_ms=15_000,
    )
    return image_msg["client_id"]


def _headers(token: str | None, body: str) -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "Content-Length": str(len(body.encode("utf-8"))),
        "X-WECHAT-UIN": _random_wechat_uin(),
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _account_dir(data_home: str) -> Path:
    p = Path(data_home) / "weixin" / "accounts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_account(data_home: str, account_id: str, token: str, base_url: str) -> None:
    from src.utils.crypto import encrypt_token
    encrypted_token = encrypt_token(token)
    payload = {"token": encrypted_token, "base_url": base_url, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "encrypted": True}
    path = _account_dir(data_home) / f"{account_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_account(data_home: str, account_id: str) -> dict[str, Any] | None:
    path = _account_dir(data_home) / f"{account_id}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("encrypted"):
            from src.utils.crypto import decrypt_token
            payload["token"] = decrypt_token(payload["token"])
        return payload
    except Exception:
        return None


def _list_accounts(data_home: str) -> list[str]:
    d = _account_dir(data_home)
    if not d.exists():
        return []
    return [p.stem for p in d.glob("*.json")]


def _clear_account(data_home: str, account_id: str) -> None:
    """Delete saved credentials for an account."""
    path = _account_dir(data_home) / f"{account_id}.json"
    if path.exists():
        path.unlink()
    logger.info("Cleared credentials for account=%s", account_id[:12])


# ---- QR Login ----

async def qr_login(data_home: str, *, bot_type: str = "3", timeout_seconds: int = 480) -> dict[str, str] | None:
    """Interactive iLink QR login. Prints QR in terminal, waits for scan."""
    import aiohttp

    ssl_ctx = _create_ssl_context()

    print("\n=== WeChat iLink Bot Login ===")
    print("正在获取二维码...")

    async with aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(ssl=ssl_ctx)) as session:

        try:
            qr_resp = await _api_get(session, f"{EP_GET_BOT_QR}?bot_type={bot_type}", timeout_ms=QR_TIMEOUT_MS)
        except Exception as exc:
            print(f"获取二维码失败: {exc}")
            logger.error("QR fetch failed: %s", exc)
            return None

        qrcode_value = str(qr_resp.get("qrcode") or "")
        qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
        if not qrcode_value:
            print(f"二维码接口返回异常: {json.dumps(qr_resp, ensure_ascii=False)[:300]}")
            logger.error("QR response missing qrcode: %s", json.dumps(qr_resp, ensure_ascii=False)[:300])
            return None

        qr_scan_data = qrcode_url or qrcode_value
        print("请使用微信扫描以下二维码：")
        print(qr_scan_data)

        try:
            import qrcode as _qrcode
            qr = _qrcode.QRCode()
            qr.add_data(qr_scan_data)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except Exception:
            print("(终端二维码渲染失败，请直接打开上面的链接)")

        deadline = time.monotonic() + timeout_seconds
        current_base_url = ILINK_BASE_URL
        refresh_count = 0

        while time.monotonic() < deadline:
            try:
                status_resp = await _api_get(
                    session, f"{EP_GET_QR_STATUS}?qrcode={qrcode_value}", timeout_ms=QR_TIMEOUT_MS
                )
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
                continue
            except Exception as exc:
                logger.warning("QR poll error: %s", exc)
                await asyncio.sleep(1)
                continue

            status = str(status_resp.get("status") or "wait")
            if status == "wait":
                print(".", end="", flush=True)
            elif status == "scaned":
                print("\n已扫码，请在微信里确认...")
            elif status == "scaned_but_redirect":
                redirect_host = str(status_resp.get("redirect_host") or "")
                if redirect_host:
                    current_base_url = f"https://{redirect_host}"
            elif status == "expired":
                refresh_count += 1
                if refresh_count > 3:
                    print("\n二维码多次过期，请重新执行登录。")
                    return None
                print(f"\n二维码已过期，正在刷新... ({refresh_count}/3)")
                try:
                    qr_resp = await _api_get(
                        session, f"{EP_GET_BOT_QR}?bot_type={bot_type}", timeout_ms=QR_TIMEOUT_MS
                    )
                    qrcode_value = str(qr_resp.get("qrcode") or "")
                    qrcode_url = str(qr_resp.get("qrcode_img_content") or "")
                    qr_scan_data = qrcode_url or qrcode_value
                    if qrcode_url:
                        print(qrcode_url)
                    try:
                        import qrcode
                        qr = qrcode.QRCode()
                        qr.add_data(qr_scan_data)
                        qr.make(fit=True)
                        qr.print_ascii(invert=True)
                    except Exception:
                        pass
                except Exception as exc:
                    logger.error("QR refresh failed: %s", exc)
                    return None
            elif status == "confirmed":
                account_id = str(status_resp.get("ilink_bot_id") or "")
                token = str(status_resp.get("bot_token") or "")
                base_url = str(status_resp.get("baseurl") or ILINK_BASE_URL)
                if not account_id or not token:
                    logger.error("QR confirmed but credentials incomplete")
                    return None
                _save_account(data_home, account_id, token, base_url)
                print(f"\n登录成功! account={account_id[:12]}...")
                return {"account_id": account_id, "token": token, "base_url": base_url}

            await asyncio.sleep(0.5)

        print("\n登录超时。")
        return None


# ---- API helpers ----

async def _api_get(session, endpoint: str, timeout_ms: int = 30_000) -> dict[str, Any]:
    import aiohttp
    url = f"{ILINK_BASE_URL}/{endpoint}"
    headers = {
        "iLink-App-Id": ILINK_APP_ID,
        "iLink-App-ClientVersion": str(ILINK_APP_CLIENT_VERSION),
    }
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.get(url, headers=headers, timeout=timeout) as resp:
        raw = await resp.text()
        if not resp.ok:
            raise RuntimeError(f"iLink GET {endpoint} HTTP {resp.status}: {raw[:200]}")
        return json.loads(raw)


async def _api_post(session, endpoint: str, payload: dict[str, Any], token: str, base_url: str, timeout_ms: int = 30_000) -> dict[str, Any]:
    import aiohttp
    body = json.dumps({**payload, "base_info": _base_info()}, ensure_ascii=False, separators=(",", ":"))
    url = f"{base_url.rstrip('/')}/{endpoint}"
    timeout = aiohttp.ClientTimeout(total=timeout_ms / 1000)
    async with session.post(url, data=body, headers=_headers(token, body), timeout=timeout) as resp:
        raw = await resp.text()
        if not resp.ok:
            raise RuntimeError(f"iLink POST {endpoint} HTTP {resp.status}: {raw[:200]}")
        return json.loads(raw)


def _extract_text(item_list: list[dict]) -> str:
    parts: list[str] = []
    for item in item_list:
        if item.get("type") == 1:
            text = item.get("text") or item.get("text_item", {}).get("text", "")
            parts.append(str(text) if text else "")
    return "".join(parts)


# ---- WeixinBot ----

class WeixinBot:
    """Minimal iLink Bot: long-poll messages, process with agent, send replies."""

    def __init__(
        self,
        account_id: str,
        token: str,
        base_url: str,
        data_home: str,
        handle_message: Any = None,  # async callable(text, user_id) -> str
    ):
        self.account_id = account_id
        self.token = token
        self.base_url = base_url
        self.data_home = data_home
        self.handle_message = handle_message

        self._poll_session: Any = None
        self._send_session: Any = None
        self._poll_task: asyncio.Task | None = None
        self._running = False
        self._context_tokens: dict[str, str] = {}
        self._sync_buf = self._load_sync_buf()
        self._last_send_time: float = 0.0   # Throttle consecutive sends
        self._send_lock = asyncio.Lock()     # Serialize _post_msg calls

    def _sync_buf_path(self) -> Path:
        return Path(self.data_home) / "weixin" / f"{self.account_id}.sync_buf"

    def _load_sync_buf(self) -> str:
        p = self._sync_buf_path()
        if p.exists():
            return p.read_text().strip()
        return ""

    def _save_sync_buf(self, buf: str) -> None:
        self._sync_buf_path().parent.mkdir(parents=True, exist_ok=True)
        self._sync_buf_path().write_text(buf)

    async def start(self) -> None:
        import aiohttp

        ssl_ctx = _create_ssl_context()

        self._poll_session = aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(ssl=ssl_ctx))
        self._send_session = aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(ssl=ssl_ctx))
        self._running = True
        # Poll loop is driven directly by run_bot(), not as a task
        logger.info("WeixinBot started: account=%s", self.account_id[:12])

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        for s in (self._poll_session, self._send_session):
            if s and not s.closed:
                await s.close()

    async def _reauth(self) -> bool:
        """Re-authenticate when token expires. Clears old creds, shows QR, updates state.

        Returns True on success, False if the user didn't scan in time.
        """
        # Clear old credentials
        _clear_account(self.data_home, self.account_id)
        # Close old sessions
        for s in (self._poll_session, self._send_session):
            if s and not s.closed:
                await s.close()
        self._poll_session = None
        self._send_session = None

        # Do QR login
        result = await qr_login(self.data_home)
        if not result:
            logger.error("Re-auth QR login failed — no result")
            return False

        # Update state
        self.account_id = result["account_id"]
        self.token = result["token"]
        self.base_url = result["base_url"]
        self._sync_buf = ""
        self._context_tokens.clear()

        # Recreate sessions
        import aiohttp as _aiohttp
        ssl_ctx = _create_ssl_context()
        self._poll_session = _aiohttp.ClientSession(trust_env=True, connector=_aiohttp.TCPConnector(ssl=ssl_ctx))
        self._send_session = _aiohttp.ClientSession(trust_env=True, connector=_aiohttp.TCPConnector(ssl=ssl_ctx))

        logger.info("Re-auth successful: account=%s", self.account_id[:12])
        print(f"✅ 重新登录成功: {self.account_id[:12]}...")
        return True

    async def _poll_loop(self) -> None:
        timeout_ms = LONG_POLL_TIMEOUT_MS
        failures = 0
        # Token expiration detection: errcode values that indicate auth failure
        _AUTH_ERRCODE = frozenset({40001, 40014, 41001, 42001, 42002, 43004})
        _MAX_AUTH_FAILURES = 3

        while self._running:
            try:
                response = await _api_post(
                    self._poll_session,
                    endpoint=EP_GET_UPDATES,
                    payload={"get_updates_buf": self._sync_buf},
                    token=self.token,
                    base_url=self.base_url,
                    timeout_ms=timeout_ms,
                )

                suggested = response.get("longpolling_timeout_ms")
                if isinstance(suggested, int) and suggested > 0:
                    timeout_ms = suggested

                ret = response.get("ret", 0)
                errcode = response.get("errcode", 0)
                if ret not in (0, None) or errcode not in (0, None):
                    failures += 1
                    logger.warning("getUpdates error ret=%s errcode=%s", ret, errcode)

                    # Detect token expiration
                    if errcode in _AUTH_ERRCODE and failures >= _MAX_AUTH_FAILURES:
                        logger.error("Token expired (errcode=%s) after %d failures — attempting re-login",
                                   errcode, failures)
                        print(f"\n⚠️  Token 已过期 (errcode={errcode})，请重新扫码登录...")
                        if await self._reauth():
                            failures = 0
                            timeout_ms = LONG_POLL_TIMEOUT_MS
                            continue
                        else:
                            logger.error("Re-login failed — stopping bot")
                            self._running = False
                            break

                    await asyncio.sleep(RETRY_DELAY_SECONDS * min(failures, 5))
                    continue

                failures = 0
                new_buf = str(response.get("get_updates_buf") or "")
                if new_buf:
                    self._sync_buf = new_buf
                    self._save_sync_buf(new_buf)

                for msg in response.get("msgs") or []:
                    await self._handle_message(msg)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                failures += 1
                exc_type = type(exc).__name__
                exc_msg = str(exc) or "(no message)"
                logger.error("poll error (%d): %s: %s", failures, exc_type, exc_msg)
                # Log full traceback on first failure of each type for diagnosis
                if failures == 1:
                    logger.debug("Poll error traceback", exc_info=True)
                await asyncio.sleep(RETRY_DELAY_SECONDS * min(failures, 10))

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        try:
            sender_id = str(msg.get("from_user_id") or "").strip()
            if not sender_id or sender_id == self.account_id:
                print(f"[weixin] 过滤消息: sender={sender_id[:20] if sender_id else '(空)'} self={self.account_id[:20]}")
                return

            ctx_token = str(msg.get("context_token") or "").strip()
            if ctx_token:
                self._context_tokens[sender_id] = ctx_token

            item_list = msg.get("item_list") or []
            text = _extract_text(item_list)
            if not text:
                return

            logger.info("[weixin] 收到: %s", text[:100])

            if self.handle_message:
                response = await self.handle_message(text, sender_id)
                if isinstance(response, dict):
                    if response.get("start_msg"):
                        await self.send_message(sender_id, response["start_msg"])
                    if response.get("reply"):
                        await self.send_message(sender_id, response["reply"])
                elif response:
                    await self.send_message(sender_id, str(response))

        except Exception as exc:
            logger.error("handle_message error: %s", exc, exc_info=True)

    async def _ensure_send_session(self) -> None:
        """Recreate _send_session if it's closed or missing.

        Closes the old session first to avoid connection leaks.
        """
        if self._send_session is not None and not self._send_session.closed:
            try:
                await self._send_session.close()
            except Exception:
                pass
        import aiohttp as _aiohttp
        ssl_ctx = _create_ssl_context()
        self._send_session = _aiohttp.ClientSession(
            trust_env=True,
            connector=_aiohttp.TCPConnector(ssl=ssl_ctx),
        )
        logger.debug("Send session (re)created")

    async def _throttle_send(self) -> None:
        """Ensure at least 200ms gap between consecutive sends to avoid rate limits."""
        now = time.monotonic()
        gap = now - self._last_send_time
        if gap < 0.2:
            await asyncio.sleep(0.2 - gap)
        self._last_send_time = time.monotonic()

    async def _send_text_locked(self, user_id: str, text: str) -> None:
        """Send text — caller MUST hold _send_lock. Internal helper."""
        import uuid
        ctx_token = self._context_tokens.get(user_id, "")
        msg: dict[str, Any] = {
            "from_user_id": "",
            "to_user_id": user_id,
            "client_id": f"terra-{uuid.uuid4().hex}",
            "message_type": 2,
            "message_state": 2,
            "item_list": [{"type": 1, "text_item": {"text": text[:2000]}}],
        }
        if ctx_token:
            msg["context_token"] = ctx_token
        await self._post_msg(msg)

    async def send_message(self, user_id: str, text: str) -> None:
        async with self._send_lock:
            await self._throttle_send()
            try:
                await self._send_text_locked(user_id, text)
            except Exception as exc:
                logger.error("send_message failed: %s", exc)

    async def send_image(self, user_id: str, image_b64: str, caption: str = "") -> None:
        """Send an image via iLink CDN upload (ported from Hermes).

        The image data (base64 JPEG) is encrypted with AES-128-ECB, uploaded
        to the WeChat CDN, then referenced in a native image message.
        Falls back to text-only if the upload or send fails.
        """
        data_len = len(image_b64)
        logger.info("Sending image to %s (data: %d chars ≈ %.0f KB)",
                     user_id[:20], data_len, data_len / 1024)

        # If data is suspiciously large, skip image and just send caption
        if data_len > 500_000:  # > ~350 KB JPEG
            logger.warning("Image too large (%d chars), sending text-only fallback", data_len)
            if caption:
                await self.send_message(user_id, caption)
            return

        async with self._send_lock:
            await self._throttle_send()

            ctx_token = self._context_tokens.get(user_id, "")
            try:
                await _send_media_message(
                    self._send_session,
                    base_url=self.base_url,
                    token=self.token,
                    to_user_id=user_id,
                    image_b64=image_b64,
                    caption=caption,
                    context_token=ctx_token,
                )
                logger.info("Image sent successfully to %s", user_id[:20])
            except Exception as exc:
                logger.warning("Image send failed, falling back to text: %s", exc)
                # Fallback: send caption as text
                if caption:
                    await asyncio.sleep(0.3)
                    try:
                        await self._send_text_locked(user_id, caption)
                    except Exception as exc2:
                        logger.error("Image fallback text also failed: %s", exc2)

    async def _post_msg(self, msg: dict[str, Any]) -> None:
        """Post a message dict to the iLink API with retry + session recovery.

        Retries up to 6 times with exponential backoff (up to 64s).
        Recreates the send session after ret=-2 failures and on every
        odd-numbered retry in case the TCP connection went stale.
        Raises RuntimeError if all retries are exhausted.
        """
        _MAX_ATTEMPTS = 6
        _BASE_BACKOFF = 2.0       # seconds — doubled each attempt
        _RECREATE_SESSION_EVERY = 3  # recreate session after every N attempts

        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                # Periodically recreate the send session to recover from
                # stale TCP connections or server-side routing changes.
                if attempt > 0 and attempt % _RECREATE_SESSION_EVERY == 0:
                    await self._ensure_send_session()

                resp = await _api_post(
                    self._send_session,
                    endpoint=EP_SEND_MESSAGE,
                    payload={"msg": msg},
                    token=self.token,
                    base_url=self.base_url,
                    timeout_ms=30_000,  # uniform timeout — backoff handles queuing
                )
                ret = resp.get("ret")
                errcode = resp.get("errcode")
                if ret == 0 or (ret is None and errcode in (0, None)):
                    return  # Success

                # iLink returned an error code
                last_exc = RuntimeError(
                    f"send error: ret={ret} errcode={errcode} errmsg={resp.get('errmsg', '')}"
                )
                # ret=-2 often means rate-limit or transient — worth retrying
                # with longer backoff to ride out rate-limiting windows.
                if ret == -2:
                    backoff = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "_post_msg ret=-2, retrying in %.1fs (attempt %d/%d)",
                        backoff, attempt + 1, _MAX_ATTEMPTS,
                    )
                    # Recreate session on every other ret=-2 — the server may
                    # have closed the connection during rate-limiting.
                    if attempt % 2 == 1:
                        await self._ensure_send_session()
                    await asyncio.sleep(backoff)
                    continue
                # Other errors: don't retry
                raise last_exc

            except (RuntimeError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    backoff = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "_post_msg attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1, _MAX_ATTEMPTS, exc, backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    raise last_exc  # from outer except

            except Exception as exc:
                # ConnectionError, ClientError etc — always retry
                last_exc = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    backoff = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "_post_msg connection error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, _MAX_ATTEMPTS, exc, backoff,
                    )
                    await self._ensure_send_session()
                    await asyncio.sleep(backoff)
                else:
                    raise RuntimeError(f"_post_msg exhausted retries: {exc}") from exc

        # All retries exhausted
        logger.error("_post_msg failed after %d attempts: %s", _MAX_ATTEMPTS, last_exc)
        if last_exc:
            raise last_exc


# Screenshot handler — called by intent classifier result


# ---- Quick chat handler: lightweight persona-aware response ----

_CHAT_SYSTEM = """你是 Terra，正在通过微信与用户进行轻松对话。这不是在执行任务——只是在闲聊。

{p persona}

回复规则：
- 用 2-4 句温暖简洁的中文回复
- 如果用户表达疲惫/烦躁/负面情绪 → 先共情再鼓励
- 如果用户问候/感谢/赞美 → 开心回应，可以反问是否需要帮忙
- 如果用户问你能做什么 → 简要介绍（可以清体力、管基建、刷材料、定时任务）
- 不要编造信息，但也不需要过分严肃"""


# ---- Schedule create: lightweight LLM time parsing ----

_SCHEDULE_CREATE_SYSTEM = """你是定时任务解析器。从用户消息中提取以下信息并输出 JSON：

{
  "name": "简短名称（如 早间清体力、午间收菜）",
  "schedule_type": "cron 或 interval",
  "schedule_value": "cron表达式 或 间隔字符串",
  "task_description": "要执行什么操作的描述",
  "one_shot": true 或 false
}

重要：服务器在中国，所有 cron 表达式使用北京时间（UTC+8）。用户说的时间就是北京时间，直接映射，不要转换为 UTC。

时间映射规则：
- "每天早上9点" → cron "0 9 * * *", one_shot: false
- "明天早上9点" → 一次性，cron "0 9 <明天日> <明天月> *", one_shot: true
- "每周一早上8点" → cron "0 8 * * 1", one_shot: false
- "每天晚上10点" → cron "0 22 * * *", one_shot: false
- "每隔30分钟" → interval "30m", one_shot: false
- "每2小时" → interval "2h", one_shot: false
- "每1天" → interval "1d", one_shot: false
- "后天下午3点" → 一次性，cron "0 15 <后天日> <后天月> *", one_shot: true
- "3分钟后" → interval "3m", one_shot: true

输出必须是合法 JSON，不要有其他文字。"""


async def _handle_schedule_create(user_id: str, text: str, bot) -> None:
    """Parse a schedule creation request via LLM and persist it."""
    import asyncio as _asyncio
    import json as _json

    loop = _asyncio.get_running_loop()

    from src.utils.llm_json import llm_json_call

    def _call():
        try:
            return llm_json_call(
                system=_SCHEDULE_CREATE_SYSTEM,
                user_text=text,
                max_retries=2,
                max_tokens=400,
            )
        except (ValueError, Exception) as exc:
            logger.warning("Schedule create LLM parse failed: %s", exc)
            return None

    try:
        data = await _asyncio.wait_for(loop.run_in_executor(None, _call), timeout=15.0)
    except _asyncio.TimeoutError:
        if bot:
            await bot.send_message(user_id, "解析定时任务超时，请重试。")
        return
    except Exception as exc:
        logger.warning("Schedule create LLM call failed: %s", exc)
        if bot:
            await bot.send_message(user_id, f"解析失败: {exc}")
        return

    if data is None:
        if bot:
            await bot.send_message(user_id, "未能解析定时任务，请用更具体的描述，如「定时每天早上9点清体力」。")
        return

    name = (data.get("name") or "").strip()
    if not name:
        name = "定时任务"
    schedule_type = data.get("schedule_type", "cron")
    schedule_value = data.get("schedule_value", "")
    task_description = data.get("task_description", text)
    one_shot = data.get("one_shot", False)

    if not schedule_value:
        if bot:
            await bot.send_message(user_id, "未能提取到时间信息，请用更具体的描述。")
        return

    if schedule_type not in ("cron", "interval"):
        if bot:
            await bot.send_message(user_id, f"不支持的调度类型: {schedule_type}")
        return

    # Validate and calculate next run
    from src.scheduler.schedule_db import schedule_db
    from src.scheduler.time_parser import calculate_next_run

    try:
        next_run_ts = calculate_next_run(schedule_type, schedule_value).timestamp()
    except ValueError as e:
        if bot:
            await bot.send_message(user_id, f"时间表达式无效 ({schedule_value}): {e}")
        return

    # Detect game from user's task description
    from src.games.registry import get_game_registry
    detected_game = get_game_registry().detect_game(task_description)

    task_id = schedule_db.create(
        name=name,
        task_payload={"custom_prompt": task_description},
        schedule_type=schedule_type,
        schedule_value=schedule_value,
        description=task_description,
        game=detected_game,
        task_type="custom",
        one_shot=one_shot,
        next_run=next_run_ts,
    )

    import time as _time
    next_run_str = _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(next_run_ts))
    one_shot_label = "（一次性）" if one_shot else "（重复）"
    conf_msg = (
        f"✅ 定时任务已创建:\n"
        f"  ID: #{task_id}\n"
        f"  名称: {name}\n"
        f"  时间: {schedule_value} ({schedule_type})\n"
        f"  类型: {one_shot_label}\n"
        f"  下次执行: {next_run_str}\n"
        f"  任务: {task_description[:100]}"
    )
    if bot:
        await bot.send_message(user_id, conf_msg)
    logger.info("WeChat schedule created: #%d %s for user=%s", task_id, name, user_id)


def _format_schedule_list(tasks: list) -> str:
    """Format scheduled tasks into a readable WeChat message."""
    if not tasks:
        return "📋 暂无定时任务。\n\n发送「定时每天早上9点...」 来创建。"

    lines = ["📋 定时任务列表:"]
    for t in tasks:
        status = "✅" if t["enabled"] else "⏸️"
        next_run = "..."
        if t["next_run"]:
            import time as _time
            next_run = _time.strftime("%m-%d %H:%M", _time.localtime(t["next_run"]))
        one_shot = "一次" if t["one_shot"] else "重复"
        lines.append(
            f"  [{status}] #{t['id']} {t['name']}\n"
            f"     {t['schedule_type']}={t['schedule_value']} | {one_shot} | 下次: {next_run} | 已执行: {t['run_count']}次"
        )

    lines.append("\n操作: 「取消定时任务#编号」 / 「暂停定时任务#编号」 / 「启用定时任务#编号」")
    return "\n".join(lines)


# ---- Quick reply: lightweight LLM call for instant WeChat responses ----

_QUICK_SYSTEM = """你是 Terra，正在通过微信协助用户管理游戏。当前你正在后台执行自动化任务，用户中途发来了消息。

{persona_section}

回复规则：
- 用友好自然的语气开头（如「好的博士！」、「收到～」、「嗯嗯」）
- 如果用户问进度 → 简洁汇报当前操作 + 已执行步数 + 预估剩余时间
- 如果用户发了新指令 → 「收到，处理完当前步骤马上来」
- 如果用户闲聊/问候 → 温暖简短回应
- 如果用户表达了疲惫/烦躁等情绪 → 先共情再回应指令
- 如果用户说停止/算了 → 尊重用户，不要勉强
- 2-4 句话为佳，不要在短的代价下失去温度
- 不需要每条回复都重复规则或说教"""


def _build_quick_context(agent) -> str:
    """Build a one-paragraph summary of the agent's current state for the quick LLM."""
    parts = [f"当前任务: {agent.state.task_description or '(无)'}"]
    parts.append(f"已执行 {agent.state.iteration_count} 次操作")

    for msg in reversed(agent.state.conversation_history):
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = str(block.get("text", "")).strip()
                    if t and len(t) > 5:
                        parts.append(f"最后决定: {t[:200]}")
                        break
            if len(parts) > 2:
                break
        elif isinstance(content, str) and content.strip() and not content.startswith("[系统"):
            parts.append(f"最近状态: {content[:200]}")
            break

    return "\n".join(parts)


async def _quick_reply(text: str, user_id: str, agent, bot) -> None:
    """Fire a lightweight LLM call (no tools, no screenshots) and reply.

    Runs in a thread pool to avoid blocking the asyncio event loop.
    Max 6s timeout — if LLM is slow, send fallback and let main loop handle it.
    """
    import asyncio as _asyncio
    loop = _asyncio.get_running_loop()

    # Send immediate acknowledgment before LLM call
    if bot:
        await bot.send_message(user_id, "收到，稍等…")

    def _call():
        from src.llm.client import acquire_client, release_client, extract_text
        from config.prompts import get_persona
        from config.settings import config as _cfg

        persona = get_persona(agent.state.game)
        system = _QUICK_SYSTEM.replace("{persona_section}", persona)

        client = acquire_client()
        try:
            ctx = _build_quick_context(agent)
            try:
                response = client.chat(
                    system=system,
                    messages=[{"role": "user", "content": f"背景:\n{ctx}\n\n用户的消息: {text}"}],
                    max_tokens=300,
                    temperature=_cfg.llm.chat_temperature,
                )
                return extract_text(response).strip()
            except Exception:
                return None
        finally:
            release_client(client)

    try:
        reply = await _asyncio.wait_for(loop.run_in_executor(None, _call), timeout=6.0)
    except _asyncio.TimeoutError:
        reply = "收到, 我还在处理中, 稍等片刻"
    except Exception:
        return

    if reply and bot:
        try:
            await bot.send_message(user_id, reply)
        except Exception:
            pass


# ---- main entry for CLI ----

def _load_game_slots(device_serials: list[str]) -> list[Any]:
    """从 config 加载 GameSlot 配置。如果没有配置，按设备自动创建。"""
    from src.concierge.slot_router import GameSlot

    cfg = getattr(config, 'game_slots', None)
    if cfg and isinstance(cfg, list) and len(cfg) > 0:
        slots = [GameSlot.from_config(c) for c in cfg]
        # 只保留设备在线的 slot
        online = set(device_serials)
        valid = [s for s in slots if s.device_serial in online]
        if len(valid) < len(slots):
            missing = [s.device_serial for s in slots if s.device_serial not in online]
            logger.warning("GameSlot devices offline: %s", missing)
        return valid

    # 没有配置 → 按设备自动创建（设备池模式）
    # 从 emulator inventory 获取每个设备的游戏安装信息和名称
    from src.device.emulator_inventory import get_emulator_inventory
    inventory = get_emulator_inventory()
    logger.info("No GAME_SLOTS config — auto-creating %d device-only slot(s) from inventory",
               len(device_serials))
    slots = []
    # Track which emulator entries have already contributed shared aliases
    # to avoid duplicates when one emulator has multiple ADB ports online.
    _emu_aliases_used: set[str] = set()  # emu_entry.id
    for s in device_serials:
        port = s.split(":")[-1] if ":" in s else s
        short = s.replace("127.0.0.1:", "")
        # Find matching inventory entry to get emulator name + installed games
        emu_entry = inventory.find_by_serial(s)
        if not emu_entry:
            # Try matching by port
            for entry in inventory.list_all():
                if port in entry.adb_ports:
                    emu_entry = entry
                    break
        # Build label from inventory (e.g. "MuMu 12 主模拟器 16384")
        label = f"{emu_entry.name} {short}" if emu_entry else f"设备 {short}"
        # Do NOT guess the game from installed_games — package list order is
        # alphabetical by package name, meaningless for deciding what's running.
        # slot.game is only set when _delegate_to_agent actually assigns a task.
        initial_game = ""
        # Build aliases: always include device-specific identifiers (serial,
        # short, port). Emulator-level aliases (e.g. "MuMu", "主号") are only
        # included for the first device from that emulator — they're inherently
        # ambiguous when one emulator exposes multiple ADB ports.
        emu_aliases: list[str] = []
        if emu_entry and emu_entry.id not in _emu_aliases_used:
            emu_aliases = emu_entry.aliases
            _emu_aliases_used.add(emu_entry.id)
        aliases = list(dict.fromkeys(
            [s, short, port] + emu_aliases
        ))
        slots.append(GameSlot(
            slot_id=f"dev_{short.replace('.', '_')}",
            label=label,
            aliases=aliases,
            game=initial_game,  # 初始绑定第一个已安装游戏
            device_serial=s,
        ))
    return slots


def _is_fast_path_command(text: str, *, running_games: frozenset[str] | None = None) -> bool:
    """Check if text is a command that should be handled by Concierge fast_path
    even when an agent is running (stop, status, greeting).

    When `running_games` is provided, stop/cancel messages that ALSO mention a
    running game are NOT intercepted — they'll be routed by the scoring system
    instead (e.g. "方舟停止" → Arknights agent).  This prevents Concierge from
    stealing game-specific management commands.
    """
    stripped = text.strip()

    # ── Stop/cancel ──
    cancel_kw = ["停止", "取消", "算了", "别打了", "不要了"]
    if any(k in stripped for k in cancel_kw) and len(stripped) <= 15:
        # If the message mentions a currently-running game, let the scoring
        # router handle it — the user likely wants to stop THAT game's agent.
        if running_games:
            from src.games.registry import get_game_registry
            for game_id in running_games:
                plugin = get_game_registry().get(game_id)
                if plugin:
                    for kw in plugin.manifest.keywords:
                        if kw in stripped:
                            return False  # Let scoring router handle it
        return True

    # ── Status query ──
    status_kw = ["状态", "进度", "在干嘛", "怎么样了", "如何了", "在做什么", "情况", "运行"]
    if any(k in stripped for k in status_kw) and len(stripped) <= 20:
        return True

    # ── Pure greetings ──
    small = stripped.lower().replace(" ", "")
    greetings = ["你好", "在吗", "在不在", "谢谢", "辛苦了", "早", "晚上好", "晚安", "嗨",
                 "hello", "hi", "hey"]
    if small in greetings:
        return True
    for g in ["你好", "谢谢", "辛苦"]:
        if small.startswith(g):
            remainder = small[len(g):]
            if not remainder or all(
                c in "啊呀哦呢啦吧嘛噢哟哎诶哈呵嘻嘿哒嘞噜呐哇！!，,。.~～…、？? " for c in remainder
            ):
                return True

    # ── Emulator management: "再开/另开/新开 模拟器" → concierge handles it
    if re.search(r'(?:再开|另开|新开|重开|多开|启动新)\S{0,3}(?:模拟器|设备|实例)', stripped):
        return True

    return False


# ── Concierge-level command detection ──────────────────────────────────
# These commands require concierge tools (start_emulator, restart_emulator,
# emulator lifecycle) that game agents lack.  They must bypass agent injection
# even when an agent is waiting_for_user.

_CONCIERGE_COMMAND_PATTERNS = [
    # Broad emulator restart: "重开", "再开吧", "你重开呀", "重启一下"
    r'(?:重开|重启|再开)(?:一下|一个|吧|呀|啊|啦|哦|咯|嘛)?(?:\s*(?:模拟器|设备|手机|这个|它))?\s*$',
    # "打开/启动/开启 模拟器/设备"
    r'(?:打开|启动|开启|再开|新开|另开|多开)\s*(?:一下|一个|个)?\s*(?:模拟器|设备|实例|手机|这个)',
    # Direct address to concierge: "你让管家开", "管家帮我重开"
    r'(?:管家|你让管家|让管家|帮我开|帮我重开|帮我重启)',
    # Emulator complaint: "模拟器崩了", "设备挂了", "连不上了"
    r'(?:模拟器|设备|手机|ADB)\s*(?:崩了|挂了|死了|不行了|有问题|连不上|断了)',
    # Standalone restart plea (short messages ending with particles)
    r'^(?:重开|重启|再开|打开|启动)(?:一下|吧|呀|啊|啦|嘛|咯)?\s*$',
]
# Additional patterns for messages that are likely NOT for the game agent:
# batch commands ("两个号都...", "全部刷..."), slash commands.
# These are already handled by other paths in weixin.py; listed here for
# completeness and as a single source of truth.

def _is_concierge_command(text: str) -> bool:
    """Detect if the message needs concierge-level system tools.

    These commands require tools the concierge has but game agents don't:
    start_emulator, restart_emulator, device lifecycle management.

    Returns True when the message should go to MessageRouter instead of
    being injected into a running game agent.
    """
    import re as _re
    stripped = text.strip()
    for pattern in _CONCIERGE_COMMAND_PATTERNS:
        if _re.search(pattern, stripped):
            return True
    # Catch bare "你重开呀" / "再开吧" forms that regex might miss
    # due to trailing particles
    _bare_restart_particles = ["重开", "重启", "再开"]
    for kw in _bare_restart_particles:
        if stripped.startswith(kw) or stripped.endswith(kw):
            remainder = stripped.replace(kw, "").strip()
            if not remainder or all(
                c in "一下个吧呀啊啦哦咯嘛呢噢哟哎诶哈呵嘿哒嘞噜呐哇！!，,。.~～…、？? " for c in remainder
            ):
                return True
    return False


# ── Observation command prefixes ──────────────────────────────────────

_OBS_START_PREFIXES = ("/record ", "/watch ", "观察 ", "录制 ",
                     "/record\n", "/watch\n", "观察\n", "录制\n")
_OBS_START_NOWORDS = {"/record", "/watch", "观察", "录制"}
_OBS_DONE_PREFIXES = ("/done", "完成观察", "结束观察")
_OBS_CANCEL_PREFIXES = ("/stop", "停止观察", "取消观察")


def _parse_observation_command(text: str) -> tuple[str, str] | None:
    """Parse observation commands from user text.

    Returns (command, body) or None if no match.
    command is one of: "start", "stop", "cancel"
    body is the task name/description (empty for cancel).
    """
    t = text.strip()

    # Start command with no body: exact match (e.g. "/record", "/watch", "观察")
    if t in _OBS_START_NOWORDS:
        return ("start", "")

    # Start command with body: prefix match (e.g. "/record 1999日常")
    for prefix in _OBS_START_PREFIXES:
        if t.startswith(prefix):
            body = t[len(prefix):].strip()
            return ("start", body)

    # Done/complete commands
    for prefix in _OBS_DONE_PREFIXES:
        if t.startswith(prefix):
            body = t[len(prefix):].strip()
            return ("stop", body if body else "")

    # Cancel commands
    for prefix in _OBS_CANCEL_PREFIXES:
        if t.startswith(prefix):
            return ("cancel", "")

    return None


def _handle_observation_command(
    text: str,
    user_id: str,
    bot: Any,
    _get_or_create_concierge,
    _sched_engine,
) -> str | None:
    """Handle /record, /done, /stop. Returns reply string or None."""
    parsed = _parse_observation_command(text)
    if parsed is None:
        return None

    command, body = parsed
    concierge = _get_or_create_concierge(user_id)
    reply = concierge.process_observation_command(command, body)
    return reply


def concierge_adapter(device_serials: list[str], bot: Any = None):
    """Create a message handler backed by MessageRouter.

    Each WeChat user gets a persistent MessageRouter instance.
    GameSlots are loaded from config for multi-game/multi-account routing.
    Schedule commands are intercepted before the MessageRouter sees the message.
    """
    from src.scheduler.cron_scheduler import get_engine as _get_sched_engine
    _sched_engine = _get_sched_engine(device_serials=device_serials)
    from src.device.emulator import emulator_manager as _emu_manager

    # ── Load GameSlots from config ──
    _all_slots: list[Any] = _load_game_slots(device_serials)

    # user_id → MessageRouter
    _concierges: dict[str, Any] = {}

    # user_id → UserGameContext (cross-message game context)
    _user_contexts: dict[str, Any] = {}

    # Per-user default device tracking
    _user_devices: dict[str, str] = {}

    # Per-user slot activity tracking for multi-agent message routing
    # _last_active_slot: updated on explicit routing + agent notifications
    _last_active_slot: dict[str, str] = {}
    # _recent_ask_user: (slot_id, timestamp) — used to route quick follow-up
    # messages ("自己看啊", "继续") to the agent that last asked the user,
    # even after the waiting slot was consumed by the first reply.
    _recent_ask_user: dict[str, tuple[str, float]] = {}
    _ASK_USER_FOLLOWUP_WINDOW = 60.0  # seconds
    # _waiting_slots: FIFO queue of slot_ids whose agents are waiting for reply.
    # When two agents call ask_user simultaneously, the first one gets the
    # next user reply — no silent overwrite.
    _waiting_slots: dict[str, list[str]] = {}
    # Lock protecting all _waiting_slots, _recent_ask_user, _last_active_slot access.
    # The _on_slot_activity callback fires from TerraAgent daemon threads while
    # the asyncio event loop reads the same dicts in handler().  This lock
    # serializes all cross-thread access to those three dicts.
    _slot_lock = threading.Lock()

    def _get_free_serial() -> str | None:
        for s in device_serials:
            if not _sched_engine.is_device_busy(s):
                return s
        return None

    def _get_or_create_concierge(user_id: str) -> Any:
        """Get existing MessageRouter for user, or create one."""
        if user_id in _concierges:
            return _concierges[user_id]

        serial = _user_devices.get(user_id)
        if serial is None:
            free = _get_free_serial()
            serial = free if free else (device_serials[0] if device_serials else "emulator-5554")
            _user_devices[user_id] = serial

        concierge = MessageRouter(
            user_id=user_id,
            device_serial=serial,
            bot=bot,
            sched_engine=_sched_engine,
            slots=_all_slots,
            emu_manager=_emu_manager,
        )

        # Register slot activity callback for intelligent multi-agent routing
        def _on_slot_activity(slot_id: str, event_type: str) -> None:
            with _slot_lock:
                if event_type == "ask_user":
                    if user_id not in _waiting_slots:
                        _waiting_slots[user_id] = []
                    if slot_id not in _waiting_slots[user_id]:
                        _waiting_slots[user_id].append(slot_id)
                        logger.debug("[weixin] Slot %s added to waiting queue for %s (now %d)",
                                   slot_id, user_id[:20], len(_waiting_slots[user_id]))
                    import time as _time
                    _recent_ask_user[user_id] = (slot_id, _time.monotonic())
                elif event_type in ("complete", "error"):
                    if user_id in _waiting_slots and slot_id in _waiting_slots[user_id]:
                        _waiting_slots[user_id] = [s for s in _waiting_slots[user_id] if s != slot_id]
                        if not _waiting_slots[user_id]:
                            del _waiting_slots[user_id]
                        logger.debug("[weixin] Slot %s removed from waiting queue for %s (%s)",
                                   slot_id, user_id[:20], event_type)
                _last_active_slot[user_id] = slot_id

        concierge.set_slot_activity_callback(_on_slot_activity)

        _concierges[user_id] = concierge
        slot_info = f" ({len(_all_slots)} slots)" if _all_slots else ""
        logger.info("Created MessageRouter for user=%s on device=%s%s",
                     user_id[:20], serial, slot_info)
        return concierge

    async def handler(text: str, user_id: str) -> dict | None:
        """Message handler (async) — schedule interception + routing."""
        from src.utils.trace import generate_trace_id, set_trace_id
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        def _run_with_trace(fn, *args):
            """Propagate trace_id to executor thread."""
            set_trace_id(trace_id)
            return fn(*args)

        loop = asyncio.get_running_loop()

        # ── Schedule commands (intercept before MessageRouter) ─
        from src.games.registry import get_game_registry
        from src.agent.router import extract_task_id
        schedule_intent = get_game_registry().classify_schedule_intent(text)

        if schedule_intent == "list":
            from src.scheduler.schedule_db import schedule_db
            tasks = schedule_db.get_all()
            reply = _format_schedule_list(tasks)
            if bot:
                await bot.send_message(user_id, reply)
            return None

        if schedule_intent == "delete":
            from src.scheduler.schedule_db import schedule_db
            task_id = extract_task_id(text)
            if task_id is None:
                if bot:
                    await bot.send_message(user_id,
                        "请指定要取消的定时任务编号，如「取消定时任务#3」。\n\n"
                        "发送「查看定时任务」获取列表。")
                return None
            task = schedule_db.get_by_id(task_id)
            if task is None:
                if bot:
                    await bot.send_message(user_id, f"未找到定时任务 #{task_id}。")
                return None
            _sched_engine.cancel_task(task_id)
            schedule_db.delete(task_id)
            if bot:
                await bot.send_message(user_id,
                    f"已删除定时任务 #{task_id} ({task.get('name', '')})")
            return None

        if schedule_intent == "disable":
            from src.scheduler.schedule_db import schedule_db
            task_id = extract_task_id(text)
            if task_id is None:
                if bot:
                    await bot.send_message(user_id,
                        "请指定要暂停的定时任务编号，如「暂停定时任务#3」。")
                return None
            _sched_engine.cancel_task(task_id)
            if schedule_db.set_enabled(task_id, False):
                task = schedule_db.get_by_id(task_id)
                if bot:
                    await bot.send_message(user_id,
                        f"已暂停定时任务 #{task_id} ({task.get('name', '')})")
            else:
                if bot:
                    await bot.send_message(user_id, f"未找到定时任务 #{task_id}。")
            return None

        if schedule_intent == "enable":
            from src.scheduler.schedule_db import schedule_db
            task_id = extract_task_id(text)
            if task_id is None:
                if bot:
                    await bot.send_message(user_id,
                        "请指定要启用的定时任务编号，如「启用定时任务#3」。")
                return None
            if schedule_db.set_enabled(task_id, True):
                task = schedule_db.get_by_id(task_id)
                if bot:
                    await bot.send_message(user_id,
                        f"已启用定时任务 #{task_id} ({task.get('name', '')})")
            else:
                if bot:
                    await bot.send_message(user_id, f"未找到定时任务 #{task_id}。")
            return None

        if schedule_intent == "stop":
            concierge = _get_or_create_concierge(user_id)
            from src.concierge.tools import _cancel_task
            reply = _cancel_task(concierge)
            if bot:
                await bot.send_message(user_id, reply)
            return None

        if schedule_intent == "create":
            asyncio.create_task(_handle_schedule_create(user_id, text, bot))
            return None

        # ── Game switch command (before agent routing) ──
        from src.concierge.game_context import UserGameContext
        ctx = _user_contexts.get(user_id)
        if ctx is None:
            ctx = UserGameContext()
            _user_contexts[user_id] = ctx

        switch_reply = ctx.handle_switch(text)
        if switch_reply is not None:
            # Check if this is a PURE game switch or a switch+task combo.
            # Strip all registered game keywords from the text; if what remains
            # is too short (< 4 chars after stripping), it's a pure switch.
            stripped = text.strip()
            for game_id, kws in ctx._game_keywords().items():
                for kw in sorted(kws, key=len, reverse=True):
                    stripped = stripped.replace(kw, "")
            # Also strip common switch-command words
            for filler in ["切换到", "切换", "到", "完成", "在", "用", "的"]:
                stripped = stripped.replace(filler, "")
            stripped_clean = stripped.strip()
            # If almost nothing left after removing game keywords, it's a pure switch
            _is_pure_switch = len(stripped_clean) <= 2
            if _is_pure_switch:
                if bot:
                    await bot.send_message(user_id, switch_reply)
                return None
            # Message contains both game mention AND task content.
            if bot:
                await bot.send_message(user_id, switch_reply)
            logger.info("Game switch + task in same message — continuing to task routing")

        # ── Observation commands (/record, /done, /stop) ──
        obs_reply = _handle_observation_command(text, user_id, bot,
                                                _get_or_create_concierge, _sched_engine)
        if obs_reply is not None:
            return obs_reply

        # ── Everything else: route to agent or MessageRouter ──
        concierge = _get_or_create_concierge(user_id)

        # Update active game context for the concierge
        concierge.game_ctx.active_game = ctx.active_game

        # ── Find running agents across all slots ──
        running_list: list[tuple[Any, Any]] = []  # [(agent, slot_or_None), ...]
        for s in concierge._slots:
            handle = s.current_task
            if handle is not None:
                a = handle.agent if hasattr(handle, 'agent') else handle
                if a is not None and a.state.running:
                    running_list.append((a, s))
        # Fallback: session-level agent (Phase 1: no GameSlot config)
        if not running_list:
            a = concierge.session.current_agent
            if a is not None and a.state.running:
                running_list.append((a, None))
        is_running = len(running_list) > 0

        # When at least one agent is running, user messages are for the task agent —
        # unless they match fast-path keywords (stop/status/greeting) which
        # should be handled by the router instead of injected to the agent.
        if is_running:
            import re
            # ── Concierge-level commands → MessageRouter (before any agent routing) ──
            # These require concierge tools (start_emulator, restart_emulator, etc.)
            # that game agents lack.  Even if an agent is waiting_for_user, a
            # concierge command must bypass injection so the concierge can handle it.
            if _is_concierge_command(text):
                reply = await asyncio.wait_for(
                    loop.run_in_executor(None, _run_with_trace, concierge.process_message, text),
                    timeout=45.0,
                )
                if reply and bot:
                    await bot.send_message(user_id, reply)
                return None
            # #N management commands → concierge
            if re.match(r'#\d+', text):
                reply = await asyncio.wait_for(
                    loop.run_in_executor(None, _run_with_trace, concierge.process_message, text),
                    timeout=45.0,
                )
                if reply and bot:
                    await bot.send_message(user_id, reply)
                return None
            # Fast-path keywords → concierge (stop/status/greeting).
            # Pass running_games so game+stop combos (e.g. "方舟停止")
            # bypass router and go through scoring-based routing.
            _running_game_set: frozenset[str] = frozenset(
                s.game for _, s in running_list if s and s.game
            )
            if _is_fast_path_command(text, running_games=_running_game_set):
                reply = await asyncio.wait_for(
                    loop.run_in_executor(None, _run_with_trace, concierge.process_message, text),
                    timeout=45.0,
                )
                if reply and bot:
                    await bot.send_message(user_id, reply)
                return None

            # ── Route to the right agent ──
            if len(running_list) == 1:
                agent_target, slot_target = running_list[0]
                running_game = slot_target.game if slot_target else ""
                # If user mentions a game that is NOT the running agent's game,
                # treat as a new task delegation instead of injecting.
                # Example: "完成方舟日常" while 1999 agent is the only runner.
                if running_game and running_game != ctx.active_game:
                    logger.info(
                        "Single agent running %s but message targets %s — delegating",
                        running_game, ctx.active_game or "?",
                    )
                    reply = await asyncio.wait_for(
                        loop.run_in_executor(None, _run_with_trace, concierge.process_message, text),
                        timeout=45.0,
                    )
                    if reply and bot:
                        await bot.send_message(user_id, reply)
                    return None
                # Same game — inject directly
                agent_target.state.inject_message(text)
                slot_id = slot_target.slot_id if slot_target else None
                if slot_id:
                    with _slot_lock:
                        _last_active_slot[user_id] = slot_id
                logger.info("[weixin] Injected message to %srunning agent: %s",
                           f"[{slot_target.label}] " if slot_target else "", text[:80])
                return None

            # ── Multiple agents: scoring-based routing ──
            # Each candidate (agent, slot) gets a score from independent
            # dimensions.  Highest unambiguous score wins.  Tie or gap too
            # small → ask user to clarify.
            from src.concierge.slot_router import SlotRouter
            router = SlotRouter(
                concierge._slots, active_game=concierge.game_ctx.active_game, validate=False,
            )

            # Build lookup: slot_id → (agent, slot)
            running_by_slot: dict[str, tuple[Any, Any]] = {}
            for a, s in running_list:
                if s:
                    running_by_slot[s.slot_id] = (a, s)

            # ── Pre-compute shared signals (under lock — shared with daemon threads) ──
            detected_game = router._detect_game_in_text(text.lower())
            with _slot_lock:
                waiting_list = list(_waiting_slots.get(user_id, []))  # shallow copy
                last_slot_id = _last_active_slot.get(user_id)
                recent_ask = _recent_ask_user.get(user_id)

            # Drain dead entries from waiting list
            while waiting_list:
                wid = waiting_list[0]
                matched = running_by_slot.get(wid)
                if matched is not None and matched[0].state._waiting_for_user:
                    break
                waiting_list.pop(0)

            # Check for stop/cancel keywords (used in scoring + fast-path override)
            _cancel_set = {"停止", "取消", "算了", "别打了", "不要了"}
            _has_cancel_word = any(kw in text for kw in _cancel_set)

            # ── Scoring function ──
            def _score(agent: Any, slot: Any) -> int:
                s = 0
                sid = slot.slot_id if slot else None
                # waiting_for_reply: +1000 (always wins)
                if sid and waiting_list and sid in waiting_list:
                    s += 1000
                # game keyword match: +500, or +800 if combined with cancel/stop
                if detected_game and slot and slot.game == detected_game:
                    if _has_cancel_word:
                        s += 800  # "方舟停止" — strong signal
                    else:
                        s += 500  # "方舟清体力" — game match
                # recent ask_user follow-up: +400
                # If agent asked user within the last 60s and waiting slot
                # was already consumed, this catches "自己看啊"/"继续" etc.
                if sid and recent_ask is not None:
                    _rau_slot, _rau_ts = recent_ask
                    import time as _time2
                    if _rau_slot == sid and (_time2.monotonic() - _rau_ts) < _ASK_USER_FOLLOWUP_WINDOW:
                        s += 400
                # slot label/alias match: +400
                if slot and slot.match_label(text):
                    s += 400
                # last_active: +200
                if sid and sid == last_slot_id:
                    s += 200
                return s

            scored = sorted(
                [(agent, slot, _score(agent, slot)) for agent, slot in running_list],
                key=lambda x: x[2], reverse=True,
            )
            best_agent, best_slot, best_score = scored[0]
            _, _, second_score = scored[1] if len(scored) > 1 else (None, None, -1)

            # ── Dispatch ──
            # Case A: waiting_for_reply → lock-in, consume the FIFO entry
            if best_score >= 1000 and best_slot:
                sid = best_slot.slot_id
                if sid and waiting_list and sid in waiting_list:
                    waiting_list.remove(sid)
                    with _slot_lock:
                        _waiting_slots[user_id] = waiting_list if waiting_list else None
                        if not waiting_list:
                            _waiting_slots.pop(user_id, None)
                best_agent.state.inject_message(text)
                if sid:
                    with _slot_lock:
                        _last_active_slot[user_id] = sid
                logger.info("[weixin] Routed message to [%s] agent (waiting for reply, score=%d): %s",
                           best_slot.label, best_score, text[:80])
                return None

            # Case B: game+cancel combo → route to matching agent, not router
            if best_score >= 800 and detected_game and _has_cancel_word:
                best_agent.state.inject_message(text)
                if best_slot and best_slot.slot_id:
                    with _slot_lock:
                        _last_active_slot[user_id] = best_slot.slot_id
                logger.info("[weixin] Routed stop/cancel to [%s] agent by game match (score=%d): %s",
                           best_slot.label if best_slot else "?", best_score, text[:80])
                return None

            # Case C: clear winner (≥500 with >200 gap) → route
            if best_score >= 500 and (best_score - second_score) >= 200:
                best_agent.state.inject_message(text)
                if best_slot and best_slot.slot_id:
                    with _slot_lock:
                        _last_active_slot[user_id] = best_slot.slot_id
                logger.info("[weixin] Routed message to [%s] agent (score=%d, gap=%d): %s",
                           best_slot.label if best_slot else "?", best_score,
                           best_score - second_score, text[:80])
                return None

            # Case D: no strong signal at all → ambiguous, ask user
            if best_score < 400:
                running_labels = []
                for a, s in running_list:
                    label = f"[{s.label}]" if s else ""
                    task = (s.current_task.task_description if s and s.current_task
                            else a.state.task_description)[:30]
                    running_labels.append(f"  {label} {task}")
                hint = (
                    f"当前有 {len(running_list)} 个任务在运行：\n"
                    + "\n".join(running_labels)
                    + f"\n\n请指定目标的游戏或编号（如「方舟 停止」「#1 进度」）。"
                )
                if bot:
                    await bot.send_message(user_id, hint)
                logger.info("[weixin] Ambiguous routing (%d agents, best_score=%d) — asking user",
                           len(running_list), best_score)
                return None

            # Case E: best_score 400-499 or close gap — fallback to last_active
            fallback_slot_id = last_slot_id
            if fallback_slot_id and fallback_slot_id in running_by_slot:
                fa, fs = running_by_slot[fallback_slot_id]
                fa.state.inject_message(text)
                logger.info("[weixin] Routed to [%s] agent (last active fallback, score=%d): %s",
                           fs.label if fs else "?", best_score, text[:80])
                return None

            # Case F: last resort — ask
            if bot and len(running_list) >= 2:
                running_labels = []
                for a, s in running_list:
                    label = f"[{s.label}]" if s else ""
                    task = (s.current_task.task_description if s and s.current_task
                            else a.state.task_description)[:30]
                    running_labels.append(f"  {label} {task}")
                hint = (
                    f"当前有 {len(running_list)} 个任务在运行：\n"
                    + "\n".join(running_labels)
                    + f"\n\n请指定目标的游戏或编号。"
                )
                await bot.send_message(user_id, hint)
                return None

            # Absolute last resort: inject to first agent
            best_agent.state.inject_message(
                f"[系统提示] 用户发来消息：{text}\n如果与你的任务无关请忽略。"
            )
            if best_slot and best_slot.slot_id:
                with _slot_lock:
                    _last_active_slot[user_id] = best_slot.slot_id
            logger.info("[weixin] Injected message to first agent (last resort): %s", text[:80])
            return None

        # No agent running — router decides whether to delegate or chat
        try:
            reply = await asyncio.wait_for(
                loop.run_in_executor(None, _run_with_trace, concierge.process_message, text),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            reply = "处理超时，请稍后再试。"
            logger.warning("process_message timeout for user=%s", user_id[:20])
        except Exception as exc:
            logger.error("process_message failed: %s", exc, exc_info=True)
            reply = f"出了点问题: {exc}"

        if reply and bot:
            await bot.send_message(user_id, reply)
        return None

    # ── Wire emulator health events → slot management ──
    # The emulator monitor emits: disconnected, reconnected, pre_restart,
    # shutting_down, post_restart, restart_failed
    def _on_emu_concierge(event_type: str, serial: str) -> None:
        if event_type == "disconnected":
            for c in _concierges.values():
                try:
                    c.on_device_offline(serial)
                except Exception:
                    pass
        elif event_type == "reconnected":
            for c in _concierges.values():
                try:
                    c.on_device_online(serial)
                except Exception:
                    pass
        elif event_type == "post_restart":
            # Restart completed — update slot serials (MuMu 12 changes serial)
            for c in _concierges.values():
                try:
                    c.on_device_restarted(serial)
                except Exception:
                    logger.exception(
                        "on_device_restarted failed for serial=%s", serial)

    _emu_manager.on_health_event(_on_emu_concierge)

    return handler


async def run_bot(data_home: str) -> None:
    """Main entry: login (if needed) then start bot."""
    # Create per-session run log so user can tail it in real time
    from logging import FileHandler
    import logging as _logging
    import time as _time
    log_dir = Path(data_home) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _time.strftime("%Y%m%d_%H%M%S")
    run_log_path = log_dir / f"run_{ts}.log"
    run_log_handler = FileHandler(str(run_log_path), encoding="utf-8")
    root_handlers = _logging.getLogger().handlers
    if root_handlers:
        run_log_handler.setFormatter(root_handlers[0].formatter)
        # Inherit AgentTagFilter from the root handler so per-agent tags
        # appear in the session log too (critical for multi-agent debugging)
        for f in root_handlers[0].filters:
            run_log_handler.addFilter(f)
    _logging.getLogger().addHandler(run_log_handler)
    logger.info("Session log: %s", run_log_path.name)

    from src.container import get_container
    get_container()

    accounts = _list_accounts(data_home)

    if accounts:
        account_id = accounts[0]
        creds = _load_account(data_home, account_id)
        if creds:
            token = creds["token"]
            base_url = creds.get("base_url", ILINK_BASE_URL)
            print(f"使用已保存的账号: {account_id[:12]}...")
        else:
            print("已保存的凭证损坏，请重新登录。")
            result = await qr_login(data_home)
            if not result:
                return
            account_id = result["account_id"]
            token = result["token"]
            base_url = result["base_url"]
    else:
        print("未找到微信账号，请先扫码登录。")
        result = await qr_login(data_home)
        if not result:
            return
        account_id = result["account_id"]
        token = result["token"]
        base_url = result["base_url"]

    # Init agent — only register devices from the configured emulator type.
    # Foreign ADB devices (e.g. LDPlayer when emu_type=mumu) are ignored.
    import traceback, sys

    print("正在初始化 ADB 设备...")
    sys.stdout.flush()

    all_devices: list[str] = []

    try:
        from src.device.emulator import emulator_manager

        online = emulator_manager.list_online
        if online:
            for serial in online:
                try:
                    from src.device.adb import init_adb
                    init_adb(serial)
                    all_devices.append(serial)
                    print(f"  ✅ {serial}")
                except Exception as e:
                    print(f"  ⚠️ {serial}: {e}")
            print(f"Agent 就绪（{len(all_devices)} 个设备可用）。")
        else:
            # No matching emulator devices online — try launching one.
            print(f"未检测到 {config.emulator.type} 模拟器设备，正在尝试启动...")
            new_serial = emulator_manager.launch_emulator()
            if new_serial:
                try:
                    from src.device.adb import init_adb
                    init_adb(new_serial)
                    all_devices.append(new_serial)
                    print(f"  ✅ {new_serial}（已自动启动）")
                    emulator_manager.start_health_monitor(new_serial)
                except Exception as e:
                    print(f"  ⚠️ {new_serial}: {e}")
            else:
                print(f"警告: 无法启动 {config.emulator.type} 模拟器。"
                      f"请手动启动模拟器后重试。")

    except Exception as e:
        print(f"Agent 初始化失败: {e}")
        traceback.print_exc()
        return

    # ── Auto-discover installed games on all devices ──
    if all_devices:
        print("正在扫描设备已安装的游戏...")
        try:
            from src.device.emulator_inventory import get_emulator_inventory
            inventory = get_emulator_inventory()
            discovered = inventory.auto_discover_all(all_devices,
                                                      adb_path=config.adb.path)
            if discovered:
                for serial, games in discovered.items():
                    from src.games.registry import get_game_registry
                    gr = get_game_registry()
                    game_names = ", ".join(gr.get_game_name(g) for g in games)
                    print(f"  {serial}: {game_names}" if game_names else f"  {serial}: （无已注册游戏）")
            else:
                print("  未检测到已注册游戏（检查 adb shell pm list packages 是否可用）")
        except Exception as e:
            print(f"  游戏扫描跳过: {e}")

    bot = WeixinBot(
        account_id=account_id,
        token=token,
        base_url=base_url,
        data_home=data_home,
        handle_message=None,
    )
    bot.handle_message = concierge_adapter(all_devices, bot)

    # Start the schedule engine daemon so timed tasks can execute
    if all_devices:
        from src.scheduler.cron_scheduler import get_engine as _get_engine
        from src.device.emulator import emulator_manager as _emu_mgr
        _sched_engine = _get_engine(device_serials=all_devices)
        _sched_engine.start()
        print(f"定时任务调度器已启动（{len(all_devices)} 个设备）。")

        # Wire emulator lifecycle → scheduler coordination
        def _on_emu_event(event_type: str, serial: str) -> None:
            if event_type == "pre_restart":
                print(f"🔄 模拟器 {serial} 正在重启，调度已暂停。")
                _sched_engine.pause_device(serial)
            elif event_type == "post_restart":
                print(f"✅ 模拟器 {serial} 重启完成，调度已恢复。")
                _sched_engine.resume_device(serial)
            elif event_type == "restart_failed":
                print(f"❌ 模拟器 {serial} 重启失败，调度已恢复。")
                _sched_engine.resume_device(serial)

        _emu_mgr.on_health_event(_on_emu_event)

        # Start health monitoring for all devices
        for serial in all_devices:
            _emu_mgr.start_health_monitor(serial)

        # Show initial memory status
        mem = _emu_mgr.get_emulator_memory_mb()
        if mem:
            from config.settings import config as _cfg
            print(f"模拟器内存: {sum(mem.values())}MB / {_cfg.emulator.memory_limit_mb}MB 上限")
    else:
        print("警告: 没有可用设备，定时任务调度器未启动。")

    print(f"Terra WeChat Bot 已启动 (account={account_id[:12]}...)")
    sys.stdout.flush()
    await bot.start()

    try:
        await bot._poll_loop()
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        await bot.stop()
        _logging.getLogger().removeHandler(run_log_handler)
        run_log_handler.close()
        print(f"已停止。日志: {run_log_path}")
