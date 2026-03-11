#!/usr/bin/env python3
"""
Register or remove the Telegram webhook.

Usage:
  python set_webhook.py <webhook_url>   — set webhook
  python set_webhook.py --delete        — delete webhook (switch to polling)
  python set_webhook.py --info          — show current webhook info
"""

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TOKEN:
    print("ERROR: TELEGRAM_TOKEN is not set.", file=sys.stderr)
    sys.exit(1)

BASE = f"https://api.telegram.org/bot{TOKEN}"


def api(method: str, **payload) -> dict:
    url = f"{BASE}/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        print(f"HTTP {exc.code}: {body}", file=sys.stderr)
        sys.exit(1)


def set_webhook(url: str):
    result = api(
        "setWebhook",
        url=url,
        max_connections=10,  # Lambda concurrency is usually low for a personal bot
        allowed_updates=[  # only request the update types the bot handles
            "message",
            "callback_query",
        ],
        drop_pending_updates=True,  # discard any updates queued while bot was offline
    )
    if result.get("ok"):
        print(f"✅  Webhook set:\n    {url}")
    else:
        print(f"❌  Failed: {result}", file=sys.stderr)
        sys.exit(1)

    info = api("getWebhookInfo")
    _print_info(info)


def delete_webhook():
    result = api("deleteWebhook", drop_pending_updates=True)
    if result.get("ok"):
        print("✅  Webhook deleted (bot is now in polling mode).")
    else:
        print(f"❌  Failed: {result}", file=sys.stderr)
        sys.exit(1)


def show_info():
    info = api("getWebhookInfo")
    _print_info(info)


def _print_info(info: dict):
    r = info.get("result", {})
    print("\nWebhook info:")
    print(f"  URL              : {r.get('url') or '(none)'}")
    print(f"  Pending updates  : {r.get('pending_update_count', 0)}")
    print(f"  Last error       : {r.get('last_error_message', 'none')}")
    print(f"  Max connections  : {r.get('max_connections', '-')}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    arg = sys.argv[1]
    if arg == "--delete":
        delete_webhook()
    elif arg == "--info":
        show_info()
    elif arg.startswith("http"):
        set_webhook(arg)
    else:
        print(f"Unknown argument: {arg}\n{__doc__}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
