from __future__ import annotations

import argparse
import asyncio
import os
import html
from typing import Optional
import logging

from telegram import Bot, Update
from telegram.error import TelegramError
from twilio.rest import Client as TwilioClient

logger = logging.getLogger(__name__)

# Telegram environment variables
_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID")

# Twilio environment variables
_TW_SID: str | None = os.getenv("TWILIO_ACCOUNT_SID")
_TW_TOKEN: str | None = os.getenv("TWILIO_AUTH_TOKEN")
_TW_FROM: str | None = os.getenv("TWILIO_FROM_NUMBER")
_TO_PHONE: str | None = os.getenv("ALERT_PHONE_NUMBER")


def _require_token() -> None:
    if not _TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Get it from @BotFather!")


def _require_twilio() -> None:
    if not all([_TW_SID, _TW_TOKEN, _TW_FROM, _TO_PHONE]):
        raise RuntimeError(
            "Twilio configuration missing. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, and ALERT_PHONE_NUMBER."
        )


async def _async_send_telegram(chat_id: str, text: str) -> None:
    bot = Bot(_TOKEN)
    logger.info("Sending telegram message to %s", chat_id)
    await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")


async def _async_get_chat_id() -> str:
    bot = Bot(_TOKEN)
    updates: tuple[Update] = await bot.get_updates(limit=1)
    if not updates:
        raise RuntimeError(
            "No updates found – send a message to your bot first and try again."
        )
    chat = updates[-1].effective_chat
    if chat is None:
        raise RuntimeError("Could not determine chat id from the last update.")
    return str(chat.id)


def get_chat_id() -> str:
    """Discover the chat‑ID by latest bot update."""
    _require_token()
    try:
        chat_id = asyncio.run(_async_get_chat_id())
        print(f"Your chat‑id is {chat_id}")
        logger.warning("Your chat-id is %s", chat_id)
        return chat_id
    except TelegramError as exc:
        raise RuntimeError(f"Telegram API error while fetching chat id: {exc}") from exc


def send_notification(failed: bool, error: Optional[str] = None, job_name: str = "") -> None:
    """Send a one‑liner status message to Telegram."""
    _require_token()
    chat_id = _CHAT_ID or get_chat_id()  # lazily fetch

    if failed:
        escaped = html.escape(error) if error else ""
        text = f"❌ <b>Job {job_name} failed</b>{':' if escaped else '.'}"
        if escaped:
            # Crop message for telegram api
            if len(escaped) > 3800:
                escaped = escaped[:3800] + "..."
            text += f"\n<pre>{escaped}</pre>"
    else:
        text = f"✅ <b>Job {job_name} completed successfully.</b>"

    try:
        asyncio.run(_async_send_telegram(chat_id, text))
    except TelegramError:
        logger.exception(f"Telegram API error in send_notification!")
    except Exception:
        logger.exception(f"Unexpected error in send_notification!")


def make_call(failed: bool, error: Optional[str] = None, job_name: str = "") -> None:
    """Make a phone call using Twilio with a status message."""
    _require_twilio()
    client = TwilioClient(_TW_SID, _TW_TOKEN)

    if error is not None:
        logger.warning("Error is not used in twilio!")

    if failed:
        body = f"Alert: Job {job_name} failed."
    else:
        body = f"Notification: Job {job_name} completed successfully."

    try:
        call = client.calls.create(
            twiml=f'<Response><Say>{body}</Say></Response>',
            to=_TO_PHONE,
            from_=_TW_FROM,
        )
        logger.info(f"Placed call SID {call.sid}")
    except Exception:
        logger.exception(f"Twilio API error!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send a success/failure notification via Telegram or phone call, or fetch the chat-ID.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chat-id",
        action="store_true",
        help="Print the first chat-ID seen by the bot. No notification is sent.",
    )
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Send a failure notification instead of success.",
    )
    parser.add_argument(
        "--error",
        metavar="TEXT",
        help="Extra error text to include when --fail is used.",
    )
    parser.add_argument(
        "--call",
        action="store_true",
        help="Make a phone call instead of sending a Telegram message.",
    )

    args = parser.parse_args()

    if args.chat_id:
        get_chat_id()
    elif args.call:
        make_call(failed=args.fail, error=args.error, job_name="")
    else:
        send_notification(failed=args.fail, error=args.error)
