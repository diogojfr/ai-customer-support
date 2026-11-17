"""Gmail client over IMAP/SMTP with an app password (no Google Cloud project, no OAuth).

Drop-in replacement for the tutorial's app/gmail.py: same class name, same method
names, same helper functions. Messages are plain dicts shaped like the Gmail API
ones (id, threadId, labelIds, snippet, payload.headers), so the poller and the AI
pipeline can keep using get_headers(), get_body(), etc.

Differences worth knowing:
- Message ids are IMAP UIDs inside Gmail's "All Mail" folder (strings), thread ids
  are Gmail's X-GM-THRID values, and labelIds holds label *names*.
- Gmail has no threadId parameter over SMTP. It threads the reply from the
  In-Reply-To / References headers and the Re: subject, which we set.
"""

import imaplib
import re
import smtplib
from email import message_from_bytes, policy
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

from app.config import settings

IMAP_HOST = "imap.gmail.com"
SMTP_HOST = "smtp.gmail.com"
PROCESSED_LABEL = "ai-processed"
ESCALATED_LABEL = "needs-human"

FETCH_ITEMS = "(UID X-GM-THRID X-GM-LABELS BODY.PEEK[])"  # PEEK: don't mark as read


def _ok(result: tuple) -> list:
    status, data = result
    if status != "OK":
        raise RuntimeError(f"IMAP command failed: {data}")
    return data


class Gmail:
    def __init__(self) -> None:
        self.address = settings.gmail_address
        self._password = settings.gmail_app_password.replace(" ", "")
        self.imap = self._connect()
        self.label_ids = self._ensure_labels()

    # --- connection -------------------------------------------------------

    def _connect(self) -> imaplib.IMAP4_SSL:
        imap = imaplib.IMAP4_SSL(IMAP_HOST)
        imap.login(self.address, self._password)
        # Everything (inbox, sent replies, archived mail) lives in "All Mail",
        # so working there lets us read whole threads. The folder name depends on
        # the account language, so look it up by its \All flag.
        imap.select(f'"{self._find_all_mail(imap)}"')
        return imap

    @staticmethod
    def _find_all_mail(imap: imaplib.IMAP4_SSL) -> str:
        for raw in _ok(imap.list()):
            line = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
            match = re.match(r'\(([^)]*)\) "[^"]*" "?([^"]+)"?$', line)
            if match and "\\All" in match.group(1).split():
                return match.group(2)
        return "[Gmail]/All Mail"

    def _ready(self) -> imaplib.IMAP4_SSL:
        """Reconnect if Gmail dropped the idle connection (the poller runs for days)."""
        try:
            self.imap.noop()
        except (imaplib.IMAP4.abort, imaplib.IMAP4.error, OSError):
            self.imap = self._connect()
        return self.imap

    def _ensure_labels(self) -> dict[str, str]:
        # Creating an IMAP folder creates a Gmail label. It fails harmlessly if it exists.
        for name in (PROCESSED_LABEL, ESCALATED_LABEL):
            self.imap.create(f'"{name}"')
        return {PROCESSED_LABEL: PROCESSED_LABEL, ESCALATED_LABEL: ESCALATED_LABEL}

    # --- reading ----------------------------------------------------------

    def list_new_message_ids(self, max_results: int = 10) -> list[str]:
        imap = self._ready()
        data = _ok(
            imap.uid("SEARCH", "X-GM-RAW", f'"in:inbox -label:{PROCESSED_LABEL}"')
        )
        uids = [uid.decode() for uid in (data[0] or b"").split()]
        return list(reversed(uids))[:max_results]  # newest first, like the Gmail API

    def get_message(self, message_id: str) -> dict:
        messages = self._fetch([message_id])
        if not messages:
            raise LookupError(f"Message {message_id} not found")
        return messages[0]

    def get_thread(self, thread_id: str) -> dict:
        imap = self._ready()
        data = _ok(imap.uid("SEARCH", "X-GM-THRID", thread_id))
        uids = (data[0] or b"").decode().split()
        messages = sorted(self._fetch(uids), key=lambda m: int(m["id"]))  # oldest first
        return {"id": thread_id, "messages": messages}

    def _fetch(self, uids: list[str]) -> list[dict]:
        if not uids:
            return []
        imap = self._ready()
        data = _ok(imap.uid("FETCH", ",".join(uids), FETCH_ITEMS))
        messages = []
        for i, item in enumerate(data):
            if not isinstance(item, tuple):
                continue
            meta, raw = item
            trailer = data[i + 1] if i + 1 < len(data) and isinstance(data[i + 1], bytes) else b""
            info = (meta + b" " + trailer).decode(errors="replace")
            uid = re.search(r"UID (\d+)", info)
            thread = re.search(r"X-GM-THRID (\d+)", info)
            labels = re.search(r'X-GM-LABELS \(((?:"[^"]*"|[^)"])*)\)', info)
            label_names = (
                [quoted or bare for quoted, bare in re.findall(r'"([^"]*)"|(\S+)', labels.group(1))]
                if labels
                else []
            )
            messages.append(
                _parse_message(
                    uid.group(1) if uid else "",
                    thread.group(1) if thread else "",
                    label_names,
                    raw,
                )
            )
        return messages

    # --- labels -----------------------------------------------------------

    def add_label(self, message_id: str, name: str) -> None:
        imap = self._ready()
        _ok(imap.uid("STORE", message_id, "+X-GM-LABELS", f'("{name}")'))

    def is_escalated(self, thread: dict) -> bool:
        return any(ESCALATED_LABEL in m.get("labelIds", []) for m in thread["messages"])

    # --- replying ---------------------------------------------------------

    def send_reply(self, message: dict, body: str) -> dict:
        headers = get_headers(message)
        subject = headers.get("subject", "")
        reply = EmailMessage()
        reply["From"] = self.address
        reply["To"] = headers.get("reply-to") or headers["from"]
        reply["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        reply["Date"] = formatdate(localtime=True)
        reply["Message-ID"] = make_msgid(domain=self.address.split("@")[-1])
        reply["In-Reply-To"] = headers["message-id"]
        reply["References"] = f"{headers.get('references', '')} {headers['message-id']}".strip()
        reply.set_content(body)
        with smtplib.SMTP_SSL(SMTP_HOST, 465) as smtp:
            smtp.login(self.address, self._password)
            smtp.send_message(reply)
        # Gmail files the sent copy under the same thread automatically.
        return {"id": reply["Message-ID"], "threadId": message["threadId"]}


def _parse_message(uid: str, thread_id: str, labels: list[str], raw: bytes) -> dict:
    msg = message_from_bytes(raw, policy=policy.default)
    try:
        part = msg.get_body(preferencelist=("plain",))
        text = part.get_content() if part else ""
    except (LookupError, UnicodeDecodeError):
        text = ""
    return {
        "id": uid,
        "threadId": thread_id,
        "labelIds": labels,
        "snippet": " ".join(text.split())[:200],
        "text": text,
        "hasAttachments": any(p.get_filename() for p in msg.walk()),
        "payload": {"headers": [{"name": n, "value": str(v)} for n, v in msg.items()]},
    }


def get_headers(message: dict) -> dict[str, str]:
    return {h["name"].lower(): h["value"] for h in message["payload"].get("headers", [])}


def get_body(message: dict) -> str:
    return message.get("text") or message.get("snippet", "")


def has_attachments(message: dict) -> bool:
    return bool(message.get("hasAttachments"))


def strip_quoted(text: str) -> str:
    """Drop the quoted earlier messages that mail clients add below a reply."""
    lines = []
    for line in text.splitlines():
        if line.startswith("On ") and line.rstrip().endswith("wrote:"):
            break
        if not line.startswith(">"):
            lines.append(line)
    return "\n".join(lines).strip()


def format_thread(thread: dict, own_address: str) -> str:
    """Render a Gmail thread as plain text, oldest message first."""
    first = get_headers(thread["messages"][0])
    parts = [f"Subject: {first.get('subject', '')}"]
    for message in thread["messages"]:
        sender = get_headers(message).get("from", "")
        role = "Harbor & Pine support" if own_address.lower() in sender.lower() else "Customer"
        parts.append(f"{role} ({sender}):\n{strip_quoted(get_body(message))}")
    return "\n\n".join(parts)