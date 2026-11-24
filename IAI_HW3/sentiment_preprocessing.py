import html
import re

import emoji


COMPANY_TOKEN = "[CPY]"
GMAIL_TOKEN = "[GMAIL]"


def preprocess(text, clean=False, url_token=False, distinguish_mentions=False):
    if clean:
        text = html.unescape(text)
        text = re.sub(r"Cancelled Flight(?:led|ed)", "cancelled", text, flags=re.I)
        text = re.sub(r"Late Flightr\b", "later", text, flags=re.I)
    text = re.sub(r"(?:https?://|www\.)\S+", " [URL] " if url_token else " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+%\-]+@gmail\.com(?![\w-]|\.[\w-])",
                  f" {GMAIL_TOKEN} ", text, flags=re.IGNORECASE)
    mentions = r"(?<!\w)@\w+"
    first = re.search(mentions, text) if distinguish_mentions else None
    if first:
        addressed = first.group().casefold()
        text = re.sub(mentions, lambda m: " [CPY] " if m.group().casefold() == addressed
                      else " [OTHER_USER] ", text)
    else:
        text = re.sub(mentions, f" {COMPANY_TOKEN} ", text)
    text = emoji.replace_emoji(
        text, replace=lambda symbol, data: " " + data["en"].strip(":").replace("_", " ") + " ",
    )
    return " ".join(text.split())
