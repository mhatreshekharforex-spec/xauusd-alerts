import os
import requests

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
message = "✅ Test message — your Telegram alert bot is connected and working."

resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=20)

if resp.ok:
    print("Test message sent successfully.")
else:
    print(f"Failed to send: {resp.text}")
