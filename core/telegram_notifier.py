import requests
import threading
import json
import os

CONFIG_FILE = "telegram_config.json"

def load_telegram_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"enabled": False, "bot_token": "", "chat_id": ""}

def save_telegram_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def _send_alert_worker(token, chat_id, message, photo_path):
    try:
        if photo_path and os.path.exists(photo_path):
            # Send photo with caption
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            with open(photo_path, "rb") as photo:
                payload = {"chat_id": chat_id, "caption": message}
                files = {"photo": photo}
                response = requests.post(url, data=payload, files=files, timeout=10)
        else:
            # Send text only
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": message}
            response = requests.post(url, json=payload, timeout=10)
            
        if response.status_code != 200:
            print(f"[Telegram] Lỗi gửi cảnh báo: {response.text}")
        else:
            print("[Telegram] Đã gửi cảnh báo thành công!")
    except Exception as e:
        print(f"[Telegram] Exception khi gửi: {e}")

def send_telegram_alert(message, photo_path=None):
    config = load_telegram_config()
    if not config.get("enabled") or not config.get("bot_token") or not config.get("chat_id"):
        return

    # Chạy trên thread riêng để không block UI hoặc xử lý camera
    thread = threading.Thread(
        target=_send_alert_worker, 
        args=(config["bot_token"], config["chat_id"], message, photo_path),
        daemon=True
    )
    thread.start()
