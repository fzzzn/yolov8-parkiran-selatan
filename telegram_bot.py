# telegram_bot.py
import os
import requests
import threading
import time
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OWNER_ID = os.getenv("OWNER_TELEGRAM_ID")

# Global callback for manual trigger
manual_trigger_callback = None
last_update_id = None

def send_telegram_photo(image_path: str, caption: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as f:
        r = requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": f},
            timeout=20,
        )
    r.raise_for_status()

def send_telegram_message(message: str, chat_id: str = None) -> None:
    """Send text message to Telegram"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": chat_id or CHAT_ID, "text": message},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        print(f"Error sending message: {e}")

def poll_telegram_updates():
    """Poll for Telegram bot updates in background"""
    global last_update_id, manual_trigger_callback
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    
    while True:
        try:
            params = {"timeout": 30, "offset": last_update_id}
            response = requests.get(url, params=params, timeout=35)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("ok") and data.get("result"):
                    for update in data["result"]:
                        last_update_id = update["update_id"] + 1
                        
                        # Check for /check command
                        if "message" in update and "text" in update["message"]:
                            message = update["message"]
                            text = message["text"]
                            user_id = str(message["from"]["id"])
                            chat_id = str(message["chat"]["id"])
                            
                            if text.startswith("/check"):
                                # Check if user is owner
                                if user_id != OWNER_ID:
                                    send_telegram_message(
                                        "⛔ Unauthorized. Only owner can trigger manual check.",
                                        chat_id
                                    )
                                else:
                                    send_telegram_message(
                                        "🔍 Starting manual parking check...",
                                        chat_id
                                    )
                                    
                                    # Trigger manual detection
                                    if manual_trigger_callback:
                                        threading.Thread(target=manual_trigger_callback).start()
                                    else:
                                        send_telegram_message(
                                            "⚠️ Detection system not ready.",
                                            chat_id
                                        )
            
            time.sleep(1)  # Small delay between polls
            
        except Exception as e:
            print(f"Telegram polling error: {e}")
            time.sleep(10)  # Wait before retry on error

def start_telegram_bot(trigger_callback):
    """Start Telegram bot polling in background thread"""
    global manual_trigger_callback
    manual_trigger_callback = trigger_callback
    
    # Run polling in background thread
    bot_thread = threading.Thread(target=poll_telegram_updates, daemon=True)
    bot_thread.start()
    print("✓ Telegram bot started. Send /check to trigger manual detection.")
