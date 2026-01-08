# telegram_bot.py
import os
import requests
import threading
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OWNER_ID = os.getenv("OWNER_TELEGRAM_ID")

# Global callback for manual trigger
manual_trigger_callback = None

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
    r = requests.post(
        url,
        data={"chat_id": chat_id or CHAT_ID, "text": message},
        timeout=10,
    )
    r.raise_for_status()

async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /check command - manual detection trigger"""
    user_id = str(update.effective_user.id)
    
    # Check if user is owner
    if user_id != OWNER_ID:
        await update.message.reply_text("⛔ Unauthorized. Only owner can trigger manual check.")
        return
    
    await update.message.reply_text("🔍 Starting manual parking check...")
    
    # Trigger manual detection
    if manual_trigger_callback:
        threading.Thread(target=manual_trigger_callback).start()
    else:
        await update.message.reply_text("⚠️ Detection system not ready.")

def start_telegram_bot(trigger_callback):
    """Start Telegram bot in background thread"""
    global manual_trigger_callback
    manual_trigger_callback = trigger_callback
    
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("check", check_command))
    
    # Run bot in a separate thread
    def run_bot():
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    print("✓ Telegram bot started. Send /check to trigger manual detection.")
