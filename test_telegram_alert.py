
import asyncio
import os
from telethon import TelegramClient

API_ID = 30881934
API_HASH = 'f21730701d0b1da80764c094c73effdb'
SESSION_NAME = 'storm_session'

async def main():
    print(f"Connecting to Telegram (ID: {API_ID})...")
    
    # Initialize Client
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    
    await client.connect()
    
    if not await client.is_user_authorized():
        print("❌ NOT AUTHORIZED. You need to log in first by running storm_commander.py interactively.")
        return

    print("✅ ACCOUNTS CONNECTED!")
    print("Sending test message to 'Saved Messages'...")
    
    try:
        await client.send_message('me', "🔔 **STORM SYSTEM TEST**\n\nTelegram bağlantısı başarılı!\nEğer bu mesajı görüyorsanız, 'Kayıp Makale' bildirimleri cebinize gelecektir.\n\n_System Online_")
        print("✅ MESSAGE SENT SUCCESSFULY.")
    except Exception as e:
        print(f"❌ FAILED TO SEND: {e}")

    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
