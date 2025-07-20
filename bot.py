import asyncio
import re
import os
import logging
from typing import Optional, Dict, Any
import tempfile
import aiohttp
import subprocess

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram import F
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from googletrans import Translator
import yt_dlp
import faster_whisper

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = os.getenv('BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is required")

# Initialize bot and dispatcher
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Initialize services
translator = Translator()

class TranscriptionStates(StatesGroup):
    waiting_for_video = State()
    processing = State()

# YouTube URL pattern
YOUTUBE_URL_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})'
)

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
    match = YOUTUBE_URL_PATTERN.search(url)
    return match.group(1) if match else None

async def get_youtube_transcript(video_id: str, target_language: str = 'en') -> Optional[str]:
    """Get transcript from YouTube video"""
    try:
        # Try to get transcript in the target language
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to find transcript in target language
        try:
            transcript = transcript_list.find_transcript([target_language])
            transcript_data = transcript.fetch()
        except:
            # Fall back to any available transcript and translate
            transcript = transcript_list.find_transcript(['en'])
            transcript_data = transcript.fetch()
            
        # Combine transcript text
        full_text = ' '.join([entry['text'] for entry in transcript_data])
        
        # Translate if necessary
        if target_language != 'en':
            translated = translator.translate(full_text, dest=target_language)
            return translated.text
            
        return full_text
        
    except Exception as e:
        logger.error(f"Error getting transcript: {e}")
        return None

async def download_audio(video_id: str) -> Optional[str]:
    """Download audio from YouTube video"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'/tmp/{video_id}.%(ext)s',
            'extractaudio': True,
            'audioformat': 'mp3',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            url = f"https://www.youtube.com/watch?v={video_id}"
            ydl.download([url])
            
        # Find the downloaded file
        import glob
        audio_files = glob.glob(f"/tmp/{video_id}.*")
        return audio_files[0] if audio_files else None
        
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None

async def transcribe_audio(audio_file: str, target_language: str = 'en') -> Optional[str]:
    """Transcribe audio using faster-whisper"""
    try:
        model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")
        
        segments, info = model.transcribe(
            audio_file,
            language=target_language if target_language != 'auto' else None
        )
        
        transcript = ' '.join([segment.text for segment in segments])
        return transcript
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

@dp.message(Command("start"))
async def start_handler(message: Message):
    """Handle /start command"""
    welcome_text = """
🎥 **YouTube Transcript Bot**

Send me a YouTube URL and I'll extract or generate a transcript for you!

**Features:**
- Extract existing YouTube transcripts
- Generate transcripts for videos without captions
- Support for multiple languages
- Translation capabilities

**Commands:**
/start - Show this message
/help - Get help information

Just send me a YouTube URL to get started!
    """
    
    await message.answer(welcome_text, parse_mode="Markdown")

@dp.message(Command("help"))
async def help_handler(message: Message):
    """Handle /help command"""
    help_text = """
**How to use this bot:**

1. Send me any YouTube URL
2. I'll try to get the existing transcript first
3. If no transcript exists, I'll generate one using AI
4. You can request translations to different languages

**Supported URL formats:**
- https://www.youtube.com/watch?v=VIDEO_ID
- https://youtu.be/VIDEO_ID
- https://youtube.com/embed/VIDEO_ID

**Commands:**
/start - Welcome message
/help - This help message

Need support? Contact the bot administrator.
    """
    
    await message.answer(help_text, parse_mode="Markdown")

@dp.message(F.text)
async def handle_message(message: Message, state: FSMContext):
    """Handle text messages (YouTube URLs)"""
    text = message.text.strip()
    
    # Check if it's a YouTube URL
    video_id = extract_video_id(text)
    if not video_id:
        await message.answer(
            "❌ Please send a valid YouTube URL.\n\n"
            "Example: https://www.youtube.com/watch?v=VIDEO_ID"
        )
        return
    
    # Send processing message
    processing_msg = await message.answer("🔄 Processing your video...")
    
    try:
        # Try to get existing transcript first
        await processing_msg.edit_text("📝 Looking for existing transcript...")
        transcript = await get_youtube_transcript(video_id)
        
        if transcript:
            await processing_msg.edit_text("✅ Found existing transcript!")
            
            # Split long transcripts
            if len(transcript) > 4000:
                chunks = [transcript[i:i+4000] for i in range(0, len(transcript), 4000)]
                await processing_msg.edit_text(f"📄 **Transcript (Part 1/{len(chunks)}):**\n\n{chunks[0]}")
                
                for i, chunk in enumerate(chunks[1:], 2):
                    await message.answer(f"📄 **Transcript (Part {i}/{len(chunks)}):**\n\n{chunk}")
            else:
                await processing_msg.edit_text(f"📄 **Transcript:**\n\n{transcript}")
                
        else:
            # No transcript found, try to generate one
            await processing_msg.edit_text("🎵 No transcript found. Downloading audio...")
            
            audio_file = await download_audio(video_id)
            if not audio_file:
                await processing_msg.edit_text("❌ Failed to download audio from video.")
                return
            
            await processing_msg.edit_text("🤖 Generating transcript with AI...")
            transcript = await transcribe_audio(audio_file)
            
            # Clean up audio file
            try:
                os.remove(audio_file)
            except:
                pass
            
            if transcript:
                await processing_msg.edit_text("✅ Generated transcript!")
                
                # Split long transcripts
                if len(transcript) > 4000:
                    chunks = [transcript[i:i+4000] for i in range(0, len(transcript), 4000)]
                    await processing_msg.edit_text(f"📄 **Generated Transcript (Part 1/{len(chunks)}):**\n\n{chunks[0]}")
                    
                    for i, chunk in enumerate(chunks[1:], 2):
                        await message.answer(f"📄 **Generated Transcript (Part {i}/{len(chunks)}):**\n\n{chunk}")
                else:
                    await processing_msg.edit_text(f"📄 **Generated Transcript:**\n\n{transcript}")
            else:
                await processing_msg.edit_text("❌ Failed to generate transcript. Please try another video.")
                
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        await processing_msg.edit_text("❌ An error occurred while processing the video. Please try again later.")

async def main():
    """Main function to run the bot"""
    logger.info("Starting YouTube Transcript Bot...")
    
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
