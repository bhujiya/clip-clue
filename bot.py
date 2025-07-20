# Fix for Python 3.13 cgi module issue
try:
    import cgi
except ImportError:
    import sys
    try:
        import legacy_cgi as cgi
        sys.modules['cgi'] = cgi
    except ImportError:
        print("Warning: cgi module not available. Please install legacy-cgi or use Python 3.12")

import asyncio
import re
import os
import json
import aiohttp
import html
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from googletrans import Translator
import yt_dlp
import tempfile
import uuid
import whisper
import torch
from datetime import datetime, date
from pathlib import Path

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# File paths for caching and usage tracking
CACHE_FILE = "transcript_cache.json"
USAGE_FILE = "daily_usage.json"

# Load Whisper model once at startup
print("🤖 Loading Whisper model...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("base", device=device)  # Use "tiny" for faster processing
    print(f"✅ Whisper model loaded successfully on {device.upper()}!")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    whisper_model = None

# Debug print (remove in production)
print(f"OpenAI API Key loaded: {OPENAI_API_KEY[:10]}..." if OPENAI_API_KEY else "No API key found")

# Initialize translator
translator = Translator()

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=MemoryStorage())

# In-memory storage
user_transcripts = {}  # user_id: transcript_text
user_languages = {}   # user_id: lang_code ("en", "hi", ...)

# Supported languages for validation
SUPPORTED_LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French',
    'de': 'German', 'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
    'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic'
}

# Configuration
DAILY_LIMIT_PER_USER = 5  # Free transcriptions per user per day
MAX_VIDEO_DURATION = 1800  # 30 minutes max for free transcription

# Cache management functions
def load_cache():
    """Load cached transcripts"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")
            return {}
    return {}

def save_cache(cache_data):
    """Save transcripts to cache"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Cache save error: {e}")

def get_cached_transcript(video_id):
    """Get transcript from cache"""
    cache = load_cache()
    return cache.get(video_id)

def cache_transcript(video_id, transcript):
    """Cache a transcript"""
    cache = load_cache()
    cache[video_id] = transcript
    save_cache(cache)
    print(f"✅ Cached transcript for video: {video_id}")

# Usage tracking functions
def load_daily_usage():
    """Load today's usage data"""
    today = str(date.today())
    
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, 'r') as f:
                data = json.load(f)
                if data.get('date') == today:
                    return data.get('users', {})
        except Exception as e:
            print(f"Usage load error: {e}")
    return {}

def save_daily_usage(usage_data):
    """Save today's usage data"""
    today = str(date.today())
    data = {
        'date': today,
        'users': usage_data
    }
    
    try:
        with open(USAGE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Usage save error: {e}")

def check_daily_limit(user_id, limit=DAILY_LIMIT_PER_USER):
    """Check if user has exceeded daily limit"""
    usage = load_daily_usage()
    user_count = usage.get(str(user_id), 0)
    return user_count < limit, user_count

def increment_usage(user_id):
    """Increment user's daily usage"""
    usage = load_daily_usage()
    user_count = usage.get(str(user_id), 0)
    usage[str(user_id)] = user_count + 1
    save_daily_usage(usage)
    print(f"📊 User {user_id} usage incremented to {user_count + 1}")

def get_video_duration(video_id):
    """Get actual video duration using yt-dlp"""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            duration = info.get('duration', None)  # Duration in seconds
            
            if duration:
                print(f"Video duration: {duration} seconds ({duration//60}:{duration%60:02d})")
                return duration
            else:
                print("Could not get video duration")
                return None
                
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return None

def download_audio(video_url, output_dir=None):
    """Download audio from YouTube video with improved error handling"""
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    # Generate unique filename to avoid conflicts
    unique_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(output_dir, f"audio_{unique_id}")
    
    try:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': f'{output_path}.%(ext)s',
            'quiet': True,  # Reduce noise
            'no_warnings': True,
            'extractaudio': True,
            'audioformat': 'mp3',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'ignoreerrors': True,
            'no_check_certificate': True,
        }
        
        print(f"📥 Downloading audio from: {video_url}")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
        # Check if the file was created
        mp3_path = f"{output_path}.mp3"
        if os.path.exists(mp3_path):
            file_size = os.path.getsize(mp3_path)
            print(f"✅ Audio downloaded: {mp3_path}, Size: {file_size} bytes")
            return mp3_path
        else:
            # Sometimes yt-dlp creates files with different extensions
            for ext in ['.m4a', '.webm', '.mp4']:
                alt_path = f"{output_path}{ext}"
                if os.path.exists(alt_path):
                    print(f"✅ Audio downloaded with {ext}: {alt_path}")
                    return alt_path
            
            raise Exception("Audio file not found after download")
            
    except Exception as e:
        print(f"❌ Audio download error: {e}")
        # Clean up any partial files
        for ext in ['.mp3', '.m4a', '.webm', '.mp4']:
            file_path = f"{output_path}{ext}"
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        raise Exception(f"Failed to download audio: {str(e)}")

def transcribe_with_local_whisper(audio_path):
    """Transcribe audio using local Whisper model - COMPLETELY FREE"""
    if not whisper_model:
        raise Exception("Whisper model not loaded")
    
    if not os.path.exists(audio_path):
        raise Exception(f"Audio file does not exist: {audio_path}")
    
    file_size = os.path.getsize(audio_path)
    print(f"🎵 Transcribing locally: {audio_path} ({file_size} bytes)")
    
    if file_size < 1000:  # Less than 1KB
        raise Exception("Audio file appears to be too small or corrupted")
    
    try:
        # Use local Whisper model with improved settings
        result = whisper_model.transcribe(
            audio_path, 
            word_timestamps=False,  # Disable word-level timestamps for now
            verbose=False
        )
        transcript_text = result["text"].strip()
        
        print(f"✅ Local Whisper transcription successful: {len(transcript_text)} characters")
        return transcript_text
        
    except Exception as e:
        print(f"❌ Local Whisper error: {e}")
        raise Exception(f"Local Whisper transcription failed: {str(e)}")

# OpenAI API call function using aiohttp
async def call_openai_api(messages, max_tokens=500):
    """OpenAI API call using aiohttp for Python 3.13 compatibility"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post("https://api.openai.com/v1/chat/completions",
                                  headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    print(f"OpenAI API error {response.status}: {error_text}")
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
    except asyncio.TimeoutError:
        raise Exception("OpenAI API request timed out")
    except Exception as e:
        print(f"OpenAI API call error: {e}")
        raise Exception(f"Failed to call OpenAI API: {str(e)}")

@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer(
        "👋 Welcome to ClipClue! 🎥\n\n"
        "🆓 <b>Completely FREE YouTube transcript bot!</b>\n\n"
        "📝 Send a YouTube video link and I'll give you:\n"
        "   • Full transcript with timestamps\n"
        "   • AI-powered summary\n"
        "   • Answer questions about the content\n\n"
        "🌍 Set language: /setlang hi (for Hindi), /setlang es (Spanish), etc.\n\n"
        f"📊 Daily limit: <b>{DAILY_LIMIT_PER_USER} transcriptions per day</b> to keep it free for everyone!\n\n"
        "🎯 Supported languages: en, hi, es, fr, de, it, pt, ru, ja, ko, zh, ar"
    )

@dp.message(Command("setlang"))
async def set_language(message: Message):
    parts = message.text.strip().split()
    if len(parts) == 2:
        lang_code = parts[1].lower()
        if lang_code in SUPPORTED_LANGUAGES:
            user_languages[message.from_user.id] = lang_code
            await message.answer(f"✅ Your preferred language is set to <b>{SUPPORTED_LANGUAGES[lang_code]} ({lang_code})</b>.")
        else:
            await message.answer(
                f"❌ Unsupported language code: <b>{lang_code}</b>\n"
                f"Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )
    else:
        await message.answer(
            "Usage: /setlang en (for English) or /setlang hi (for Hindi), etc.\n"
            f"Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}"
        )

@dp.message(Command("usage"))
async def usage_handler(message: Message):
    """Show user's daily usage"""
    user_id = message.from_user.id
    can_use, current_usage = check_daily_limit(user_id)
    remaining = DAILY_LIMIT_PER_USER - current_usage
    
    await message.answer(
        f"📊 <b>Your Daily Usage</b>\n\n"
        f"Used today: <b>{current_usage}/{DAILY_LIMIT_PER_USER}</b>\n"
        f"Remaining: <b>{remaining}</b>\n\n"
        f"Usage resets daily at midnight UTC 🌍"
    )

@dp.message(Command("debug"))
async def debug_transcript(message: Message):
    """Debug command to see stored transcript"""
    user_id = message.from_user.id
    transcript_text = user_transcripts.get(user_id)
    
    if transcript_text:
        # Show first 300 characters of stored transcript
        preview = transcript_text[:300] + "..." if len(transcript_text) > 300 else transcript_text
        await message.answer(f"📝 <b>Stored transcript preview:</b>\n<pre>{html.escape(preview)}</pre>\n\n<b>Length:</b> {len(transcript_text)} characters")
    else:
        await message.answer("❌ No transcript stored for your account.")

def extract_youtube_id(link: str) -> str | None:
    """Extract YouTube video ID from various YouTube URL formats"""
    if not link:
        return None
    
    link = link.strip()
    
    # Handle youtu.be short links
    short_match = re.match(r'(https?://)?(www\.)?youtu\.be/([a-zA-Z0-9_-]{11})', link)
    if short_match:
        return short_match.group(3)
    
    # Handle youtube.com/watch links
    parsed_url = urlparse(link)
    if 'youtube.com' in parsed_url.netloc and parsed_url.path == '/watch':
        query = parse_qs(parsed_url.query)
        video_ids = query.get('v')
        if video_ids and len(video_ids[0]) == 11:
            return video_ids[0]
    
    # Handle youtube.com/embed/ links
    embed_match = re.match(r'(https?://)?(www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})', link)
    if embed_match:
        return embed_match.group(3)
    
    # Fallback pattern matching
    fallback_match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})(?:\S+)?', link)
    if fallback_match:
        return fallback_match.group(1)
    
    return None

def chunk_transcript(transcript, max_chars=10000, video_duration=None):
    """Split transcript into chunks for processing large videos with proper timestamp validation"""
    chunks = []
    current_chunk = ""
    current_chunk_entries = []
    
    for entry in transcript:
        # Use the actual start time from the entry (in seconds)
        start_seconds = float(entry['start'])
        
        # Cap the timestamp at video duration if available
        if video_duration and start_seconds > video_duration:
            start_seconds = video_duration
        
        minutes = int(start_seconds // 60)
        seconds = int(start_seconds % 60)
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        line = f"{timestamp} {entry['text']}\n"
        
        if len(current_chunk + line) > max_chars and current_chunk:
            chunks.append((current_chunk, current_chunk_entries))
            current_chunk = line
            current_chunk_entries = [entry]
        else:
            current_chunk += line
            current_chunk_entries.append(entry)
    
    if current_chunk:
        chunks.append((current_chunk, current_chunk_entries))
    
    return chunks

def translate_text(text, target_lang):
    """Translate text to target language using Google Translate"""
    if not text or target_lang == "en":
        return text
    
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return f"[⚠️ Translation failed: {e}]\n{text}"

async def fetch_transcript_and_summarize(video_id, message, target_lang="en", user_id=None):
    """Fetch YouTube transcript and generate AI summary with caching and local Whisper"""
    try:
        # FIRST: Check cache
        cached_transcript = get_cached_transcript(video_id)
        if cached_transcript:
            await message.answer("✨ Found cached transcript! Processing instantly...")
            
            # Generate summary from cached transcript
            await message.answer("🧠 Generating AI summary...")
            
            prompt = f"""
Summarize the following YouTube transcript using bullet points.
Each bullet should include the timestamp in the format [MM:SS], followed by the main point explained in that time section.

Transcript:
{cached_transcript}
"""
            
            try:
                summary = await call_openai_api([
                    {"role": "system", "content": "You are a helpful assistant that summarizes YouTube videos."},
                    {"role": "user", "content": prompt}
                ], max_tokens=500)
                
                summary = translate_text(summary, target_lang)
                await message.answer(f"🧠 <b>AI Summary:</b>\n{summary}")
                
            except Exception as e:
                print(f"Error generating summary from cache: {e}")
                await message.answer(f"❌ Failed to generate AI summary: {e}")
            
            return cached_transcript
        
        # Get video duration
        video_duration = get_video_duration(video_id)
        if video_duration:
            duration_min, duration_sec = divmod(video_duration, 60)
            await message.answer(f"📹 Video duration: {duration_min}:{duration_sec:02d}")
            
            # Check video length limit for free transcription
            if video_duration > MAX_VIDEO_DURATION:
                await message.answer(
                    f"⏰ Video is {duration_min}:{duration_sec:02d} long.\n"
                    f"Free transcription is limited to {MAX_VIDEO_DURATION//60} minutes.\n"
                    f"Please try a shorter video or check if it has YouTube captions."
                )
                return f"❌ Video too long for free transcription: {duration_min}:{duration_sec:02d}"
        
        # Try to get YouTube transcript first
        transcript = None
        transcript_text = ""
        
        try:
            print(f"Attempting to get YouTube transcript for video ID: {video_id}")
            # Try multiple language options
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en', 'en-US', 'hi', 'auto', 'en-GB']
            )
            
            # Build transcript text with validated timestamps
            for entry in transcript:
                start_seconds = float(entry['start'])
                
                # Cap timestamp at actual video duration
                if video_duration and start_seconds > video_duration:
                    start_seconds = video_duration
                
                minutes = int(start_seconds // 60)
                seconds = int(start_seconds % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                transcript_text += f"{timestamp} {entry['text']}\n"
            
            print(f"✅ YouTube transcript retrieved: {len(transcript_text)} characters")
            
        except (TranscriptsDisabled, NoTranscriptFound) as youtube_error:
            print(f"YouTube transcript not available: {youtube_error}")
            
            # Check daily limit before using local Whisper
            can_use, current_usage = check_daily_limit(user_id)
            if not can_use:
                await message.answer(
                    f"📊 You've reached your daily limit of {DAILY_LIMIT_PER_USER} free transcriptions.\n\n"
                    f"Current usage: <b>{current_usage}/{DAILY_LIMIT_PER_USER}</b>\n\n"
                    f"🔄 Your limit resets daily at midnight UTC.\n"
                    f"💡 Try videos with existing YouTube captions in the meantime!"
                )
                return f"❌ Daily limit reached: {current_usage}/{DAILY_LIMIT_PER_USER}"
            
            # Fallback to local Whisper transcription
            audio_path = None
            try:
                await message.answer(
                    f"📝 YouTube transcript not available. Using local Whisper...\n"
                    f"📊 This counts towards your daily limit: {current_usage + 1}/{DAILY_LIMIT_PER_USER}"
                )
                await message.answer("⏳ Downloading audio... this may take a moment...")
                
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Download audio
                audio_path = download_audio(video_url)
                
                await message.answer("🎵 Audio downloaded. Transcribing with local Whisper...")
                
                # Transcribe with local Whisper (FREE!)
                raw_transcript = transcribe_with_local_whisper(audio_path)
                
                if not raw_transcript or len(raw_transcript.strip()) < 10:
                    raise Exception("Whisper returned empty or very short transcript")
                
                # Since Whisper doesn't provide timestamps, create a simple format
                transcript_text = f"[00:00] {raw_transcript}"
                
                # Increment usage counter for Whisper transcription
                increment_usage(user_id)
                
                print(f"✅ Local Whisper transcription successful: {len(transcript_text)} characters")
                
            except Exception as whisper_error:
                print(f"❌ Local Whisper error: {whisper_error}")
                error_msg = (f"❌ Failed to get transcript from both YouTube and Whisper.\n\n"
                           f"🔴 YouTube error: {str(youtube_error)}\n"
                           f"🔴 Whisper error: {str(whisper_error)}\n\n"
                           f"This might happen if:\n"
                           f"• The video has no audio content\n"
                           f"• The video is age-restricted\n"
                           f"• Network connectivity issues\n"
                           f"• File processing errors")
                
                await message.answer(error_msg)
                return error_msg
                
            finally:
                # Clean up audio file
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                        print("🧹 Audio file cleaned up")
                    except Exception as cleanup_error:
                        print(f"Cleanup error: {cleanup_error}")
        
        # Cache the transcript for future use
        if transcript_text:
            cache_transcript(video_id, transcript_text)
        
        # Handle long videos with chunking
        if len(transcript_text) > 12000 and transcript:
            chunks = chunk_transcript(transcript, max_chars=10000, video_duration=video_duration)
            await message.answer(f"📹 Long video detected! Processing in {len(chunks)} parts...")
            
            all_summaries = []
            for i, (chunk_text, chunk_entries) in enumerate(chunks):
                try:
                    start_time = float(chunk_entries[0]['start'])
                    end_time = float(chunk_entries[-1]['start'])
                    
                    # Cap times at video duration
                    if video_duration:
                        start_time = min(start_time, video_duration)
                        end_time = min(end_time, video_duration)
                    
                    start_min, start_sec = divmod(int(start_time), 60)
                    end_min, end_sec = divmod(int(end_time), 60)
                    
                    prompt = f"""
Summarize this part of a YouTube transcript using bullet points.
Each bullet should include the timestamp in the format [MM:SS], followed by the main point.
This is part {i+1} of {len(chunks)} (from {start_min:02d}:{start_sec:02d} to {end_min:02d}:{end_sec:02d}).

Transcript:
{chunk_text}
"""
                    
                    chunk_summary = await call_openai_api([
                        {"role": "system", "content": "You are a helpful assistant that summarizes YouTube videos."},
                        {"role": "user", "content": prompt}
                    ], max_tokens=300)
                    
                    chunk_summary = translate_text(chunk_summary, target_lang)
                    all_summaries.append(
                        f"<b>Part {i+1} ({start_min:02d}:{start_sec:02d}-{end_min:02d}:{end_sec:02d}):</b>\n{chunk_summary}"
                    )
                    
                except Exception as e:
                    print(f"Error processing chunk {i+1}: {e}")
                    all_summaries.append(f"<b>Part {i+1}:</b>\n❌ Failed to summarize this part: {e}")
            
            final_summary = "\n\n".join(all_summaries)
            
            # Send summary in chunks if too long
            if len(final_summary) > 4000:
                await message.answer("🧠 <b>AI Summary:</b> (Sending in parts due to length...)")
                for i in range(0, len(final_summary), 4000):
                    chunk = final_summary[i:i+4000]
                    await message.answer(chunk)
            else:
                await message.answer(f"🧠 <b>AI Summary:</b>\n\n{final_summary}")
        
        else:
            # Handle shorter videos
            await message.answer("🧠 Generating AI summary...")
            
            prompt = f"""
Summarize the following YouTube transcript using bullet points.
Each bullet should include the timestamp in the format [MM:SS], followed by the main point explained in that time section.

Transcript:
{transcript_text}
"""
            
            try:
                summary = await call_openai_api([
                    {"role": "system", "content": "You are a helpful assistant that summarizes YouTube videos."},
                    {"role": "user", "content": prompt}
                ], max_tokens=500)
                
                summary = translate_text(summary, target_lang)
                await message.answer(f"🧠 <b>AI Summary:</b>\n{summary}")
                
            except Exception as e:
                print(f"Error generating summary: {e}")
                await message.answer(f"❌ Failed to generate AI summary: {e}")
        
        return transcript_text
        
    except Exception as e:
        print(f"Unexpected error in fetch_transcript_and_summarize: {e}")
        error_msg = f"❌ Unexpected error occurred: {str(e)}"
        await message.answer(error_msg)
        return error_msg

@dp.message()
async def handle_message(message: Message):
    """Handle all incoming messages - YouTube links or Q&A"""
    user_id = message.from_user.id
    text = message.text.strip() if message.text else ""
    user_lang = user_languages.get(user_id, "en")
    
    # Check if message contains a YouTube link
    video_id = extract_youtube_id(text)
    
    if video_id:
        await message.answer(f"🎯 Video ID extracted: <code>{video_id}</code>\n⏳ Processing transcript and generating summary...")
        
        transcript_text = await fetch_transcript_and_summarize(video_id, message, target_lang=user_lang, user_id=user_id)
        
        if transcript_text.startswith("❌"):
            return  # Error already sent to user
        else:
            # Store CLEAN transcript for Q&A (without HTML escaping)
            user_transcripts[user_id] = transcript_text
            print(f"🗂️ Stored transcript for user {user_id}: {len(transcript_text)} characters")
            
            # Send transcript in chunks if too long
            if len(transcript_text) > 4000:
                await message.answer("📝 <b>Full Transcript:</b> (Due to length, sending in parts...)")
                for i in range(0, len(transcript_text), 4000):
                    chunk = transcript_text[i:i+4000]
                    translated_chunk = translate_text(chunk, user_lang)
                    escaped_chunk = html.escape(translated_chunk)
                    await message.answer(f"<pre>{escaped_chunk}</pre>")
            else:
                translated_transcript = translate_text(transcript_text, user_lang)
                escaped_transcript = html.escape(translated_transcript)
                await message.answer(f"📝 <b>Full Transcript:</b>\n<pre>{escaped_transcript}</pre>")
            
            await message.answer("❓ You can now ask questions about this video's content! Try asking something specific about what was discussed.")
        return
    
    # --- IMPROVED Q&A Mode ---
    transcript_text = user_transcripts.get(user_id)
    if not transcript_text:
        await message.answer(
            "❌ Please send a YouTube video link first. Then you can ask questions about the video's content!\n\n"
            f"💡 You have <b>{DAILY_LIMIT_PER_USER - load_daily_usage().get(str(user_id), 0)}</b> free transcriptions remaining today."
        )
        return
    
    if not text:
        await message.answer("❌ Please send a text message with your question about the video.")
        return
    
    await message.answer("🤔 Analyzing video content to answer your question...")
    
    # Clean and format transcript for better AI processing
    clean_transcript = transcript_text.strip()
    
    # Enhanced Q&A prompt
    qa_prompt = f"""You are a helpful AI assistant analyzing a YouTube video transcript to answer user questions.

TRANSCRIPT CONTENT:
{clean_transcript}

USER QUESTION: {text}

INSTRUCTIONS:
- Answer the question using ONLY the information from the transcript above
- Be specific and detailed in your response
- Include relevant timestamps when they exist in the format [MM:SS]
- If the exact answer isn't in the transcript, explain what related information IS available
- Quote relevant parts of the transcript when helpful
- If truly no related information exists, say so clearly but briefly

Provide a helpful, informative answer based on the transcript content:"""
    
    try:
        answer = await call_openai_api([
            {
                "role": "system", 
                "content": "You are an expert at analyzing transcripts and answering questions accurately. Always base your answers on the provided content and be helpful and specific."
            },
            {
                "role": "user", 
                "content": qa_prompt
            }
        ], max_tokens=800)  # Increased token limit for better answers
        
        # Only translate if user has set a non-English language
        if user_lang != "en":
            answer = translate_text(answer, user_lang)
        
        await message.answer(f"🤖 <b>Answer:</b>\n{answer}")
        
        print(f"✅ Q&A successful for user {user_id}, question: {text[:50]}...")
        
    except Exception as e:
        print(f"Error generating Q&A answer: {e}")
        await message.answer(f"❌ Failed to generate answer: {e}")

# Main function to run the bot
async def main():
    """Start the bot"""
    print("🤖 ClipClue Bot is starting...")
    print("🔑 Checking API credentials...")
    
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN not found!")
        return
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found!")
        return
    
    if not whisper_model:
        print("❌ Whisper model not loaded!")
        return
    
    print("✅ API credentials loaded")
    print("✅ Whisper model ready")
    print("🚀 Starting bot polling...")
    
    try:
        await dp.start_polling(bot)
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
