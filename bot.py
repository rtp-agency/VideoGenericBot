import asyncio
import json
import os
import random
import zipfile
from io import BytesIO
from pathlib import Path
import aiohttp
import fal_client
from PIL import Image
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile, BufferedInputFile
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import subprocess

logger = logging.getLogger("videogenericbot")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(ch)
fh = RotatingFileHandler("videogenericbot.log", maxBytes=5_000_000, backupCount=2, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(fh)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("aiohttp").setLevel(logging.INFO)

# ================== Configuration ==================
# Load environment variables from .env file
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
FAL_KEY = os.getenv("FAL_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set")
if not FAL_KEY:
    raise ValueError("FAL_KEY environment variable not set")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable not set")

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Load Google Drive folder ID
with open("Google_Drive_ID.txt", "r") as f:
    GOOGLE_DRIVE_FOLDER_ID = f.read().strip()

# ================== FSM States ==================
class BotStates(StatesGroup):
    waiting_photos = State()
    choosing_prompt_mode = State()
    entering_single_prompt = State()
    entering_individual_prompts = State()
    choosing_num_images = State()
    generating_images = State()
    waiting_zip_confirmation = State()
    waiting_video_zip = State()
    choosing_num_videos = State()
    entering_video_prompts = State()
    generating_videos = State()
    
    # ElevenLabs Lip-sync states
    choosing_lipsync_mode = State()
    waiting_lipsync_videos = State()
    choosing_voice_profile = State()
    processing_lipsync = State()
    
    # ElevenLabs image to video with voice
    waiting_lipsync_images = State()
    entering_lipsync_prompt = State()
    choosing_lipsync_voice = State()
    generating_lipsync_video = State()
    
    # Add voice profile states
    choosing_add_voice_method = State()
    waiting_voice_audio_files = State()
    entering_voice_name = State()
    entering_voice_description = State()
    entering_voice_generation_prompt = State()
    entering_generated_voice_name = State()

# ================== Bot Initialization ==================
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# ================== Helper Functions ==================
async def download_file(url: str) -> bytes:
    """Download file from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def log_image_details(img_bytes: bytes, filename: str):
    """Log image format, resolution, color mode, and size without modifying the image"""
    try:
        img = Image.open(BytesIO(img_bytes))
        logger.info(
            "Image details: filename=%s, format=%s, size=%dx%d, mode=%s, bytes=%d",
            filename, img.format, img.width, img.height, img.mode, len(img_bytes)
        )
    except Exception as e:
        logger.error("Failed to read image details for %s: %s", filename, str(e))

def create_zip_from_images(image_data_list: list) -> BytesIO:
    """Create ZIP file from list of (filename, bytes) tuples

    Uses writestr to store raw image bytes without re-encoding or modification.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in image_data_list:
            zip_file.writestr(filename, data)  # Raw bytes, no re-encoding
    zip_buffer.seek(0)
    return zip_buffer

def create_zip_from_videos(video_data_list: list) -> BytesIO:
    """Create ZIP file from list of (filename, bytes) tuples for videos"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in video_data_list:
            zip_file.writestr(filename, data)
    zip_buffer.seek(0)
    return zip_buffer

async def extract_images_from_zip(zip_data: bytes) -> list:
    """Extract images from ZIP file, returns list of (filename, bytes)"""
    images = []
    with zipfile.ZipFile(BytesIO(zip_data), 'r') as zip_file:
        for filename in zip_file.namelist():
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                images.append((filename, zip_file.read(filename)))
    return images

def extract_audio_from_video(video_path: str, audio_path: str):
    """Extract audio from video using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'libmp3lame',
            '-y', audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Audio extracted: {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract audio: {e.stderr.decode()}")
        return False

def merge_audio_with_video(video_path: str, audio_path: str, output_path: str):
    """Merge audio with video using ffmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', '-y', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Audio merged with video: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to merge audio: {e.stderr.decode()}")
        return False

async def get_elevenlabs_voices():
    """Get available ElevenLabs voice profiles"""
    try:
        voices = await asyncio.to_thread(elevenlabs_client.voices.get_all)
        return voices.voices
    except Exception as e:
        logger.error(f"Failed to get ElevenLabs voices: {str(e)}")
        return []

async def change_voice_elevenlabs(audio_path: str, voice_id: str, output_path: str):
    """Change voice using ElevenLabs voice changer"""
    try:
        with open(audio_path, 'rb') as audio_file:
            result = await asyncio.to_thread(
                elevenlabs_client.voice_generation.create_voice_from_audio,
                audio=audio_file,
                voice_id=voice_id
            )
        
        # Save the result
        with open(output_path, 'wb') as f:
            f.write(result)
        
        logger.info(f"Voice changed successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to change voice: {str(e)}")
        return False

async def generate_speech_elevenlabs(text: str, voice_id: str, output_path: str):
    """Generate speech using ElevenLabs TTS"""
    try:
        audio = await asyncio.to_thread(
            elevenlabs_client.generate,
            text=text,
            voice=voice_id,
            model="eleven_multilingual_v2"
        )
        
        # Save audio
        with open(output_path, 'wb') as f:
            for chunk in audio:
                f.write(chunk)
        
        logger.info(f"Speech generated: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate speech: {str(e)}")
        return False

async def add_voice_from_samples(name: str, description: str, audio_files: list):
    """Add a new voice profile to ElevenLabs from audio samples (Instant Voice Cloning)"""
    try:
        # audio_files — это список строк с путями к локальным файлам
        voice = await asyncio.to_thread(
            elevenlabs_client.voices.ivc.create,
            name=name,
            description=description or None,  # description опционально
            files=audio_files  # список путей к файлам (str)
        )
        logger.info(f"Voice profile created: {voice.voice_id}")
        return voice
    except Exception as e:
        logger.error(f"Failed to create voice profile: {str(e)}")
        return None

async def generate_voice_from_prompt(text: str, gender: str = "male"):
    """Generate a voice using ElevenLabs voice design"""
    try:
        result = await asyncio.to_thread(
            elevenlabs_client.voice_generation.generate,
            text=text,
            gender=gender
        )
        logger.info("Voice generated from prompt")
        return result
    except Exception as e:
        logger.error(f"Failed to generate voice from prompt: {str(e)}")
        return None

def get_google_drive_credentials():
    """Get Google Drive credentials using OAuth 2.0 with token caching"""
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None

    # Try to load cached token
    if os.path.exists('token.pickle'):
        try:
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        except Exception as e:
            logger.warning(f"Failed to load token.pickle: {e}")

    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            logger.info("Starting OAuth flow - browser will open for authentication...")
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
            logger.info("Credentials saved to token.pickle")

    return creds

async def upload_to_google_drive(file_path: str) -> str:
    """Upload file to Google Drive and return the file URL"""
    try:
        # Get OAuth credentials
        credentials = await asyncio.to_thread(get_google_drive_credentials)

        # Build Drive service
        service = build('drive', 'v3', credentials=credentials)

        # File metadata
        file_metadata = {
            'name': Path(file_path).name,
            'parents': [GOOGLE_DRIVE_FOLDER_ID]
        }

        # Upload file
        media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=True)
        file = await asyncio.to_thread(
            service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute
        )

        logger.info(f"File uploaded to Google Drive: {file.get('webViewLink')}")
        return file.get('webViewLink')
    except Exception as e:
        logger.error(f"Error uploading to Google Drive: {str(e)}")
        raise

def create_number_keyboard(max_num: int, callback_prefix: str) -> InlineKeyboardMarkup:
    """Create inline keyboard with numbers from 1 to max_num in a single row"""
    buttons = [
        InlineKeyboardButton(text=str(i), callback_data=f"{callback_prefix}_{i}")
        for i in range(1, max_num + 1)
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons])

def create_start_mode_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for choosing start mode"""
    buttons = [
        [InlineKeyboardButton(text="📸 Generate images from photos", callback_data="start_mode_photos")],
        [InlineKeyboardButton(text="🎬 Generate video from existing images", callback_data="start_mode_video")],
        [InlineKeyboardButton(text="🎤 Generate Lip-sync", callback_data="start_mode_lipsync")],
        [InlineKeyboardButton(text="➕ Add ElevenLabs voice profile", callback_data="start_mode_add_voice")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_prompt_mode_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for choosing prompt mode"""
    buttons = [
        [InlineKeyboardButton(text="✨ One prompt for ALL", callback_data="prompt_mode_single")],
        [InlineKeyboardButton(text="📝 Individual prompts for EACH", callback_data="prompt_mode_individual")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_confirmation_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for ZIP confirmation"""
    buttons = [
        [InlineKeyboardButton(text="✅ Confirm - looks good!", callback_data="zip_confirm")],
        [InlineKeyboardButton(text="📎 Upload corrected ZIP", callback_data="zip_upload")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_drive_upload_keyboard(video_path: str) -> InlineKeyboardMarkup:
    """Create keyboard for Google Drive upload"""
    buttons = [
        [InlineKeyboardButton(text="📤 Upload to Google Drive", callback_data=f"drive_upload:{video_path}")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_lipsync_mode_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for choosing lip-sync mode"""
    buttons = [
        [InlineKeyboardButton(text="🎬 From Video", callback_data="lipsync_mode_video")],
        [InlineKeyboardButton(text="📸 Create from Image", callback_data="lipsync_mode_image")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_voice_profiles_keyboard(voices: list) -> InlineKeyboardMarkup:
    """Create keyboard with available voice profiles"""
    buttons = []
    for voice in voices[:20]:  # Limit to 20 voices
        buttons.append([InlineKeyboardButton(
            text=f"🎤 {voice.name}",
            callback_data=f"voice_select:{voice.voice_id}"
        )])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def create_add_voice_method_keyboard() -> InlineKeyboardMarkup:
    """Create keyboard for choosing voice addition method"""
    buttons = [
        [InlineKeyboardButton(text="🎙️ Clone from audio samples", callback_data="add_voice_clone")],
        [InlineKeyboardButton(text="✨ Generate from prompt", callback_data="add_voice_generate")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)

# ================== Command Handlers ==================
@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    """Start command - begin the workflow"""
    await state.clear()
    await message.answer(
        "🎬 <b>Welcome to the Video Generation Bot!</b>\n\n"
        "Please choose what you want to do:",
        reply_markup=create_start_mode_keyboard(),
        parse_mode="HTML"
    )

@dp.callback_query(F.data == "start_mode_photos")
async def start_mode_photos(callback: CallbackQuery, state: FSMContext):
    """Start with photo upload and image generation"""
    await callback.message.edit_text(
        "📸 <b>Generate images from photos</b>\n\n"
        "Please send me your source photos.\n"
        "You can send multiple photos.\n\n"
        "When you're done, type /done",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.waiting_photos)
    await state.update_data(source_photos=[])
    await callback.answer()

@dp.callback_query(F.data == "start_mode_video")
async def start_mode_video(callback: CallbackQuery, state: FSMContext):
    """Start with existing images for video generation"""
    await callback.message.edit_text(
        "🎬 <b>Generate videos from existing images</b>\n\n"
        "Please upload your images:\n"
        "• Upload a ZIP file with images, OR\n"
        "• Send individual photos (one or multiple)\n\n"
        "When you're done, type /done",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.waiting_video_zip)
    await state.update_data(video_images=[])
    await callback.answer()

@dp.callback_query(F.data == "start_mode_lipsync")
async def start_mode_lipsync(callback: CallbackQuery, state: FSMContext):
    """Start lip-sync generation mode"""
    await callback.message.edit_text(
        "🎤 <b>Generate Lip-sync</b>\n\n"
        "Choose how you want to create lip-sync video:",
        reply_markup=create_lipsync_mode_keyboard(),
        parse_mode="HTML"
    )
    await callback.answer()

@dp.callback_query(F.data == "start_mode_add_voice")
async def start_mode_add_voice(callback: CallbackQuery, state: FSMContext):
    """Start adding voice profile"""
    await callback.message.edit_text(
        "➕ <b>Add ElevenLabs Voice Profile</b>\n\n"
        "Choose how you want to create a voice profile:",
        reply_markup=create_add_voice_method_keyboard(),
        parse_mode="HTML"
    )
    await state.set_state(BotStates.choosing_add_voice_method)
    await callback.answer()

@dp.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    """Cancel current operation"""
    await state.clear()
    await message.answer("❌ Operation cancelled. Use /start to begin again.")

# ================== Photo Collection ==================
@dp.message(BotStates.waiting_photos, F.photo)
async def receive_photo(message: Message, state: FSMContext):
    """Receive photos from user"""
    data = await state.get_data()
    photos = data.get("source_photos", [])

    # Get the largest photo size
    photo = message.photo[-1]
    photos.append(photo.file_id)

    await state.update_data(source_photos=photos)
    await message.answer(f"✅ Photo {len(photos)} received. Send more or type /done to continue.")

@dp.message(BotStates.waiting_photos, Command("done"))
async def photos_done(message: Message, state: FSMContext):
    """User finished sending photos"""
    data = await state.get_data()
    photos = data.get("source_photos", [])

    if not photos:
        await message.answer("❌ Please send at least one photo first!")
        return

    await message.answer(
        f"📸 Great! I received <b>{len(photos)}</b> photos.\n\n"
        "Now, how would you like to create prompts for image generation?",
        reply_markup=create_prompt_mode_keyboard(),
        parse_mode="HTML"
    )
    await state.set_state(BotStates.choosing_prompt_mode)

# ================== Prompt Mode Selection ==================
@dp.callback_query(BotStates.choosing_prompt_mode, F.data.startswith("prompt_mode_"))
async def choose_prompt_mode(callback: CallbackQuery, state: FSMContext):
    """Handle prompt mode selection"""
    mode = callback.data.split("_")[-1]  # 'single' or 'individual'
    await state.update_data(prompt_mode=mode)

    data = await state.get_data()
    num_photos = len(data.get("source_photos", []))

    if mode == "single":
        await callback.message.edit_text("✨ Perfect! You chose <b>one prompt for all photos</b>.", parse_mode="HTML")
        await callback.message.answer("📝 Please write your prompt for image generation:")
        await state.set_state(BotStates.entering_single_prompt)
    else:
        await callback.message.edit_text(
            f"📝 Perfect! You chose <b>individual prompts</b>.\n\n"
            f"Please send me <b>{num_photos} prompts</b>, one for each photo.\n"
            f"Send them one by one.",
            parse_mode="HTML"
        )
        await state.update_data(individual_prompts=[], current_prompt_index=0)
        await state.set_state(BotStates.entering_individual_prompts)

    await callback.answer()

# ================== Prompt Entry ==================
@dp.message(BotStates.entering_single_prompt, F.text)
async def receive_single_prompt(message: Message, state: FSMContext):
    """Receive single prompt for all photos"""
    await state.update_data(single_prompt=message.text)
    await message.answer(
        "✨ Excellent! Now tell me:\n\n"
        "📸 <b>How many images should I generate per source photo?</b>",
        reply_markup=create_number_keyboard(10, "num_images"),
        parse_mode="HTML"
    )
    await state.set_state(BotStates.choosing_num_images)

@dp.message(BotStates.entering_individual_prompts, F.text)
async def receive_individual_prompt(message: Message, state: FSMContext):
    """Receive individual prompts"""
    data = await state.get_data()
    prompts = data.get("individual_prompts", [])
    prompts.append(message.text)

    num_photos = len(data.get("source_photos", []))
    current_index = len(prompts)

    await state.update_data(individual_prompts=prompts)

    if current_index < num_photos:
        await message.answer(f"✅ Prompt {current_index}/{num_photos} received.\n\n📝 Send prompt for photo #{current_index + 1}:")
    else:
        await message.answer(
            f"✅ All {num_photos} prompts received!\n\n"
            "📸 <b>How many images should I generate per source photo?</b>",
            reply_markup=create_number_keyboard(10, "num_images"),
            parse_mode="HTML"
        )
        await state.set_state(BotStates.choosing_num_images)

# ================== Number of Images Selection ==================
@dp.callback_query(BotStates.choosing_num_images, F.data.startswith("num_images_"))
async def choose_num_images(callback: CallbackQuery, state: FSMContext):
    """Handle number of images selection"""
    num_images = int(callback.data.split("_")[-1])
    await state.update_data(num_images_per_photo=num_images)

    data = await state.get_data()
    prompt_mode = data.get("prompt_mode")

    await callback.message.edit_text(
        f"✅ Got it! I'll generate <b>{num_images}</b> image(s) per source photo.",
        parse_mode="HTML"
    )

    await callback.message.answer("⏳ <b>Generating images...</b>\n\nThis may take a few moments. Please wait...", parse_mode="HTML")
    await callback.answer()

    # Start image generation
    await generate_images(callback.message, state)

# ================== Image Generation ==================
async def generate_images(message: Message, state: FSMContext):
    """Generate images using fal-ai nano-banana-pro with parallel processing"""
    data = await state.get_data()
    source_photos = data.get("source_photos", [])
    num_images = data.get("num_images_per_photo", 1)
    prompt_mode = data.get("prompt_mode")
    single_prompt = data.get("single_prompt")
    individual_prompts = data.get("individual_prompts", [])

    all_generated_images = []
    temp_dir = Path("temp_photos")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Функция для генерации изображений из одного фото
        async def process_single_photo(idx: int, photo_file_id: str):
            # Download photo from Telegram
            file = await bot.get_file(photo_file_id)
            photo_path = temp_dir / f"source_{idx}.jpg"
            await bot.download_file(file.file_path, photo_path)

            # Upload to FAL
            image_url = fal_client.upload_file(str(photo_path))

            # Determine which prompt to use
            if prompt_mode == "single":
                prompt = single_prompt
            else:
                prompt = individual_prompts[idx]

            # Log nano-banana request (truncated)
            request_args = {
                "prompt": prompt,
                "num_images": num_images,
                "image_urls": [image_url],
                "output_format": "png",
                "aspect_ratio": "auto"
            }
            logger.debug("nano-banana request: %s", str(request_args)[:1500])

            result = await asyncio.to_thread(
                fal_client.subscribe,
                "fal-ai/nano-banana/edit",
                arguments=request_args,
                with_logs=False
            )

            # Log nano-banana response info
            logger.debug("nano-banana response: images_count=%d", len(result.get("images", [])))

            # Download generated images with safety check
            photo_images = []
            for img_idx, img in enumerate(result.get("images", [])):
                img_url = img.get("url")
                if img_url:
                    img_data = await download_file(img_url)
                    filename = f"generated_{idx}_{img_idx}.png"
                    log_image_details(img_data, filename)
                    photo_images.append((filename, img_data))
            
            return photo_images

        # Параллельная обработка всех фотографий
        await message.answer(f"🎨 Generating images for {len(source_photos)} photos in parallel...")
        
        tasks = [process_single_photo(idx, photo_id) for idx, photo_id in enumerate(source_photos)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Собираем результаты
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing photo {idx}: {str(result)}")
                await message.answer(f"⚠️ Error processing photo {idx + 1}: {str(result)}")
            else:
                all_generated_images.extend(result)
                await message.answer(f"✅ Photo {idx + 1}/{len(source_photos)} completed ({len(result)} images)")

        if not all_generated_images:
            await message.answer("❌ No images were generated successfully. Please try again.")
            await state.clear()
            return

        # Create ZIP file
        zip_buffer = create_zip_from_images(all_generated_images)

        # Save generated images metadata
        await state.update_data(generated_images=all_generated_images)

        # Send ZIP to user
        await message.answer_document(
            document=BufferedInputFile(zip_buffer.getvalue(), filename="generated_images.zip"),
            caption=(
                "✨ <b>Your generated images are ready!</b>\n\n"
                f"📦 Total images: {len(all_generated_images)}\n\n"
                "Please review the ZIP file.\n"
                "You can confirm or upload a corrected ZIP with removed bad images."
            ),
            parse_mode="HTML",
            reply_markup=create_confirmation_keyboard()
        )

        await state.set_state(BotStates.waiting_zip_confirmation)

    except Exception as e:
        await message.answer(f"❌ <b>Error during generation:</b>\n{str(e)}\n\nPlease try again with /start", parse_mode="HTML")
        await state.clear()
    finally:
        # Cleanup temp files
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()

# ================== ZIP Confirmation ==================
@dp.callback_query(BotStates.waiting_zip_confirmation, F.data == "zip_confirm")
async def confirm_zip(callback: CallbackQuery, state: FSMContext):
    """User confirms the generated images"""
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(
        "🎬 Great! Now let's create videos.\n\n"
        "<b>How many videos should I generate for each image?</b>",
        reply_markup=create_number_keyboard(5, "num_videos"),
        parse_mode="HTML"
    )
    await state.set_state(BotStates.choosing_num_videos)
    await callback.answer()

@dp.callback_query(BotStates.waiting_zip_confirmation, F.data == "zip_upload")
async def request_corrected_zip(callback: CallbackQuery, state: FSMContext):
    """User wants to upload corrected ZIP"""
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(
        "📎 Please upload your corrected ZIP file with the images you want to keep.\n\n"
        "(Only images inside the ZIP will be used for video generation)"
    )
    await callback.answer()

@dp.message(BotStates.waiting_zip_confirmation, F.document)
async def receive_corrected_zip(message: Message, state: FSMContext):
    """Receive corrected ZIP from user"""
    if not message.document.file_name.endswith('.zip'):
        await message.answer("❌ Please upload a ZIP file.")
        return

    # Download ZIP
    file = await bot.get_file(message.document.file_id)
    zip_data = BytesIO()
    await bot.download_file(file.file_path, zip_data)

    # Extract images
    images = await extract_images_from_zip(zip_data.getvalue())

    if not images:
        await message.answer("❌ No images found in ZIP. Please upload a valid ZIP with images.")
        return

    await state.update_data(generated_images=images)
    await message.answer(
        f"✅ Corrected ZIP received! Found <b>{len(images)}</b> images.\n\n"
        "🎬 Now, <b>how many videos should I generate for each image?</b>",
        reply_markup=create_number_keyboard(5, "num_videos"),
        parse_mode="HTML"
    )
    await state.set_state(BotStates.choosing_num_videos)

@dp.message(BotStates.waiting_video_zip, F.photo)
async def receive_video_photo(message: Message, state: FSMContext):
    """Receive individual photos for video generation"""
    data = await state.get_data()
    video_images = data.get("video_images", [])

    # Download photo
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    photo_data = BytesIO()
    await bot.download_file(file.file_path, photo_data)

    # Add to collection
    filename = f"image_{len(video_images)}.jpg"
    video_images.append((filename, photo_data.getvalue()))

    await state.update_data(video_images=video_images)
    await message.answer(f"✅ Photo {len(video_images)} received. Send more or type /done to continue.")

@dp.message(BotStates.waiting_video_zip, F.document)
async def receive_video_zip(message: Message, state: FSMContext):
    """Receive ZIP with images for video generation"""
    if not message.document.file_name.endswith('.zip'):
        await message.answer("❌ Please upload a ZIP file or send photos directly.")
        return

    data = await state.get_data()
    video_images = data.get("video_images", [])

    # Download ZIP
    file = await bot.get_file(message.document.file_id)
    zip_data = BytesIO()
    await bot.download_file(file.file_path, zip_data)

    # Extract images
    images_from_zip = await extract_images_from_zip(zip_data.getvalue())

    if not images_from_zip:
        await message.answer("❌ No images found in ZIP. Please upload a valid ZIP with images.")
        return

    # Merge with existing images
    video_images.extend(images_from_zip)
    await state.update_data(video_images=video_images)
    await message.answer(f"✅ ZIP received! Added {len(images_from_zip)} images. Total: {len(video_images)} images.\n\nSend more or type /done to continue.")

@dp.message(BotStates.waiting_video_zip, Command("done"))
async def video_images_done(message: Message, state: FSMContext):
    """User finished uploading images for video generation"""
    data = await state.get_data()
    video_images = data.get("video_images", [])

    if not video_images:
        await message.answer("❌ Please upload at least one image or ZIP file first!")
        return

    await state.update_data(generated_images=video_images)
    await message.answer(
        f"✅ Great! I have <b>{len(video_images)}</b> images.\n\n"
        "🎬 Now, <b>how many videos should I generate for each image?</b>",
        reply_markup=create_number_keyboard(5, "num_videos"),
        parse_mode="HTML"
    )
    await state.set_state(BotStates.choosing_num_videos)

# ================== Number of Videos Selection ==================
@dp.callback_query(BotStates.choosing_num_videos, F.data.startswith("num_videos_"))
async def choose_num_videos(callback: CallbackQuery, state: FSMContext):
    """Handle number of videos selection"""
    num_videos = int(callback.data.split("_")[-1])
    await state.update_data(num_videos_per_image=num_videos)

    data = await state.get_data()
    num_images = len(data.get("generated_images", []))
    total_videos = num_images * num_videos

    await callback.message.edit_text(
        f"✅ Got it! I'll generate <b>{num_videos}</b> video(s) per image.\n"
        f"Total videos to create: <b>{total_videos}</b>",
        parse_mode="HTML"
    )

    await callback.message.answer(
        f"📝 Now please send me <b>{total_videos} video prompts</b>.\n\n"
        f"Send them one by one, describing the motion/animation you want for each video.",
        parse_mode="HTML"
    )

    await state.update_data(video_prompts=[], current_video_prompt_index=0)
    await state.set_state(BotStates.entering_video_prompts)
    await callback.answer()

# ================== Video Prompts Entry ==================
@dp.message(BotStates.entering_video_prompts, F.text)
async def receive_video_prompt(message: Message, state: FSMContext):
    """Receive video prompts"""
    data = await state.get_data()
    video_prompts = data.get("video_prompts", [])
    video_prompts.append(message.text)

    num_images = len(data.get("generated_images", []))
    num_videos_per_image = data.get("num_videos_per_image", 1)
    total_videos = num_images * num_videos_per_image
    current_index = len(video_prompts)

    await state.update_data(video_prompts=video_prompts)

    if current_index < total_videos:
        await message.answer(f"✅ Prompt {current_index}/{total_videos} received.\n\n📝 Send prompt for video #{current_index + 1}:")
    else:
        await message.answer(
            f"✅ All {total_videos} prompts received!\n\n"
            "⏳ <b>Generating videos...</b>\n\n"
            "This will take several minutes. Please be patient...",
            parse_mode="HTML"
        )
        # Start video generation
        await generate_videos(message, state)

# ================== Video Generation ==================
async def generate_videos(message: Message, state: FSMContext):
    """Generate videos using FAL AI WAN 2.5 with parallel processing"""
    data = await state.get_data()
    generated_images = data.get("generated_images", [])
    num_videos_per_image = data.get("num_videos_per_image", 1)
    video_prompts = data.get("video_prompts", [])

    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)

    temp_videos_dir = Path("temp_videos")
    temp_videos_dir.mkdir(exist_ok=True)

    try:
        # Функция для генерации одного видео
        async def generate_single_video(img_idx: int, filename: str, img_data: bytes, vid_idx: int, prompt: str, prompt_idx: int):
            # Save image temporarily
            img_path = temp_dir / f"{filename}_{vid_idx}"
            with open(img_path, 'wb') as f:
                f.write(img_data)

            # Upload to FAL to get URL
            image_url = fal_client.upload_file(str(img_path))

            # Generate random seed for this video
            video_seed = random.randint(0, 2**31 - 1)

            # Generate video using FAL AI WAN 2.5
            result = await asyncio.to_thread(
                fal_client.subscribe,
                "fal-ai/wan-25-preview/image-to-video",
                arguments={
                    "prompt": prompt,
                    "image_url": image_url,
                    "resolution": "480p",
                    "duration": "5",
                    "negative_prompt": "deformed face, distorted body, extra limbs, missing limbs, mismatched eyes, warped anatomy, AI artifacts, glitch, blur, low resolution, oversharpening, unnatural skin, plastic texture, flickering frames, jitter, unstable motion, unnatural hair movement, exaggerated expressions, incorrect lighting, watermark, text, logo, double face, duplicate features, asymmetrical eyes, bad proportions, cartoonish look, unrealistic body physics, temporal inconsistency, morphing objects, appearing objects, disappearing objects, duplicating limbs, multiplying objects, teleportation, physics violations, impossible movements, shape-shifting, object fusion, floating objects, gravity defects.",
                    "enable_prompt_expansion": False,
                    "enable_safety_checker": False,
                    "seed": video_seed
                },
                with_logs=False
            )

            # Get video URL from result
            video_url = result.get("video", {}).get("url")

            if video_url:
                video_data = await download_file(video_url)

                # Save video to temp file
                video_filename = f"video_{img_idx}_{vid_idx}.mp4"
                video_path = temp_videos_dir / video_filename
                with open(video_path, 'wb') as f:
                    f.write(video_data)

                return (prompt_idx, video_filename, video_data, video_path, prompt, filename)
            
            return None

        # Подготовка всех задач
        tasks = []
        prompt_idx = 0
        
        for img_idx, (filename, img_data) in enumerate(generated_images):
            for vid_idx in range(num_videos_per_image):
                prompt = video_prompts[prompt_idx]
                tasks.append(generate_single_video(img_idx, filename, img_data, vid_idx, prompt, prompt_idx))
                prompt_idx += 1

        # Параллельная генерация видео (батчами по 5 для контроля нагрузки)
        batch_size = 5
        all_results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await message.answer(f"🎬 Generating videos batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}...")
            
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error generating video: {str(result)}")
                elif result:
                    all_results.append(result)

        # Отправка результатов
        for prompt_idx, video_filename, video_data, video_path, prompt, source_img in sorted(all_results, key=lambda x: x[0]):
            await message.answer_video(
                video=BufferedInputFile(video_data, filename=video_filename),
                caption=f"✨ Video {prompt_idx + 1}/{len(video_prompts)}\n\n📝 Prompt: {prompt}",
                reply_markup=create_drive_upload_keyboard(str(video_path))
            )

        await message.answer(
            "🎉 <b>All videos generated successfully!</b>\n\n"
            "Thank you for using the bot. Use /start to create more!",
            parse_mode="HTML"
        )
        await state.clear()

    except Exception as e:
        await message.answer(f"❌ <b>Error during video generation:</b>\n{str(e)}\n\nPlease try again with /start", parse_mode="HTML")
        await state.clear()
    finally:
        # Cleanup temp files
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()

# ================== ElevenLabs Lip-sync: From Video ==================
@dp.callback_query(F.data == "lipsync_mode_video")
async def lipsync_mode_video(callback: CallbackQuery, state: FSMContext):
    """Start lip-sync from video mode"""
    await callback.message.edit_text(
        "🎬 <b>Lip-sync from Video</b>\n\n"
        "Please send me your video(s).\n"
        "You can send multiple videos.\n\n"
        "When you're done, type /done",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.waiting_lipsync_videos)
    await state.update_data(lipsync_videos=[])
    await callback.answer()

@dp.message(BotStates.waiting_lipsync_videos, F.video)
async def receive_lipsync_video(message: Message, state: FSMContext):
    """Receive videos for lip-sync"""
    data = await state.get_data()
    videos = data.get("lipsync_videos", [])
    
    videos.append(message.video.file_id)
    await state.update_data(lipsync_videos=videos)
    await message.answer(f"✅ Video {len(videos)} received. Send more or type /done to continue.")

@dp.message(BotStates.waiting_lipsync_videos, Command("done"))
async def lipsync_videos_done(message: Message, state: FSMContext):
    """User finished sending videos"""
    data = await state.get_data()
    videos = data.get("lipsync_videos", [])

    if not videos:
        await message.answer("❌ Please send at least one video first!")
        return

    # Get available voice profiles
    await message.answer("⏳ Loading available voice profiles...")
    voices = await get_elevenlabs_voices()
    
    if not voices:
        await message.answer("❌ Failed to load voice profiles. Please try again later.")
        await state.clear()
        return

    await message.answer(
        f"🎤 Great! I received <b>{len(videos)}</b> videos.\n\n"
        "Now, choose which voice profile you want to use:",
        reply_markup=create_voice_profiles_keyboard(voices),
        parse_mode="HTML"
    )
    await state.update_data(available_voices=voices)
    await state.set_state(BotStates.choosing_voice_profile)

@dp.callback_query(BotStates.choosing_voice_profile, F.data.startswith("voice_select:"))
async def select_voice_for_lipsync(callback: CallbackQuery, state: FSMContext):
    """User selected a voice profile"""
    voice_id = callback.data.split(":", 1)[1]
    await state.update_data(selected_voice_id=voice_id)
    
    data = await state.get_data()
    voices = data.get("available_voices", [])
    selected_voice = next((v for v in voices if v.voice_id == voice_id), None)
    
    voice_name = selected_voice.name if selected_voice else "Unknown"
    
    await callback.message.edit_text(
        f"✅ Voice profile selected: <b>{voice_name}</b>\n\n"
        "⏳ <b>Processing videos...</b>\n\n"
        "This will take several minutes. Please be patient...",
        parse_mode="HTML"
    )
    await callback.answer()
    
    # Start processing
    await process_lipsync_videos(callback.message, state)

async def process_lipsync_videos(message: Message, state: FSMContext):
    """Process videos with voice change using parallel processing"""
    data = await state.get_data()
    video_ids = data.get("lipsync_videos", [])
    voice_id = data.get("selected_voice_id")
    
    temp_dir = Path("temp_lipsync")
    temp_dir.mkdir(exist_ok=True)
    
    processed_videos = []
    
    try:
        # Функция для обработки одного видео
        async def process_single_video(idx: int, video_id: str):
            # Download video
            file = await bot.get_file(video_id)
            video_path = temp_dir / f"input_{idx}.mp4"
            await bot.download_file(file.file_path, video_path)
            
            # Extract audio
            audio_path = temp_dir / f"audio_{idx}.mp3"
            if not extract_audio_from_video(str(video_path), str(audio_path)):
                return (idx, None, "Failed to extract audio")
            
            # Change voice using ElevenLabs
            changed_audio_path = temp_dir / f"changed_audio_{idx}.mp3"
            
            try:
                with open(audio_path, 'rb') as audio_file:
                    # Using ElevenLabs speech-to-speech
                    result = await asyncio.to_thread(
                        elevenlabs_client.speech_to_speech.convert,
                        audio=audio_file.read(),
                        voice_id=voice_id,
                        model_id="eleven_multilingual_sts_v2"
                    )
                    
                    # Save the result
                    with open(changed_audio_path, 'wb') as f:
                        for chunk in result:
                            f.write(chunk)
                    
                    logger.info(f"Voice changed successfully for video {idx}")
            except Exception as e:
                logger.error(f"Failed to change voice: {str(e)}")
                return (idx, None, f"Failed to change voice: {str(e)}")
            
            # Merge audio back with video
            output_path = temp_dir / f"output_{idx}.mp4"
            if not merge_audio_with_video(str(video_path), str(changed_audio_path), str(output_path)):
                return (idx, None, "Failed to merge audio")
            
            # Read the processed video
            with open(output_path, 'rb') as f:
                video_data = f.read()
            
            return (idx, (f"lipsync_video_{idx}.mp4", video_data, str(output_path)), None)

        # Параллельная обработка видео (батчами по 3 для контроля нагрузки)
        batch_size = 3
        all_results = []
        
        for i in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[i:i + batch_size]
            batch_indices = range(i, min(i + batch_size, len(video_ids)))
            
            await message.answer(f"🎬 Processing videos batch {i//batch_size + 1}/{(len(video_ids) + batch_size - 1)//batch_size}...")
            
            tasks = [process_single_video(idx, video_id) for idx, video_id in zip(batch_indices, batch_ids)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing video: {str(result)}")
                    await message.answer(f"⚠️ Error: {str(result)}")
                else:
                    idx, data, error = result
                    if error:
                        await message.answer(f"⚠️ Video {idx + 1}: {error}")
                    else:
                        all_results.append(data)

        # Отправка результатов
        for filename, video_data, output_path in all_results:
            processed_videos.append((filename, video_data))
            
            await message.answer_video(
                video=BufferedInputFile(video_data, filename=filename),
                caption=f"✨ Processed video",
                reply_markup=create_drive_upload_keyboard(output_path)
            )
        
        # Create ZIP with all videos
        if processed_videos:
            zip_buffer = create_zip_from_videos(processed_videos)
            await message.answer_document(
                document=BufferedInputFile(zip_buffer.getvalue(), filename="lipsync_videos.zip"),
                caption="📦 <b>All processed videos in ZIP</b>",
                parse_mode="HTML"
            )
        
        await message.answer(
            "🎉 <b>All videos processed successfully!</b>\n\n"
            "Thank you for using the bot. Use /start to create more!",
            parse_mode="HTML"
        )
        await state.clear()
        
    except Exception as e:
        await message.answer(f"❌ <b>Error during processing:</b>\n{str(e)}\n\nPlease try again with /start", parse_mode="HTML")
        await state.clear()
    finally:
        # Cleanup
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass

# ================== ElevenLabs Lip-sync: From Image ==================
@dp.callback_query(F.data == "lipsync_mode_image")
async def lipsync_mode_image(callback: CallbackQuery, state: FSMContext):
    """Start lip-sync from image mode"""
    await callback.message.edit_text(
        "📸 <b>Create Lip-sync from Image</b>\n\n"
        "Please send me your image(s).\n"
        "You can send multiple images.\n\n"
        "When you're done, type /done",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.waiting_lipsync_images)
    await state.update_data(lipsync_images=[])
    await callback.answer()

@dp.message(BotStates.waiting_lipsync_images, F.photo)
async def receive_lipsync_image(message: Message, state: FSMContext):
    """Receive images for lip-sync video generation"""
    data = await state.get_data()
    images = data.get("lipsync_images", [])
    
    # Download image
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    photo_data = BytesIO()
    await bot.download_file(file.file_path, photo_data)
    
    filename = f"image_{len(images)}.jpg"
    images.append((filename, photo_data.getvalue()))
    
    await state.update_data(lipsync_images=images)
    await message.answer(f"✅ Image {len(images)} received. Send more or type /done to continue.")

@dp.message(BotStates.waiting_lipsync_images, Command("done"))
async def lipsync_images_done(message: Message, state: FSMContext):
    """User finished sending images"""
    data = await state.get_data()
    images = data.get("lipsync_images", [])

    if not images:
        await message.answer("❌ Please send at least one image first!")
        return

    await message.answer(
        f"✨ Great! I received <b>{len(images)}</b> images.\n\n"
        "📝 Now, please write a prompt describing what should happen in the video:",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.entering_lipsync_prompt)

@dp.message(BotStates.entering_lipsync_prompt, F.text)
async def receive_lipsync_prompt(message: Message, state: FSMContext):
    """Receive video generation prompt"""
    await state.update_data(lipsync_video_prompt=message.text)
    
    # Get available voice profiles
    await message.answer("⏳ Loading available voice profiles...")
    voices = await get_elevenlabs_voices()
    
    if not voices:
        await message.answer("❌ Failed to load voice profiles. Please try again later.")
        await state.clear()
        return

    await message.answer(
        "🎤 Perfect! Now choose which voice profile you want to use:",
        reply_markup=create_voice_profiles_keyboard(voices),
        parse_mode="HTML"
    )
    await state.update_data(available_voices=voices)
    await state.set_state(BotStates.choosing_lipsync_voice)

@dp.callback_query(BotStates.choosing_lipsync_voice, F.data.startswith("voice_select:"))
async def select_voice_for_image_lipsync(callback: CallbackQuery, state: FSMContext):
    """User selected a voice profile for image-to-video"""
    voice_id = callback.data.split(":", 1)[1]
    await state.update_data(selected_voice_id=voice_id)
    
    data = await state.get_data()
    voices = data.get("available_voices", [])
    selected_voice = next((v for v in voices if v.voice_id == voice_id), None)
    
    voice_name = selected_voice.name if selected_voice else "Unknown"
    
    await callback.message.edit_text(
        f"✅ Voice profile selected: <b>{voice_name}</b>\n\n"
        "⏳ <b>Generating videos with voice...</b>\n\n"
        "This will take several minutes. Please be patient...",
        parse_mode="HTML"
    )
    await callback.answer()
    
    # Start processing
    await generate_lipsync_from_images(callback.message, state)

async def generate_lipsync_from_images(message: Message, state: FSMContext):
    """Generate videos from images with voice change using parallel processing"""
    data = await state.get_data()
    images = data.get("lipsync_images", [])
    prompt = data.get("lipsync_video_prompt")
    voice_id = data.get("selected_voice_id")
    
    temp_dir = Path("temp_lipsync_gen")
    temp_dir.mkdir(exist_ok=True)
    
    processed_videos = []
    
    try:
        # Функция для обработки одного изображения
        async def process_single_image(idx: int, filename: str, img_data: bytes):
            # Save image
            img_path = temp_dir / f"{idx}_{filename}"
            with open(img_path, 'wb') as f:
                f.write(img_data)
            
            # Upload to FAL
            image_url = fal_client.upload_file(str(img_path))
            
            # Generate video using WAN 2.5
            video_seed = random.randint(0, 2**31 - 1)
            result = await asyncio.to_thread(
                fal_client.subscribe,
                "fal-ai/wan-25-preview/image-to-video",
                arguments={
                    "prompt": prompt,
                    "image_url": image_url,
                    "resolution": "480p",
                    "duration": "5",
                    "negative_prompt": "deformed face, distorted body, extra limbs, missing limbs, mismatched eyes, warped anatomy, AI artifacts, glitch, blur, low resolution",
                    "enable_prompt_expansion": False,
                    "enable_safety_checker": False,
                    "seed": video_seed
                },
                with_logs=False
            )
            
            video_url = result.get("video", {}).get("url")
            if not video_url:
                return (idx, None, "Failed to generate video")
            
            # Download video
            video_data = await download_file(video_url)
            video_path = temp_dir / f"video_{idx}.mp4"
            with open(video_path, 'wb') as f:
                f.write(video_data)
            
            # Extract audio from generated video
            audio_path = temp_dir / f"audio_{idx}.mp3"
            if not extract_audio_from_video(str(video_path), str(audio_path)):
                return (idx, None, "Failed to extract audio")
            
            # Change voice using ElevenLabs speech-to-speech
            changed_audio_path = temp_dir / f"changed_audio_{idx}.mp3"
            try:
                with open(audio_path, 'rb') as audio_file:
                    result_audio = await asyncio.to_thread(
                        elevenlabs_client.speech_to_speech.convert,
                        audio=audio_file.read(),
                        voice_id=voice_id,
                        model_id="eleven_multilingual_sts_v2"
                    )
                    
                    with open(changed_audio_path, 'wb') as f:
                        for chunk in result_audio:
                            f.write(chunk)
                    
                    logger.info(f"Voice changed successfully for video {idx}")
            except Exception as e:
                logger.error(f"Failed to change voice: {str(e)}")
                return (idx, None, f"Failed to change voice: {str(e)}")
            
            # Merge changed audio with video
            output_path = temp_dir / f"output_{idx}.mp4"
            if not merge_audio_with_video(str(video_path), str(changed_audio_path), str(output_path)):
                return (idx, None, "Failed to merge audio")
            
            # Read processed video
            with open(output_path, 'rb') as f:
                final_video_data = f.read()
            
            return (idx, (f"lipsync_generated_{idx}.mp4", final_video_data, str(output_path)), None)

        # Параллельная обработка изображений (батчами по 3)
        batch_size = 3
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_indices = range(i, min(i + batch_size, len(images)))
            
            await message.answer(f"🎬 Processing images batch {i//batch_size + 1}/{(len(images) + batch_size - 1)//batch_size}...")
            
            tasks = [process_single_image(idx, filename, img_data) for idx, (filename, img_data) in zip(batch_indices, batch_images)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing image: {str(result)}")
                    await message.answer(f"⚠️ Error: {str(result)}")
                else:
                    idx, data, error = result
                    if error:
                        await message.answer(f"⚠️ Image {idx + 1}: {error}")
                    else:
                        all_results.append(data)

        # Отправка результатов
        for filename, video_data, output_path in all_results:
            processed_videos.append((filename, video_data))
            
            await message.answer_video(
                video=BufferedInputFile(video_data, filename=filename),
                caption=f"✨ Video with voice change",
                reply_markup=create_drive_upload_keyboard(output_path)
            )
        
        # Create ZIP
        if processed_videos:
            zip_buffer = create_zip_from_videos(processed_videos)
            await message.answer_document(
                document=BufferedInputFile(zip_buffer.getvalue(), filename="lipsync_generated_videos.zip"),
                caption="📦 <b>All generated videos in ZIP</b>",
                parse_mode="HTML"
            )
        
        await message.answer(
            "🎉 <b>All videos generated successfully!</b>\n\n"
            "Thank you for using the bot. Use /start to create more!",
            parse_mode="HTML"
        )
        await state.clear()
        
    except Exception as e:
        await message.answer(f"❌ <b>Error during generation:</b>\n{str(e)}\n\nPlease try again with /start", parse_mode="HTML")
        await state.clear()
    finally:
        # Cleanup
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass

# ================== Add Voice Profile ==================
@dp.callback_query(BotStates.choosing_add_voice_method, F.data == "add_voice_clone")
async def add_voice_clone(callback: CallbackQuery, state: FSMContext):
    """Clone voice from audio samples"""
    await callback.message.edit_text(
        "🎙️ <b>Clone Voice from Audio Samples</b>\n\n"
        "Please send me audio files with voice samples.\n"
        "Send 1-25 audio files (MP3, WAV, etc.)\n\n"
        "When you're done, type /done",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.waiting_voice_audio_files)
    await state.update_data(voice_audio_files=[])
    await callback.answer()

@dp.message(BotStates.waiting_voice_audio_files, F.audio | F.voice | F.document)
async def receive_voice_audio(message: Message, state: FSMContext):
    """Receive audio files for voice cloning"""
    data = await state.get_data()
    audio_files = data.get("voice_audio_files", [])
    
    # Download audio
    if message.audio:
        file = await bot.get_file(message.audio.file_id)
        filename = message.audio.file_name or f"audio_{len(audio_files)}.mp3"
    elif message.voice:
        file = await bot.get_file(message.voice.file_id)
        filename = f"voice_{len(audio_files)}.ogg"
    elif message.document:
        file = await bot.get_file(message.document.file_id)
        filename = message.document.file_name or f"audio_{len(audio_files)}.mp3"
    else:
        return
    
    audio_data = BytesIO()
    await bot.download_file(file.file_path, audio_data)
    
    audio_files.append((filename, audio_data.getvalue()))
    await state.update_data(voice_audio_files=audio_files)
    await message.answer(f"✅ Audio file {len(audio_files)} received. Send more or type /done to continue.")

@dp.message(BotStates.waiting_voice_audio_files, Command("done"))
async def voice_audio_done(message: Message, state: FSMContext):
    """User finished sending audio files"""
    data = await state.get_data()
    audio_files = data.get("voice_audio_files", [])

    if not audio_files:
        await message.answer("❌ Please send at least one audio file first!")
        return

    await message.answer(
        f"✨ Great! I received <b>{len(audio_files)}</b> audio files.\n\n"
        "📝 Now, please enter a name for this voice profile:",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.entering_voice_name)

@dp.message(BotStates.entering_voice_name, F.text)
async def receive_voice_name(message: Message, state: FSMContext):
    """Receive voice profile name"""
    await state.update_data(voice_name=message.text)
    await message.answer(
        "📝 Perfect! Now please enter a description for this voice profile\n"
        "(optional, or type 'skip'):"
    )
    await state.set_state(BotStates.entering_voice_description)

@dp.message(BotStates.entering_voice_description, F.text)
async def receive_voice_description(message: Message, state: FSMContext):
    """Receive voice profile description and create the voice"""
    description = message.text if message.text.lower() != "skip" else ""
    await state.update_data(voice_description=description)
    
    data = await state.get_data()
    voice_name = data.get("voice_name")
    audio_files = data.get("voice_audio_files", [])
    
    await message.answer("⏳ <b>Creating voice profile...</b>\n\nThis may take a moment...", parse_mode="HTML")
    
    # Save audio files temporarily
    temp_dir = Path("temp_voice_clone")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Сохраняем аудио файлы временно и собираем список путей (str)
        saved_files = []
        for idx, (filename, audio_data) in enumerate(audio_files):
            file_path = temp_dir / f"sample_{idx}{Path(filename).suffix}"  # сохраняем оригинальное расширение
            with open(file_path, 'wb') as f:
                f.write(audio_data)
            saved_files.append(str(file_path))  # список строк-путей
        
        # Создаём голос
        voice = await add_voice_from_samples(voice_name, description, saved_files)
        
        if voice:
            await message.answer(
                f"✅ <b>Voice profile created successfully!</b>\n\n"
                f"🎤 Name: {voice_name}\n"
                f"🆔 Voice ID: {voice.voice_id}\n\n"
                f"You can now use this voice for lip-sync generation!",
                parse_mode="HTML"
            )
        else:
            await message.answer("❌ Failed to create voice profile. Please try again.")
        
        await state.clear()
        
    except Exception as e:
        await message.answer(f"❌ <b>Error creating voice profile:</b>\n{str(e)}", parse_mode="HTML")
        await state.clear()
    finally:
        # Cleanup
        for file in temp_dir.glob("*"):
            try:
                file.unlink()
            except:
                pass
        try:
            temp_dir.rmdir()
        except:
            pass

@dp.callback_query(BotStates.choosing_add_voice_method, F.data == "add_voice_generate")
async def add_voice_generate(callback: CallbackQuery, state: FSMContext):
    """Generate voice from prompt"""
    await callback.message.edit_text(
        "✨ <b>Generate Voice from Prompt</b>\n\n"
        "Please describe the voice you want to generate.\n"
        "For example: 'A young female voice with a British accent, warm and friendly tone'\n\n"
        "📝 Enter your voice description:",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.entering_voice_generation_prompt)
    await callback.answer()

@dp.message(BotStates.entering_voice_generation_prompt, F.text)
async def receive_voice_generation_prompt(message: Message, state: FSMContext):
    """Receive voice generation prompt"""
    await state.update_data(voice_gen_prompt=message.text)
    await message.answer("📝 Great! Now enter a name for this voice profile:")
    await state.set_state(BotStates.entering_generated_voice_name)

@dp.message(BotStates.entering_generated_voice_name, F.text)
async def receive_generated_voice_name(message: Message, state: FSMContext):
    """Receive name and generate voice"""
    voice_name = message.text
    data = await state.get_data()
    prompt = data.get("voice_gen_prompt")
    
    await message.answer("⏳ <b>Generating voice...</b>\n\nThis may take a moment...", parse_mode="HTML")
    
    try:
        # Generate voice (note: this is a preview, actual implementation may vary)
        result = await generate_voice_from_prompt(prompt)
        
        if result:
            await message.answer(
                f"✅ <b>Voice generated successfully!</b>\n\n"
                f"🎤 Name: {voice_name}\n\n"
                f"Note: The generated voice has been previewed. "
                f"To save it permanently to your ElevenLabs account, please use the ElevenLabs website.",
                parse_mode="HTML"
            )
            
            # If result contains audio, send it
            if hasattr(result, 'audio'):
                await message.answer_audio(
                    audio=BufferedInputFile(result.audio, filename=f"{voice_name}_preview.mp3"),
                    caption="🎵 Voice preview"
                )
        else:
            await message.answer("❌ Failed to generate voice. Please try again.")
        
        await state.clear()
        
    except Exception as e:
        await message.answer(f"❌ <b>Error generating voice:</b>\n{str(e)}", parse_mode="HTML")
        await state.clear()

# ================== Google Drive Upload Handler ==================
@dp.callback_query(F.data.startswith("drive_upload:"))
async def handle_drive_upload(callback: CallbackQuery):
    """Handle Google Drive upload button press"""
    try:
        # Extract video path from callback data
        video_path = callback.data.split("drive_upload:", 1)[1]

        # Check if file exists
        if not Path(video_path).exists():
            await callback.answer("❌ Video file not found!", show_alert=True)
            return

        # Upload to Google Drive
        await callback.answer("⏳ Uploading to Google Drive...", show_alert=False)
        await callback.message.edit_reply_markup(reply_markup=None)

        drive_url = await upload_to_google_drive(video_path)

        # Send success message
        await callback.message.answer(
            f"✅ <b>Video uploaded to Google Drive!</b>\n\n"
            f"🔗 Link: {drive_url}",
            parse_mode="HTML"
        )

        # Delete the local file after successful upload
        try:
            Path(video_path).unlink()
        except Exception as e:
            logger.warning(f"Failed to delete video file {video_path}: {str(e)}")

    except Exception as e:
        logger.error(f"Error in drive upload handler: {str(e)}")
        await callback.answer("❌ Upload failed! Please try again.", show_alert=True)

# ================== Main ==================
async def main():
    """Start the bot"""
    print("🤖 Bot is starting...")
    print("✅ Bot is running and waiting for updates...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())