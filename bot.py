import asyncio
import json
import os
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set")
if not FAL_KEY:
    raise ValueError("FAL_KEY environment variable not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

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

async def extract_images_from_zip(zip_data: bytes) -> list:
    """Extract images from ZIP file, returns list of (filename, bytes)"""
    images = []
    with zipfile.ZipFile(BytesIO(zip_data), 'r') as zip_file:
        for filename in zip_file.namelist():
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                images.append((filename, zip_file.read(filename)))
    return images

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
        [InlineKeyboardButton(text="🎬 Generate video from existing images", callback_data="start_mode_video")]
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

async def generate_unique_prompts(base_prompt: str, count: int) -> list[str]:
    """Generate unique prompts using OpenAI API"""
    logger.info("generate_unique_prompts START count=%s prompt=%s", count, base_prompt[:120])
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            instruction = f"""Generate exactly {count} short image prompts based on: "{base_prompt}"

Rules:
- Return ONLY a plain JSON array of strings: ["prompt 1", "prompt 2", ...]
- NO nested JSON objects, NO markdown, NO triple backticks
- Each prompt: maximum 200 characters, 1-2 short sentences
- Vary ONLY: environment, scene, clothing style, small props, lighting, camera angle
- DO NOT describe: body shape, anatomy, face details, pose, sexual/explicit content
- Keep person's appearance implicit and unchanged
- Example: "young woman in a cozy kitchen, soft morning light, holding a mug, candid social-media style"

Output {count} prompts as a JSON array now:"""

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": instruction
                    }
                ]
            }

            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
                logger.info("OpenAI status=%s", response.status)
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                logger.debug("OpenAI raw: %s", content[:4000])

                # Remove markdown code blocks if present
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # Parse JSON array from response
                prompts = json.loads(content)

                # Validate: must be a list of strings
                if not isinstance(prompts, list):
                    logger.error("OpenAI returned non-list: %s", type(prompts))
                    return [base_prompt] * count
                if not all(isinstance(p, str) for p in prompts):
                    logger.error("OpenAI returned non-string items in list")
                    return [base_prompt] * count

                return prompts
    except Exception as e:
        logger.exception("OpenAI error")
        # Fallback: return list repeating the base prompt
        return [base_prompt] * count

# ================== Command Handlers ==================
@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    """Start command - begin the workflow"""
    await state.clear()
    await message.answer(
        "🎬 <b>Welcome to the Video Generation Bot!</b>\n\n"
        "Please choose how you want to start:",
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

    # OpenAI unique prompt generation DISABLED - use user's exact prompt
    # if prompt_mode == "single":
    #     num_photos = len(data.get("source_photos", []))
    #     total_images = num_photos * num_images
    #     base_prompt = data.get("single_prompt")
    #
    #     await callback.message.answer("🤖 <b>Generating unique prompts...</b>\n\nPlease wait...", parse_mode="HTML")
    #     unique_prompts = await generate_unique_prompts(base_prompt, total_images)
    #     await state.update_data(unique_prompts=unique_prompts)

    await callback.message.answer("⏳ <b>Generating images...</b>\n\nThis may take a few moments. Please wait...", parse_mode="HTML")
    await callback.answer()

    # Start image generation
    await generate_images(callback.message, state)

# ================== Image Generation ==================
async def generate_images(message: Message, state: FSMContext):
    """Generate images using fal-ai nano-banana-pro"""
    data = await state.get_data()
    source_photos = data.get("source_photos", [])
    num_images = data.get("num_images_per_photo", 1)
    prompt_mode = data.get("prompt_mode")
    single_prompt = data.get("single_prompt")
    individual_prompts = data.get("individual_prompts", [])
    unique_prompts = data.get("unique_prompts", [])

    all_generated_images = []
    temp_dir = Path("temp_photos")
    temp_dir.mkdir(exist_ok=True)

    global_index = 0

    try:
        for idx, photo_file_id in enumerate(source_photos):
            # Download photo from Telegram
            file = await bot.get_file(photo_file_id)
            photo_path = temp_dir / f"source_{idx}.jpg"
            await bot.download_file(file.file_path, photo_path)

            # Upload to FAL
            image_url = fal_client.upload_file(str(photo_path))

            # Generate images using nano-banana-pro
            await message.answer(f"🎨 Generating images for photo {idx + 1}/{len(source_photos)}...")

            # Use user's exact prompt - unique prompt generation DISABLED
            if False:  # prompt_mode == "single" and unique_prompts:
                # Generate images one by one with unique prompts (DISABLED)
                total_images = len(source_photos) * num_images
                for img_idx in range(num_images):
                    prompt = unique_prompts[global_index]

                    # Show progress
                    await message.answer(f"🔄 Generating image {global_index + 1}/{total_images}...")
                    logger.info("Generating image %s/%s (global_index=%s)", global_index + 1, total_images, global_index)

                    # Log nano-banana-pro request (truncated)
                    request_args = {
                        "prompt": prompt,
                        "num_images": 1,
                        "image_urls": [image_url],
                        "output_format": "png",
                        "resolution": "1K",
                        "aspect_ratio": "1:1"
                    }
                    logger.debug("nano-banana-pro request: %s", str(request_args)[:1500])

                    result = await asyncio.to_thread(
                        fal_client.subscribe,
                        "fal-ai/nano-banana-pro/edit",
                        arguments=request_args,
                        with_logs=False
                    )

                    # Log nano-banana-pro response info
                    logger.debug("nano-banana-pro response: images_count=%d", len(result.get("images", [])))

                    # Download generated image with safety check
                    images = result.get("images", [])
                    if images and len(images) > 0:
                        img = images[0]
                        img_url = img.get("url")
                        if img_url:
                            img_data = await download_file(img_url)
                            filename = f"generated_{idx}_{img_idx}.png"
                            log_image_details(img_data, filename)
                            all_generated_images.append((filename, img_data))
                            logger.info("Image generated global_index=%s filename=%s", global_index, filename)
                    else:
                        logger.error("Empty images array for prompt=%s global_index=%s", prompt[:120], global_index)

                    global_index += 1
            else:
                # Original logic for individual mode
                if prompt_mode == "single":
                    prompt = single_prompt
                else:
                    prompt = individual_prompts[idx]

                # Log nano-banana-pro request (truncated)
                request_args = {
                    "prompt": prompt,
                    "num_images": num_images,
                    "image_urls": [image_url],
                    "output_format": "png",
                    "resolution": "1K",
                    "aspect_ratio": "1:1"
                }
                logger.debug("nano-banana-pro request: %s", str(request_args)[:1500])

                result = await asyncio.to_thread(
                    fal_client.subscribe,
                    "fal-ai/nano-banana-pro/edit",
                    arguments=request_args,
                    with_logs=False
                )

                # Log nano-banana-pro response info
                logger.debug("nano-banana-pro response: images_count=%d", len(result.get("images", [])))

                # Download generated images with safety check
                for img_idx, img in enumerate(result.get("images", [])):
                    img_url = img.get("url")
                    if img_url:
                        img_data = await download_file(img_url)
                        filename = f"generated_{idx}_{img_idx}.png"
                        log_image_details(img_data, filename)
                        all_generated_images.append((filename, img_data))

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
    """Generate videos using FAL AI WAN 2.5"""
    data = await state.get_data()
    generated_images = data.get("generated_images", [])
    num_videos_per_image = data.get("num_videos_per_image", 1)
    video_prompts = data.get("video_prompts", [])

    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)

    try:
        prompt_idx = 0

        for img_idx, (filename, img_data) in enumerate(generated_images):
            # Save image temporarily
            img_path = temp_dir / filename
            with open(img_path, 'wb') as f:
                f.write(img_data)

            # Upload to FAL to get URL
            image_url = fal_client.upload_file(str(img_path))

            for vid_idx in range(num_videos_per_image):
                prompt = video_prompts[prompt_idx]
                prompt_idx += 1

                await message.answer(
                    f"🎬 Generating video {prompt_idx}/{len(video_prompts)}...\n"
                    f"Image: {filename}\n"
                    f"Prompt: {prompt[:50]}..."
                )

                # Generate video using FAL AI WAN 2.5
                result = await asyncio.to_thread(
                    fal_client.subscribe,
                    "fal-ai/wan-25-preview/image-to-video",
                    arguments={
                        "prompt": prompt,
                        "image_url": image_url,
                        "resolution": "720p",
                        "duration": "5",
                        "negative_prompt": "deformed face, distorted body, extra limbs, missing limbs, mismatched eyes, warped anatomy, AI artifacts, glitch, blur, low resolution, oversharpening, unnatural skin, plastic texture, flickering frames, jitter, unstable motion, unnatural hair movement, exaggerated expressions, incorrect lighting, watermark, text, logo, double face, duplicate features, asymmetrical eyes, bad proportions, cartoonish look, unrealistic body physics.",
                        "enable_prompt_expansion": True,
                        "enable_safety_checker": False
                    },
                    with_logs=False
                )

                # Get video URL from result
                video_url = result.get("video", {}).get("url")

                # Download and send video
                if video_url:
                    video_data = await download_file(video_url)
                    await message.answer_video(
                        video=BufferedInputFile(video_data, filename=f"video_{img_idx}_{vid_idx}.mp4"),
                        caption=f"✨ Video {prompt_idx}/{len(video_prompts)}\n\n📝 Prompt: {prompt}"
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

# ================== Main ==================
async def main():
    """Start the bot"""
    print("🤖 Bot is starting...")
    print("✅ Bot is running and waiting for updates...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
