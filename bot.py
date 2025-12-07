import asyncio
import json
import os
import zipfile
from io import BytesIO
from pathlib import Path
import aiohttp
import fal_client
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile, BufferedInputFile
from dotenv import load_dotenv

# ================== Configuration ==================
# Load environment variables from .env file
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
FAL_KEY = os.getenv("FAL_KEY")
KIE_API_KEY = os.getenv("KIE_API_KEY")

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable not set")
if not FAL_KEY:
    raise ValueError("FAL_KEY environment variable not set")
if not KIE_API_KEY:
    raise ValueError("KIE_API_KEY environment variable not set")

# ================== FSM States ==================
class BotStates(StatesGroup):
    waiting_photos = State()
    choosing_prompt_mode = State()
    entering_single_prompt = State()
    entering_individual_prompts = State()
    choosing_num_images = State()
    generating_images = State()
    waiting_zip_confirmation = State()
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

def create_zip_from_images(image_data_list: list) -> BytesIO:
    """Create ZIP file from list of (filename, bytes) tuples"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, data in image_data_list:
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

def create_number_keyboard(max_num: int, callback_prefix: str) -> InlineKeyboardMarkup:
    """Create inline keyboard with numbers from 1 to max_num in a single row"""
    buttons = [
        InlineKeyboardButton(text=str(i), callback_data=f"{callback_prefix}_{i}")
        for i in range(1, max_num + 1)
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons])

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

# ================== Command Handlers ==================
@dp.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    """Start command - begin the workflow"""
    await state.clear()
    await message.answer(
        "🎬 <b>Welcome to the Video Generation Bot!</b>\n\n"
        "I'll help you transform your photos into stunning generated images and videos.\n\n"
        "📸 <b>Step 1:</b> Please send me your source photos.\n"
        "You can send multiple photos (up to 50-60 in production).\n\n"
        "When you're done, type /done",
        parse_mode="HTML"
    )
    await state.set_state(BotStates.waiting_photos)
    await state.update_data(source_photos=[])

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
    """Generate images using fal-ai nano-banana"""
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
        for idx, photo_file_id in enumerate(source_photos):
            # Download photo from Telegram
            file = await bot.get_file(photo_file_id)
            photo_path = temp_dir / f"source_{idx}.jpg"
            await bot.download_file(file.file_path, photo_path)

            # Upload to FAL
            image_url = fal_client.upload_file(str(photo_path))

            # Get prompt for this photo
            if prompt_mode == "single":
                prompt = single_prompt
            else:
                prompt = individual_prompts[idx]

            # Generate images using nano-banana
            await message.answer(f"🎨 Generating images for photo {idx + 1}/{len(source_photos)}...")

            result = await asyncio.to_thread(
                fal_client.subscribe,
                "fal-ai/nano-banana/edit",
                arguments={
                    "prompt": prompt,
                    "num_images": num_images,
                    "image_urls": [image_url],
                    "output_format": "png"
                },
                with_logs=False
            )

            # Download generated images
            for img_idx, img in enumerate(result.get("images", [])):
                img_url = img.get("url")
                img_data = await download_file(img_url)
                filename = f"generated_{idx}_{img_idx}.png"
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
    """Generate videos using KIE AI Grok Imagine"""
    data = await state.get_data()
    generated_images = data.get("generated_images", [])
    num_videos_per_image = data.get("num_videos_per_image", 1)
    video_prompts = data.get("video_prompts", [])

    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)

    try:
        prompt_idx = 0

        async with aiohttp.ClientSession() as session:
            for img_idx, (filename, img_data) in enumerate(generated_images):
                # Save image temporarily
                img_path = temp_dir / filename
                with open(img_path, 'wb') as f:
                    f.write(img_data)

                # Upload to FAL to get URL (KIE accepts external image URLs)
                image_url = fal_client.upload_file(str(img_path))

                for vid_idx in range(num_videos_per_image):
                    prompt = video_prompts[prompt_idx]
                    prompt_idx += 1

                    await message.answer(
                        f"🎬 Generating video {prompt_idx}/{len(video_prompts)}...\n"
                        f"Image: {filename}\n"
                        f"Prompt: {prompt[:50]}..."
                    )

                    # Create video task using KIE AI Grok Imagine
                    create_url = "https://api.kie.ai/api/v1/jobs/createTask"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {KIE_API_KEY}"
                    }
                    payload = {
                        "model": "grok-imagine/image-to-video",
                        "input": {
                            "image_urls": [image_url],
                            "prompt": prompt,
                            "mode": "normal"
                        }
                    }

                    async with session.post(create_url, headers=headers, json=payload) as response:
                        result = await response.json()
                        if result.get("code") != 200:
                            raise Exception(f"Failed to create task: {result.get('message')}")
                        task_id = result.get("data", {}).get("taskId")

                    # Poll for completion
                    query_url = f"https://api.kie.ai/api/v1/jobs/queryTask/{task_id}"
                    max_attempts = 120  # 10 minutes with 5 second intervals
                    attempt = 0
                    video_url = None

                    while attempt < max_attempts:
                        await asyncio.sleep(5)
                        async with session.get(query_url, headers=headers) as response:
                            task_result = await response.json()

                            task_data = task_result.get("data", {})
                            state_value = task_data.get("state")

                            if state_value == "success":
                                # Get video URL from result
                                result_json = task_data.get("resultJson")
                                if isinstance(result_json, str):
                                    result_json = json.loads(result_json)
                                video_url = result_json.get("resultUrls", [])[0]
                                break
                            elif state_value == "fail":
                                fail_msg = task_data.get("failMsg", "Unknown error")
                                raise Exception(f"Video generation failed: {fail_msg}")
                            # If state is not success or fail, continue polling (task is processing)

                        attempt += 1

                    if attempt >= max_attempts:
                        raise Exception("Video generation timed out")

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
