# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Telegram bot built with aiogram 3.x that transforms user photos into AI-generated images (FAL Seedream V4) and videos (FAL WAN 2.5). Features OpenAI-powered unique prompt generation. Single-file architecture in bot.py (~730 lines), minimalist design with no over-engineering.

## Development Commands

### Running the Bot
```bash
python bot.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Create/edit `.env` file with:
```
BOT_TOKEN=your_telegram_bot_token
FAL_KEY=your_fal_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Architecture & Key Patterns

### Single-File Structure
The entire bot logic is in `bot.py`. No separate modules, classes, or abstractions. Functions are simple and purpose-driven.

### FSM State Flow

**Photo-to-video workflow:**
```
waiting_photos → choosing_prompt_mode → entering_single_prompt/entering_individual_prompts →
choosing_num_images → generating_images → waiting_zip_confirmation →
choosing_num_videos → entering_video_prompts → generating_videos
```

**Direct video-from-images workflow:**
```
waiting_video_zip → choosing_num_videos → entering_video_prompts → generating_videos
```

State data stored in FSMContext:
- `source_photos`: List of Telegram file IDs
- `prompt_mode`: "single" or "individual"
- `single_prompt` or `individual_prompts`: User prompts
- `unique_prompts`: Generated unique prompts (when using OpenAI)
- `num_images_per_photo`: 1-10 images per source photo
- `generated_images`: List of (filename, bytes) tuples
- `video_images`: List of (filename, bytes) for direct video workflow
- `num_videos_per_image`: 1-5 videos per image
- `video_prompts`: List of video animation prompts

### Critical Implementation Details

**aiogram 3.x File Handling:**
```python
# CORRECT - aiogram 3.x does NOT have FSInputFile.from_buffer()
from aiogram.types import BufferedInputFile
document = BufferedInputFile(zip_bytes, filename="file.zip")
await message.answer_document(document=document)
```

**FAL Client Async Wrapping:**
```python
# FAL client is synchronous - must wrap in asyncio.to_thread()
result = await asyncio.to_thread(
    fal_client.subscribe,
    "fal-ai/bytedance/seedream/v4/edit",
    arguments={...},
    with_logs=False
)
```

**Environment Loading:**
```python
# MUST call load_dotenv() before reading os.getenv()
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
```

## API Integration

### Image Generation (FAL Seedream V4)
Model: `fal-ai/bytedance/seedream/v4/edit`

Method: Synchronous FAL client wrapped in `asyncio.to_thread()`

Arguments:
- `prompt`: str - Image editing prompt
- `num_images`: int - Number of images to generate
- `image_urls`: list[str] - Uploaded image URLs (via `fal_client.upload_file()`)
- `image_size`: dict - `{"width": 1024, "height": 1024}`

Returns: `{"images": [{"url": str, ...}], "seed": int}`

**Note:** Seedream V4 does not support negative prompts or output format parameters.

**Unique Prompt Generation:**
When using single-prompt mode with multiple images, the bot generates unique variations via OpenAI's `gpt-4o-mini` model:
- Calls `generate_unique_prompts(base_prompt, count)`
- Returns list of unique prompts that vary environment, scene, clothing, lighting, camera angle
- Does NOT vary body shape, anatomy, face details, pose
- Each generated image uses one unique prompt (prevents identical outputs)

### Video Generation (FAL WAN 2.5)
Model: `fal-ai/wan-25-preview/image-to-video`

Method: Synchronous FAL client wrapped in `asyncio.to_thread()`

Arguments:
- `prompt`: str - Motion/animation description
- `image_url`: str - Public URL of image (uploaded via `fal_client.upload_file()`)
- `resolution`: "720p"
- `duration`: "5" (seconds)

Returns: `{"video": {"url": str}}`

Video is downloaded and streamed to user immediately after generation.

## Workflow Patterns

### Photo-to-Video Workflow
1. User selects "Generate images from photos" mode
2. User uploads photos → stored as Telegram file IDs
3. User selects prompt mode (single vs individual)
4. User writes prompts
5. User selects number of images per photo (1-10)
6. Bot generates images:
   - If single-prompt mode: calls OpenAI to generate unique prompts
   - Downloads photos from Telegram
   - Uploads to FAL
   - Calls Seedream V4 API (one by one if using unique prompts)
   - Downloads generated images
   - Creates ZIP
7. User reviews ZIP (can upload corrected version)
8. User selects number of videos per image (1-5)
9. User writes video prompts
10. Bot generates videos:
    - Uploads images to FAL (to get public URLs)
    - Calls FAL WAN 2.5 API
    - Downloads and streams videos to user one-by-one

### Direct Video-from-Images Workflow
1. User selects "Generate video from existing images" mode
2. User uploads images (individual photos or ZIP file)
3. User types /done
4. User selects number of videos per image (1-5)
5. User writes video prompts
6. Bot generates videos (same as step 10 above)

## Common Patterns

### Inline Keyboards
Single row for numbers:
```python
buttons = [InlineKeyboardButton(text=str(i), callback_data=f"prefix_{i}")
           for i in range(1, max_num + 1)]
keyboard = InlineKeyboardMarkup(inline_keyboard=[buttons])
```

Multiple rows for options:
```python
buttons = [
    [InlineKeyboardButton(text="Option 1", callback_data="opt1")],
    [InlineKeyboardButton(text="Option 2", callback_data="opt2")]
]
```

### Callback Query Handling
Always answer callback queries and edit message to remove keyboard:
```python
@dp.callback_query(state, F.data.startswith("prefix_"))
async def handler(callback: CallbackQuery, state: FSMContext):
    value = callback.data.split("_")[-1]
    await callback.message.edit_text("Confirmed!")
    await callback.answer()
```

### Temporary File Cleanup
Always use try/finally:
```python
temp_dir = Path("temp_photos")
temp_dir.mkdir(exist_ok=True)
try:
    # Work with files
    ...
finally:
    for file in temp_dir.glob("*"):
        file.unlink()
    temp_dir.rmdir()
```

### Photo Collection
```python
@dp.message(state, F.photo)
async def receive_photo(message: Message, state: FSMContext):
    photo = message.photo[-1]  # Get largest size
    photos.append(photo.file_id)
```

## Important Notes

- Message formatting uses HTML: `parse_mode="HTML"` with `<b>` tags
- Progress messages during generation keep user informed
- ZIP workflow allows batch review and correction
- Videos are sent individually as they're generated (streaming pattern)
- All errors send user-friendly messages and clear state
- No persistent storage - all data in FSM MemoryStorage
- Image uploads use FAL's `upload_file()` which returns public URLs
- Two start modes: photo-to-video (full workflow) or direct video generation (existing images)
- Seedream V4 does not support negative prompts (unlike previous nano-banana model)
- Logging configured with rotating file handler (max 5MB, 2 backups) at videogenericbot.log
- Unique prompt generation uses strict JSON array format, fallback to base prompt if OpenAI fails
