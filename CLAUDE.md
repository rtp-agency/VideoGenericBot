# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Telegram bot built with aiogram 3.x that transforms user photos into AI-generated images (FAL nano-banana) and videos (KIE AI Grok Imagine). Single-file architecture in bot.py (~530 lines), minimalist design with no over-engineering.

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
KIE_API_KEY=your_kie_api_key
```

## Architecture & Key Patterns

### Single-File Structure
The entire bot logic is in `bot.py`. No separate modules, classes, or abstractions. Functions are simple and purpose-driven.

### FSM State Flow
```
waiting_photos → choosing_prompt_mode → entering_prompts →
choosing_num_images → generating_images → waiting_zip_confirmation →
choosing_num_videos → entering_video_prompts → generating_videos
```

State data stored in FSMContext:
- `source_photos`: List of Telegram file IDs
- `prompt_mode`: "single" or "individual"
- `single_prompt` or `individual_prompts`: User prompts
- `num_images_per_photo`: 1-10 images per source photo
- `generated_images`: List of (filename, bytes) tuples
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
    "fal-ai/nano-banana/edit",
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

### Image Generation (FAL nano-banana)
Model: `fal-ai/nano-banana/edit`

Method: Synchronous FAL client wrapped in `asyncio.to_thread()`

Arguments:
- `prompt`: str - Image editing prompt
- `num_images`: int - Number of images to generate
- `image_urls`: list[str] - Uploaded image URLs (via `fal_client.upload_file()`)
- `output_format`: "png"

Returns: `{"images": [{"url": str, ...}], ...}`

### Video Generation (KIE AI Grok Imagine)
Model: `grok-imagine/image-to-video`

Method: Async HTTP requests with polling

**Create Task:**
- Endpoint: `POST https://api.kie.ai/api/v1/jobs/createTask`
- Headers: `Authorization: Bearer {KIE_API_KEY}`
- Payload:
  ```json
  {
    "model": "grok-imagine/image-to-video",
    "input": {
      "image_urls": ["https://..."],
      "prompt": "motion description",
      "mode": "normal"
    }
  }
  ```
- Response: `{"code": 200, "data": {"taskId": "..."}}`

**Poll Task Status:**
- Endpoint: `GET https://api.kie.ai/api/v1/jobs/queryTask/{taskId}`
- Poll every 5 seconds, max 120 attempts (10 minutes)
- Check `data.state`:
  - `"success"` → parse `data.resultJson` (JSON string) to get `resultUrls[0]`
  - `"fail"` → raise error with `data.failMsg`
  - Other → continue polling (task processing)

## Workflow Pattern

1. User uploads photos → stored as Telegram file IDs
2. User selects prompt mode (single vs individual)
3. User writes prompts
4. User selects number of images per photo
5. Bot generates images:
   - Downloads photos from Telegram
   - Uploads to FAL
   - Calls nano-banana API
   - Downloads generated images
   - Creates ZIP
6. User reviews ZIP (can upload corrected version)
7. User selects number of videos per image
8. User writes video prompts
9. Bot generates videos:
   - Uploads images to FAL (to get public URLs)
   - Creates KIE AI task with image URL
   - Polls task status until completion
   - Downloads and streams videos to user one-by-one

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
- Image uploads use FAL's `upload_file()` which returns public URLs for both FAL and KIE APIs
- KIE API responses: create task uses `"message"` field, query task uses `"msg"` field
- KIE API `resultJson` field is returned as JSON string and must be parsed with `json.loads()`
