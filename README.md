# 🎬 Telegram Video Generation Bot

A simple and clean Telegram bot that transforms photos into AI-generated images and videos using FAL.ai's nano-banana and WAN 2.5 models.

## ✨ Features

- 📸 Accept multiple source photos (up to 50-60)
- 🎨 Generate edited images using FAL nano-banana model
- 🎬 Animate images into videos using FAL WAN 2.5 model
- 📝 Support for single prompt or individual prompts per photo
- 🗜️ ZIP file handling for batch operations
- ✅ Review and correction workflow

## 🔧 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your API Keys

- **Telegram Bot Token**: Talk to [@BotFather](https://t.me/BotFather) on Telegram and create a new bot
- **FAL API Key**: Sign up at [fal.ai](https://fal.ai/) and get your API key from the dashboard

### 3. Configure Environment Variables

**RECOMMENDED: Use .env file**

Edit the `.env` file in the project root and add your API keys:

```env
# Telegram Bot Configuration
BOT_TOKEN=your_telegram_bot_token_here

# FAL.ai API Configuration
FAL_KEY=your_fal_api_key_here
```

**Alternative: Set environment variables manually**

If you prefer not to use .env file:

**Windows (Command Prompt):**
```cmd
set BOT_TOKEN=your_telegram_bot_token_here
set FAL_KEY=your_fal_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:BOT_TOKEN="your_telegram_bot_token_here"
$env:FAL_KEY="your_fal_api_key_here"
```

**Linux/Mac:**
```bash
export BOT_TOKEN="your_telegram_bot_token_here"
export FAL_KEY="your_fal_api_key_here"
```

## 🚀 Running the Bot

```bash
python bot.py
```

You should see:
```
🤖 Bot is starting...
```

## 📖 How to Use

### Step 1: Start the Bot
Send `/start` to your bot on Telegram

### Step 2: Upload Photos
Send one or more photos (the bot will wait for all photos)
When done, send `/done`

### Step 3: Choose Prompt Mode
- **One prompt for ALL photos**: Use the same prompt for all images
- **Individual prompts for EACH**: Write a different prompt for each photo

### Step 4: Write Prompts
Write your image generation prompt(s) according to the mode you selected

### Step 5: Select Number of Images
Choose how many images to generate per source photo (1-10)

### Step 6: Review Generated Images
The bot will send a ZIP file with all generated images
You can either:
- ✅ **Confirm**: Use all generated images
- 📎 **Upload corrected ZIP**: Remove bad images and upload the corrected ZIP

### Step 7: Select Number of Videos
Choose how many videos to generate per image (1-5)

### Step 8: Write Video Prompts
Write prompts for each video, describing the motion/animation you want

### Step 9: Receive Videos
The bot will generate and send all videos one by one

## 🎯 Workflow Example

1. User sends 3 photos
2. User chooses "One prompt for ALL"
3. User writes: "make the person smile and add sunglasses"
4. User selects: 2 images per photo → 6 total images generated
5. Bot sends ZIP with 6 images
6. User confirms or uploads corrected ZIP (e.g., removes 1 bad image → 5 images remain)
7. User selects: 1 video per image → 5 videos total
8. User writes 5 video prompts (one for each image)
9. Bot generates and sends 5 videos

## ⚙️ Technical Details

### API Integration

**Image Generation (nano-banana):**
- Model: `fal-ai/nano-banana/edit`
- Method: `fal_client.subscribe()`
- Input: prompt, num_images, image_urls
- Output: list of edited images

**Video Generation (WAN 2.5):**
- Model: `fal-ai/wan-25-preview/image-to-video`
- Method: `fal_client.subscribe()`
- Input: prompt, image_url, resolution (1080p), duration (5s)
- Output: generated video file

### Project Structure

```
VideoGenericBot/
├── bot.py                      # Main bot code (single file)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── nano banana api docs.txt    # FAL nano-banana API documentation
└── wan2.5 docs.txt            # FAL WAN 2.5 API documentation
```

### FSM States

- `waiting_photos` - Waiting for user to send photos
- `choosing_prompt_mode` - User selecting prompt mode
- `entering_single_prompt` - User entering single prompt
- `entering_individual_prompts` - User entering multiple prompts
- `choosing_num_images` - User selecting number of images
- `generating_images` - Bot generating images
- `waiting_zip_confirmation` - Waiting for user to confirm/correct ZIP
- `choosing_num_videos` - User selecting number of videos
- `entering_video_prompts` - User entering video prompts
- `generating_videos` - Bot generating videos

## 🛠️ Commands

- `/start` - Start the bot and begin the workflow
- `/cancel` - Cancel current operation
- `/done` - Finish sending photos and proceed to next step

## ⚠️ Error Handling

The bot handles:
- API errors (422, network issues)
- Invalid file formats
- Empty ZIP files
- Missing environment variables

All errors are reported to the user with clear messages.

## 📝 Notes

- Images are generated in PNG format
- Videos are generated in 1080p, 5 seconds duration
- Temporary files are automatically cleaned up
- All operations are asynchronous for better performance

## 🎨 UI Features

- Clean inline keyboards with emojis
- Beautiful formatted messages with HTML
- Progress updates during generation
- Compact button layouts (single row when possible)

## 🔒 Security

- Bot token and FAL API key are loaded from environment variables
- No credentials stored in code
- Temporary files are deleted after use

---

**Enjoy creating amazing AI-generated videos! ✨**
