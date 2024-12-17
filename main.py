import os
import logging
from dotenv import load_dotenv
import discord
from discord.ext import commands
from openai import AsyncOpenAI
import aiofiles
import uuid

# 全局常數
NAME_MAPPING = {
    "o1": "o1-preview",
    "o1m": "o1-mini",
    "4o": "chatgpt-4o-latest"
}

# 設定日誌
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("discord_bot.log", encoding="utf-8")
        ]
    )
    return logging.getLogger(__name__)

# 載入並檢查環境變數
def load_configuration():
    load_dotenv()
    bot_token = os.getenv("DC_BOT_TOKEN")
    openai_api_key = os.getenv("OPENAI_KEY")
    raw_channel_mapping = os.getenv("ALLOWED_CHANNEL_IDS", "")

    if not bot_token:
        logging.error("缺少 Discord Bot Token (DC_BOT_TOKEN)")
        exit(1)

    if not openai_api_key:
        logging.error("缺少 OpenAI API Key (OPENAI_KEY)")
        exit(1)

    allowed_channels = parse_allowed_channels(raw_channel_mapping)
    return bot_token, openai_api_key, allowed_channels

# 解析允許的伺服器與頻道 ID
def parse_allowed_channels(raw_channel_mapping: str) -> set:
    allowed_channels = set()
    for entry in raw_channel_mapping.split(","):
        if ":" in entry:
            server_id, channel_id = entry.split(":", 1)
            try:
                allowed_channels.add((int(server_id.strip()), int(channel_id.strip())))
            except ValueError:
                logging.warning(f"伺服器或頻道 ID 不是有效的整數：{entry}")
        else:
            logging.warning(f"條目格式錯誤（缺少冒號）：{entry}")
    return allowed_channels

# 初始化 OpenAI 客戶端
def initialize_openai_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)

# 初始化 Discord Bot
def initialize_bot() -> commands.Bot:
    intents = discord.Intents.default()
    intents.guilds = True
    intents.messages = True
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)
    return bot

# 非同步函數：向 OpenAI API 發送請求並獲取回覆
async def fetch_openai_response(openai_client: AsyncOpenAI, model: str, user_message: str, logger: logging.Logger) -> str:
    """
    向 OpenAI API 發送請求並獲取回覆。

    :param openai_client: 初始化好的 OpenAI 客戶端。
    :param model: 使用的 OpenAI 模型。
    :param user_message: 用戶提供的內容。
    :param logger: 日誌紀錄器。
    :return: OpenAI 回應的內容。
    """
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenAI API 請求失敗: %s", e)
        return "抱歉，發生錯誤，無法獲取回覆。"

# 非同步函數：將回應內容儲存到文本檔案
async def save_response_to_file(response: str) -> str:
    """
    將回應內容儲存到唯一的文本檔案。

    :param response: OpenAI 回覆內容。
    :return: 儲存的文件路徑。
    """
    unique_filename = f"message_{uuid.uuid4()}.txt"
    async with aiofiles.open(unique_filename, mode="w", encoding="utf-8") as file:
        await file.write(response)
    return unique_filename

# 檢查附件並讀取檔案內容
async def process_attachments(message):
    """
    檢查附件，讀取 .txt 文件內容，並解析模型名稱和訊息。
    格式: 模型名稱/內容
    """
    if not message.attachments:
        return None, None, "訊息中沒有附件。"

    # 找到第一個 .txt 文件
    txt_file = next((att for att in message.attachments if att.filename.endswith('.txt')), None)
    if not txt_file:
        return None, None, "附件中未包含 .txt 文件。"

    try:
        # 下載和讀取 .txt 文件內容
        async with aiofiles.open(f"temp_{txt_file.filename}", mode="wb") as file:
            await txt_file.save(file.name)

        async with aiofiles.open(f"temp_{txt_file.filename}", mode="r", encoding="utf-8") as file:
            content = await file.read()

        os.remove(f"temp_{txt_file.filename}")  # 刪除臨時文件

        # 解析文件內容
        first_line, *remaining_lines = content.split("\n", 1)
        if "/" not in first_line:
            return None, None, "格式錯誤，第一行必須包含模型名稱與內容（格式：模型名稱/內容）。"

        # 提取模型名稱和訊息內容
        model_name, *user_message_parts = first_line.split("/", 1)
        user_message = user_message_parts[0].strip() if user_message_parts else ""

        # 剩下的行加入訊息內容
        if remaining_lines:
            user_message += "\n" + remaining_lines[0].strip()

        return model_name.strip(), user_message.strip(), None
    except Exception as e:
        return None, None, f"讀取文件失敗：{str(e)}"

async def handle_message(message: discord.Message, bot: commands.Bot, openai_client: AsyncOpenAI, allowed_channels: set, logger: logging.Logger):
    if message.author == bot.user:
        return

    guild_id = message.guild.id if message.guild else None
    channel_id = message.channel.id

    # 確保處理允許的頻道
    if not guild_id or (guild_id, channel_id) not in allowed_channels:
        return

    # 處理討論串的刪除邏輯
    if isinstance(message.channel, discord.Thread) and message.content.strip() == "!del":
        thread_name = message.channel.name
        await message.channel.delete()
        logger.info(f"[討論串刪除] 討論串: {thread_name}, 由 {message.author.name} 觸發")
        return

    # 確認訊息是否提到 Bot
    if not bot.user.mentioned_in(message):
        return

    logger.info("[訊息記錄] 用戶: %s, 訊息: %s", message.author.name, message.content)

    user_message = None
    model_name = None

    try:
        # 處理附件（.txt 文件）
        model_name, user_message, error = await process_attachments(message)
        if error:
            logger.warning(f"附件處理錯誤：{error}")
            await message.channel.send(error)
            return

        # 如果沒有附件，解析訊息格式
        if not user_message:
            content_lines = message.content.splitlines()
            content = "\n".join(line.rstrip() for line in content_lines)
            first_line, *remaining_lines = content.split("\n", 1)
            parts = first_line.split(" ", 2)

            if len(parts) >= 3:
                _, name, *info = parts
                remaining_info = "\n".join(remaining_lines) if remaining_lines else ""
                model_name = name.strip()
                user_message = f"{' '.join(info)}\n{remaining_info}".strip()
            else:
                await message.channel.send("訊息格式錯誤，請使用正確的格式或上傳 .txt 文件。")
                return

        # 確保模型名稱存在
        converted_name = NAME_MAPPING.get(model_name, None)
        if not converted_name:
            await message.channel.send(f"未知的模型名稱：{model_name}")
            return

        # 檢查訊息內容
        if not user_message:
            await message.channel.send("訊息內容為空，請檢查文件格式。")
            return

        # 獲取 OpenAI 回覆
        openai_reply = await fetch_openai_response(openai_client, converted_name, user_message, logger)

        # 回覆邏輯：如果在討論串內則直接回覆，否則創建討論串
        if isinstance(message.channel, discord.Thread):
            await message.channel.send(openai_reply)
        else:
            thread_name = f"AI 回覆：{model_name}" if model_name else "AI 討論串"
            thread = await message.create_thread(name=thread_name, auto_archive_duration=60)
            await thread.send(openai_reply)

    except Exception as e:
        logger.error("處理訊息時發生錯誤：%s", str(e))
        await message.channel.send("處理您的請求時出現錯誤，請稍後重試。")

# 主函數
def main():
    # 設定日誌
    logger = setup_logging()

    # 載入配置
    bot_token, openai_api_key, allowed_channels = load_configuration()

    # 初始化 OpenAI 客戶端
    openai_client = initialize_openai_client(openai_api_key)

    # 初始化 Discord Bot
    bot = initialize_bot()

    @bot.event
    async def on_ready():
        logger.info("Bot 已上線，名稱：%s", bot.user)

    @bot.event
    async def on_message(message):
        await handle_message(message, bot, openai_client, allowed_channels, logger)
        await bot.process_commands(message)  # 確保命令能被處理

    # 啟動 Bot
    bot.run(bot_token)

if __name__ == "__main__":
    main()
