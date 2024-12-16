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
    檢查附件，讀取 .txt 文件內容。
    """
    if not message.attachments:
        return None, "訊息中沒有附件。"

    # 找到第一個 .txt 文件
    txt_file = next((att for att in message.attachments if att.filename.endswith('.txt')), None)
    if not txt_file:
        return None, "附件中未包含 .txt 文件。"

    try:
        # 下載文件
        async with aiofiles.open(f"temp_{txt_file.filename}", mode="wb") as file:
            await txt_file.save(file.name)

        # 讀取文件內容
        async with aiofiles.open(f"temp_{txt_file.filename}", mode="r", encoding="utf-8") as file:
            content = await file.read()

        os.remove(f"temp_{txt_file.filename}")  # 刪除臨時文件
        return content, None
    except Exception as e:
        return None, f"讀取文件失敗：{str(e)}"

# 處理訊息的主要函數
async def handle_message(message: discord.Message, bot: commands.Bot, openai_client: AsyncOpenAI, allowed_channels: set, logger: logging.Logger):
    if message.author == bot.user:
        return

    guild_id = message.guild.id if message.guild else None
    channel_id = message.channel.id

    # 檢查是否允許處理此頻道的訊息
    if not guild_id or (guild_id, channel_id) not in allowed_channels:
        return

    # 日誌記錄
    logger.info(
        "[訊息記錄] 時間: %s, 伺服器: %s, 頻道: %s, 用戶: %s, 訊息: %s",
        message.created_at,
        message.guild.name if message.guild else "DM",
        message.channel.name if hasattr(message.channel, "name") else "Unknown",
        message.author.name,
        message.content
    )

    # 初始化變數
    user_message = None
    name = None

    try:
        # 嘗試處理附件
        if message.attachments:
            txt_file = next((att for att in message.attachments if att.filename.endswith('.txt')), None)
            if txt_file:
                # 下載和讀取 .txt 文件內容
                async with aiofiles.open(f"temp_{txt_file.filename}", mode="wb") as file:
                    await txt_file.save(file.name)
                async with aiofiles.open(f"temp_{txt_file.filename}", mode="r", encoding="utf-8") as file:
                    user_message = await file.read()
                os.remove(f"temp_{txt_file.filename}")
            else:
                await message.channel.send("未找到 .txt 文件，請重新上傳。")
                return

        # 如果沒有附件，解析文字訊息
        if not user_message:
            content_lines = message.content.splitlines()
            content = "\n".join(line.rstrip() for line in content_lines)
            first_line, *remaining_lines = content.split("\n", 1)
            parts = first_line.split(" ", 2)

            if len(parts) >= 3:
                _, name, *info = parts
                remaining_info = "\n".join(remaining_lines) if remaining_lines else ""
                user_message = f"{' '.join(info)}\n{remaining_info}".strip()
            else:
                await message.channel.send("訊息格式錯誤，請使用正確的格式或上傳 .txt 文件。")
                return

        # 確保 name 和 user_message 已被正確設置
        if not name:
            name = "4o"  # 提供一個安全的默認值
        if not user_message:
            await message.channel.send("未提供有效的訊息內容。")
            return

        # 轉換名稱
        converted_name = NAME_MAPPING.get(name, None)
        if not converted_name:
            await message.channel.send(f"未知的名稱：{name}")
            return

        # 獲取 OpenAI 回覆
        openai_reply = await fetch_openai_response(openai_client, converted_name, user_message, logger)

        # 發送回覆
        if len(openai_reply) <= 2000:
            await message.channel.send(openai_reply)
        else:
            file_path = await save_response_to_file(openai_reply)
            try:
                await message.channel.send(file=discord.File(file_path))
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

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
