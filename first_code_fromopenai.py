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
        logger.info("從 OpenAI 獲取回覆：%s", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenAI API 請求失敗: %s", e)
        return "抱歉，發生錯誤，無法獲取回覆。"

# 非同步函數：將回應內容儲存到唯一的文本檔案
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

# 非同步函數：下載附件並讀取內容
async def download_and_read_txt(attachment: discord.Attachment, logger: logging.Logger) -> str:
    """
    下載 Discord 附件並讀取其內容。

    :param attachment: Discord 附件對象。
    :param logger: 日誌紀錄器。
    :return: 附件中的文本內容。
    """
    try:
        if not attachment.filename.endswith('.txt'):
            logger.warning("收到非txt檔案：%s", attachment.filename)
            return ""
        
        # 生成唯一檔案名稱以避免競爭
        unique_filename = f"downloaded_{uuid.uuid4()}.txt"
        await attachment.save(fp=unique_filename)
        logger.info("下載附件完成：%s", unique_filename)

        async with aiofiles.open(unique_filename, mode="r", encoding="utf-8") as file:
            content = await file.read()
        
        # 刪除下載的檔案
        os.remove(unique_filename)
        logger.info("刪除下載的檔案：%s", unique_filename)
        return content
    except Exception as e:
        logger.error("下載或讀取附件失敗: %s", e)
        return ""

# 處理訊息的主要函數
async def handle_message(message: discord.Message, bot: commands.Bot, openai_client: AsyncOpenAI, allowed_channels: set, logger: logging.Logger):
    if message.author == bot.user:
        return

    # 獲取伺服器與頻道資訊
    guild_id = message.guild.id if message.guild else None
    channel_id = message.channel.id

    # 只處理 ALLOWED_CHANNELS 的訊息
    if not guild_id or (guild_id, channel_id) not in allowed_channels:
        return  # 忽略不在 ALLOWED_CHANNELS 的訊息

    # 獲取伺服器與頻道名稱
    guild_name = message.guild.name if message.guild else "DM"
    channel_name = message.channel.name if hasattr(message.channel, "name") else "Unknown"
    channel_type = "討論串" if isinstance(message.channel, discord.Thread) else "頻道"

    # 記錄收到的訊息
    logger.info(
        "[訊息記錄] 時間: %s, 伺服器: %s, %s: %s, 用戶: %s, 訊息: %s",
        message.created_at,
        guild_name,
        channel_type,
        channel_name,
        message.author.name,
        message.content
    )

    try:
        # 判斷是否為討論串
        is_thread = isinstance(message.channel, discord.Thread)

        # 刪除討論串邏輯
        if is_thread and message.content.strip() == "!del":
            thread_name = message.channel.name
            await message.channel.delete()
            logger.info(
                "[討論串刪除] 時間: %s, 討論串: %s, 由用戶: %s 觸發",
                message.created_at,
                thread_name,
                message.author.name
            )
            return

        # 處理附件中的txt檔案
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.endswith('.txt'):
                    txt_content = await download_and_read_txt(attachment, logger)
                    if txt_content:
                        logger.info("將附件內容傳送給 OpenAI：%s", txt_content[:100] + "..." if len(txt_content) > 100 else txt_content)
                        openai_reply = await fetch_openai_response(openai_client, "gpt-4", txt_content, logger)
                        
                        # 提取回覆的第一句，作為討論串名稱
                        first_sentence = openai_reply.split(".")[0].strip()
                        if len(first_sentence) > 100:  # Discord 討論串名稱限制
                            first_sentence = first_sentence[:97] + "..."
                        thread_name = first_sentence if first_sentence else f"討論：{NAME_MAPPING.get('unknown', 'unknown')}"

                        if len(openai_reply) <= 2000:
                            if is_thread:
                                await message.channel.send(openai_reply)
                            else:
                                thread = await message.create_thread(name=thread_name, auto_archive_duration=60)
                                await thread.send(openai_reply)
                        else:
                            file_path = await save_response_to_file(openai_reply)  # 生成唯一檔案名稱

                            try:
                                # 發送文件
                                if is_thread:
                                    await message.channel.send(file=discord.File(file_path))
                                else:
                                    thread = await message.create_thread(name=thread_name, auto_archive_duration=60)
                                    await thread.send(file=discord.File(file_path))
                            finally:
                                # 刪除文件
                                if os.path.exists(file_path):
                                    os.remove(file_path)
            # 當處理完附件後，返回以避免處理消息的其他部分
            return

        elif bot.user.mentioned_in(message):
            # 處理提及 Bot 的訊息
            # 解析用戶訊息
            content_lines = message.content.splitlines()
            content = "\n".join(line.rstrip() for line in content_lines)
            first_line, *remaining_lines = content.split("\n", 1)
            parts = first_line.split(" ", 2)
            if len(parts) < 3:
                await message.channel.send("訊息格式錯誤。請使用正確的格式。")
                logger.warning(
                    "[訊息格式錯誤] 時間: %s, 用戶: %s, 訊息: %s",
                    message.created_at,
                    message.author.name,
                    content
                )
                return
            _, name, *info = parts
            remaining_info = "\n".join(remaining_lines) if remaining_lines else ""
            user_message = f"{' '.join(info)}\n{remaining_info}".strip()

            # 轉換名稱
            converted_name = NAME_MAPPING.get(name, None)
            if not converted_name:
                await message.channel.send(f"未知的名稱：{name}")
                logger.warning(
                    "[未知名稱] 時間: %s, 名稱: %s, 訊息: %s",
                    message.created_at,
                    name,
                    content
                )
                return

            # 獲取 OpenAI 回覆
            openai_reply = await fetch_openai_response(openai_client, converted_name, user_message, logger)

            # 提取回覆的第一句，作為討論串名稱
            first_sentence = openai_reply.split(".")[0].strip()
            if len(first_sentence) > 100:  # Discord 討論串名稱限制
                first_sentence = first_sentence[:97] + "..."
            thread_name = first_sentence if first_sentence else f"討論：{converted_name}"

            if len(openai_reply) <= 2000:
                # 如果訊息長度小於等於 2000 字符，直接發送
                if is_thread:
                    await message.channel.send(openai_reply)
                else:
                    thread = await message.create_thread(name=thread_name, auto_archive_duration=60)
                    await thread.send(openai_reply)
            else:
                # 如果訊息過長，儲存到文件並發送
                file_path = await save_response_to_file(openai_reply)  # 生成唯一檔案名稱

                try:
                    # 發送文件
                    if is_thread:
                        await message.channel.send(file=discord.File(file_path))
                    else:
                        thread = await message.create_thread(name=thread_name, auto_archive_duration=60)
                        await thread.send(file=discord.File(file_path))
                finally:
                    # 刪除文件
                    if os.path.exists(file_path):
                        os.remove(file_path)
    except Exception as e:
            logger.error(
                "[錯誤記錄] 時間: %s, 伺服器: %s, 頻道: %s, 用戶: %s, 錯誤內容: %s",
                message.created_at,
                guild_name,
                channel_name,
                message.author.name,
                str(e)
            )

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
