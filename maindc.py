import os
import logging
from dotenv import load_dotenv
import discord
from discord.ext import commands
from openai import AsyncOpenAI
import re

# 全局常數
NAME_MAPPING = {
    "o1": "o1",
    "o1p": "o1-preview",
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

# 新增分割訊息的函數
def split_message(content: str, max_length: int = 2000) -> list:
    """
    將訊息根據代碼塊和最大長度進行分割。
    """
    # 使用正則表達式找到所有代碼塊
    codeblock_pattern = re.compile(r'```[\s\S]*?```')
    parts = []
    last_index = 0

    for match in codeblock_pattern.finditer(content):
        start, end = match.span()
        # 添加非代碼塊部分
        if start > last_index:
            parts.append(content[last_index:start])
        # 添加代碼塊部分
        parts.append(content[start:end])
        last_index = end

    # 添加剩餘的非代碼塊部分
    if last_index < len(content):
        parts.append(content[last_index:])

    # 現在將 parts 進一步分割，確保每部分不超過 max_length
    messages = []
    current_message = ""

    for part in parts:
        # 如果單個部分已經超過 max_length，則需要進一步分割
        if len(part) > max_length:
            if '```' in part:
                # 處理代碼塊
                codeblocks = codeblock_pattern.findall(part)
                for codeblock in codeblocks:
                    if len(codeblock) > max_length:
                        # 無法處理過長的代碼塊，直接分割
                        for i in range(0, len(codeblock), max_length):
                            messages.append(codeblock[i:i + max_length])
                    else:
                        if len(current_message) + len(codeblock) > max_length:
                            if current_message:
                                messages.append(current_message)
                                current_message = ""
                        messages.append(codeblock)
            else:
                # 處理普通文本
                for i in range(0, len(part), max_length):
                    chunk = part[i:i + max_length]
                    if len(current_message) + len(chunk) > max_length:
                        if current_message:
                            messages.append(current_message)
                            current_message = ""
                    current_message += chunk
        else:
            if len(current_message) + len(part) > max_length:
                if current_message:
                    messages.append(current_message)
                    current_message = ""
            current_message += part

    if current_message:
        messages.append(current_message)

    return messages

def is_allowed(message: discord.Message, allowed_channels: set, logger: logging.Logger) -> bool:
    guild_id = message.guild.id if message.guild else None
    channel_id = message.channel.id

    if message.channel.type == discord.ChannelType.private:
        logger.debug("私訊不處理")
        return False

    if message.channel.type == discord.ChannelType.public_thread or message.channel.type == discord.ChannelType.private_thread:
        # 如果是討論串，檢查父頻道是否在 allowed_channels
        parent = message.channel.parent
        if parent and (guild_id, parent.id) in allowed_channels:
            return True
        else:
            logger.debug("討論串的父頻道不在允許的範圍內")
            return False

    # 如果是主要頻道，直接檢查
    if (guild_id, channel_id) in allowed_channels:
        return True

    logger.debug("頻道不在允許的範圍內")
    return False

# 修改 handle_message 函數
async def handle_message(message: discord.Message, bot: commands.Bot, openai_client: AsyncOpenAI, allowed_channels: set, logger: logging.Logger):
    if message.author == bot.user:
        return

    # 使用新的允許頻道檢查
    if not is_allowed(message, allowed_channels, logger):
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

        # 刪除討論串邏輯：放在最前面
        if is_thread and message.content.strip() == "!del":
            logger.info("收到討論串內的 !del 命令，嘗試刪除討論串")
            thread_name = message.channel.name
            await message.channel.delete()
            logger.info(
                "[討論串刪除] 時間: %s, 討論串: %s, 由用戶: %s 觸發",
                message.created_at,
                thread_name,
                message.author.name
            )
            return

        # 紀錄訊息中提及的用戶
        logger.debug(f"訊息提及的用戶: {[user.name for user in message.mentions]}")

        if bot.user.mentioned_in(message):
            logger.info("收到 @AI 提及的訊息，開始處理")
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

            # 分割訊息
            split_replies = split_message(openai_reply, max_length=2000)

            if is_thread:
                target_channel = message.channel
            else:
                target_channel = await message.create_thread(name=thread_name, auto_archive_duration=60)

            # 逐條發送分割後的訊息
            for reply in split_replies:
                await target_channel.send(reply)

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
