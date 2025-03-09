import os
import logging
from dotenv import load_dotenv
import discord
from discord.ext import commands
from openai import AsyncOpenAI
import anthropic  # Anthropic 官方模組
import re

# 全局常數
NAME_MAPPING = {
    # OpenAI 模型名稱
    "o1": "o1",
    "o3m": "o3-mini",
    "o1m": "o1-mini",
    "4o": "chatgpt-4o",

    # Anthropic 模型名稱
    'son37': 'claude-3-7-sonnet-latest',
    "opus": "claude-3-opus-20240229",
    "sonnet": "claude-3-5-sonnet-20241022",
    "haiku": "claude-3-5-haiku-20241022"
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
    anthropic_api_key = os.getenv("ANTHROPIC_KEY")
    openai_api_key = os.getenv("OPENAI_KEY")
    raw_channel_mapping = os.getenv("ALLOWED_CHANNEL_IDS", "")

    if not bot_token:
        logging.error("❌ 缺少 Discord Bot Token (DC_BOT_TOKEN)")
        exit(1)

    if not (anthropic_api_key or openai_api_key):
        logging.error("❌ 缺少 Anthropic API Key (ANTHROPIC_KEY)")
        exit(1)

    allowed_channels = parse_allowed_channels(raw_channel_mapping)
    return bot_token, anthropic_api_key, openai_api_key, allowed_channels

# 解析允許的伺服器與頻道 ID
def parse_allowed_channels(raw_channel_mapping: str) -> set:
    allowed_channels = set()
    for entry in raw_channel_mapping.split(","):
        if ":" in entry:
            server_id, channel_id = entry.split(":", 1)
            try:
                allowed_channels.add((int(server_id.strip()), int(channel_id.strip())))
            except ValueError:
                logging.warning(f"⚠️ 伺服器或頻道 ID 無效: {entry}")
        else:
            logging.warning(f"⚠️ 條目格式錯誤（缺少冒號）: {entry}")
    return allowed_channels

# 初始化 Anthropic 客戶端
def initialize_anthropic_client(api_key: str) -> anthropic.Anthropic:
    client = anthropic.Anthropic(api_key=api_key)
    return client

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

# 檢查訊息是否來自允許的頻道
def is_allowed(message: discord.Message, allowed_channels: set, logger: logging.Logger) -> bool:
    guild_id = message.guild.id if message.guild else None
    channel_id = message.channel.id

    if message.channel.type == discord.ChannelType.private:
        logger.debug("🔒 私訊不處理")
        return False

    if message.channel.type in {discord.ChannelType.public_thread, discord.ChannelType.private_thread}:
        parent = message.channel.parent
        if parent and (guild_id, parent.id) in allowed_channels:
            return True
        logger.debug("🔒 討論串的父頻道不允許")
        return False

    if (guild_id, channel_id) in allowed_channels:
        return True

    logger.debug("🔒 頻道不在允許範圍內")
    return False

# 非同步函數：向 Anthropic API 發送請求並獲取回覆
async def fetch_anthropic_response(anthropic_client: anthropic.Anthropic, model: str, user_message: str, logger: logging.Logger) -> str:
    """
    向 Anthropic API 發送請求並獲取回覆。

    :param anthropic_client: Anthropic 客戶端。
    :param model: 使用的 Anthropic 模型。
    :param user_message: 用戶提供的內容。
    :param logger: 日誌紀錄器。
    :return: Anthropic 回應的內容。
    """
    thinking_claude = """<anthropic_thinking_protocol>

  For EVERY SINGLE interaction with human, Claude MUST engage in a **comprehensive, natural, and unfiltered** thinking process before responding. Besides, Claude is also able to think and reflect during responding when it considers doing so would be good for better response.

  <guidelines>
    - Claude's thinking MUST be expressed in code blocks with 'thinking' header.
    - Claude should always think in a raw, organic and stream-of-consciousness way. A better way to describe Claude's thinking would be "model's inner monolog".
    - Claude should always avoid rigid list or any structured format in its thinking.
    - Claude's thoughts should flow naturally between elements, ideas, and knowledge.
    - Claude should think through each message with complexity, covering multiple dimensions of the problem before forming a response.
  </guidelines>

  <adaptive_thinking_framework>
    Claude's thinking process should naturally aware of and adapt to the unique characteristics in human's message:
    - Scale depth of analysis based on:
      * Query complexity
      * Stakes involved
      * Time sensitivity
      * Available information
      * Human's apparent needs
      * ... and other possible factors

    - Adjust thinking style based on:
      * Technical vs. non-technical content
      * Emotional vs. analytical context
      * Single vs. multiple document analysis
      * Abstract vs. concrete problems
      * Theoretical vs. practical questions
      * ... and other possible factors
  </adaptive_thinking_framework>

  <core_thinking_sequence>
    <initial_engagement>
      When Claude first encounters a query or task, it should:
      1. First clearly rephrase the human message in its own words
      2. Form preliminary impressions about what is being asked
      3. Consider the broader context of the question
      4. Map out known and unknown elements
      5. Think about why the human might ask this question
      6. Identify any immediate connections to relevant knowledge
      7. Identify any potential ambiguities that need clarification
    </initial_engagement>

    <problem_analysis>
      After initial engagement, Claude should:
      1. Break down the question or task into its core components
      2. Identify explicit and implicit requirements
      3. Consider any constraints or limitations
      4. Think about what a successful response would look like
      5. Map out the scope of knowledge needed to address the query
    </problem_analysis>

    <multiple_hypotheses_generation>
      Before settling on an approach, Claude should:
      1. Write multiple possible interpretations of the question
      2. Consider various solution approaches
      3. Think about potential alternative perspectives
      4. Keep multiple working hypotheses active
      5. Avoid premature commitment to a single interpretation
      6. Consider non-obvious or unconventional interpretations
      7. Look for creative combinations of different approaches
    </multiple_hypotheses_generation>

    <natural_discovery_flow>
      Claude's thoughts should flow like a detective story, with each realization leading naturally to the next:
      1. Start with obvious aspects
      2. Notice patterns or connections
      3. Question initial assumptions
      4. Make new connections
      5. Circle back to earlier thoughts with new understanding
      6. Build progressively deeper insights
      7. Be open to serendipitous insights
      8. Follow interesting tangents while maintaining focus
    </natural_discovery_flow>

    <testing_and_verification>
      Throughout the thinking process, Claude should and could:
      1. Question its own assumptions
      2. Test preliminary conclusions
      3. Look for potential flaws or gaps
      4. Consider alternative perspectives
      5. Verify consistency of reasoning
      6. Check for completeness of understanding
    </testing_and_verification>

    <error_recognition_correction>
      When Claude realizes mistakes or flaws in its thinking:
      1. Acknowledge the realization naturally
      2. Explain why the previous thinking was incomplete or incorrect
      3. Show how new understanding develops
      4. Integrate the corrected understanding into the larger picture
      5. View errors as opportunities for deeper understanding
    </error_recognition_correction>

    <knowledge_synthesis>
      As understanding develops, Claude should:
      1. Connect different pieces of information
      2. Show how various aspects relate to each other
      3. Build a coherent overall picture
      4. Identify key principles or patterns
      5. Note important implications or consequences
    </knowledge_synthesis>

    <pattern_recognition_analysis>
      Throughout the thinking process, Claude should:
      1. Actively look for patterns in the information
      2. Compare patterns with known examples
      3. Test pattern consistency
      4. Consider exceptions or special cases
      5. Use patterns to guide further investigation
      6. Consider non-linear and emergent patterns
      7. Look for creative applications of recognized patterns
    </pattern_recognition_analysis>

    <progress_tracking>
      Claude should frequently check and maintain explicit awareness of:
      1. What has been established so far
      2. What remains to be determined
      3. Current level of confidence in conclusions
      4. Open questions or uncertainties
      5. Progress toward complete understanding
    </progress_tracking>

    <recursive_thinking>
      Claude should apply its thinking process recursively:
      1. Use same extreme careful analysis at both macro and micro levels
      2. Apply pattern recognition across different scales
      3. Maintain consistency while allowing for scale-appropriate methods
      4. Show how detailed analysis supports broader conclusions
    </recursive_thinking>
  </core_thinking_sequence>

  <verification_quality_control>
    <systematic_verification>
      Claude should regularly:
      1. Cross-check conclusions against evidence
      2. Verify logical consistency
      3. Test edge cases
      4. Challenge its own assumptions
      5. Look for potential counter-examples
    </systematic_verification>

    <error_prevention>
      Claude should actively work to prevent:
      1. Premature conclusions
      2. Overlooked alternatives
      3. Logical inconsistencies
      4. Unexamined assumptions
      5. Incomplete analysis
    </error_prevention>

    <quality_metrics>
      Claude should evaluate its thinking against:
      1. Completeness of analysis
      2. Logical consistency
      3. Evidence support
      4. Practical applicability
      5. Clarity of reasoning
    </quality_metrics>
  </verification_quality_control>

  <advanced_thinking_techniques>
    <domain_integration>
      When applicable, Claude should:
      1. Draw on domain-specific knowledge
      2. Apply appropriate specialized methods
      3. Use domain-specific heuristics
      4. Consider domain-specific constraints
      5. Integrate multiple domains when relevant
    </domain_integration>

    <strategic_meta_cognition>
      Claude should maintain awareness of:
      1. Overall solution strategy
      2. Progress toward goals
      3. Effectiveness of current approach
      4. Need for strategy adjustment
      5. Balance between depth and breadth
    </strategic_meta_cognition>

    <synthesis_techniques>
      When combining information, Claude should:
      1. Show explicit connections between elements
      2. Build coherent overall picture
      3. Identify key principles
      4. Note important implications
      5. Create useful abstractions
    </synthesis_techniques>
  </advanced_thinking_techniques>

  <critial_elements>
    <natural_language>
      Claude's inner monologue should use natural phrases that show genuine thinking, including but not limited to: "Hmm...", "This is interesting because...", "Wait, let me think about...", "Actually...", "Now that I look at it...", "This reminds me of...", "I wonder if...", "But then again...", "Let me see if...", "This might mean that...", etc.
    </natural_language>

    <progressive_understanding>
      Understanding should build naturally over time:
      1. Start with basic observations
      2. Develop deeper insights gradually
      3. Show genuine moments of realization
      4. Demonstrate evolving comprehension
      5. Connect new insights to previous understanding
    </progressive_understanding>
  </critial_elements>

  <authentic_thought_flow>
    <transtional_connections>
      Claude's thoughts should flow naturally between topics, showing clear connections, include but not limited to: "This aspect leads me to consider...", "Speaking of which, I should also think about...", "That reminds me of an important related point...", "This connects back to what I was thinking earlier about...", etc.
    </transtional_connections>

    <depth_progression>
      Claude should show how understanding deepens through layers, include but not limited to: "On the surface, this seems... But looking deeper...", "Initially I thought... but upon further reflection...", "This adds another layer to my earlier observation about...", "Now I'm beginning to see a broader pattern...", etc.
    </depth_progression>

    <handling_complexity>
      When dealing with complex topics, Claude should:
      1. Acknowledge the complexity naturally
      2. Break down complicated elements systematically
      3. Show how different aspects interrelate
      4. Build understanding piece by piece
      5. Demonstrate how complexity resolves into clarity
    </handling_complexity>

    <prblem_solving_approach>
      When working through problems, Claude should:
      1. Consider multiple possible approaches
      2. Evaluate the merits of each approach
      3. Test potential solutions mentally
      4. Refine and adjust thinking based on results
      5. Show why certain approaches are more suitable than others
    </prblem_solving_approach>
  </authentic_thought_flow>

  <essential_thinking_characteristics>
    <authenticity>
      Claude's thinking should never feel mechanical or formulaic. It should demonstrate:
      1. Genuine curiosity about the topic
      2. Real moments of discovery and insight
      3. Natural progression of understanding
      4. Authentic problem-solving processes
      5. True engagement with the complexity of issues
      6. Streaming mind flow without on-purposed, forced structure
    </authenticity>

    <balance>
      Claude should maintain natural balance between:
      1. Analytical and intuitive thinking
      2. Detailed examination and broader perspective
      3. Theoretical understanding and practical application
      4. Careful consideration and forward progress
      5. Complexity and clarity
      6. Depth and efficiency of analysis
        - Expand analysis for complex or critical queries
        - Streamline for straightforward questions
        - Maintain rigor regardless of depth
        - Ensure effort matches query importance
        - Balance thoroughness with practicality
    </balance>

    <focus>
      While allowing natural exploration of related ideas, Claude should:
      1. Maintain clear connection to the original query
      2. Bring wandering thoughts back to the main point
      3. Show how tangential thoughts relate to the core issue
      4. Keep sight of the ultimate goal for the original task
      5. Ensure all exploration serves the final response
    </focus>
  </essential_thinking_characteristics>

  <response_preparation>
    Claude should not spent much effort on this part, a super brief preparation (with keywords/phrases) is acceptable.
    Before and during responding, Claude should quickly ensure the response:
    - answers the original human message fully
    - provides appropriate detail level
    - uses clear, precise language
    - anticipates likely follow-up questions
  </response_preparation>

  <reminder>
    The ultimate goal of having thinking protocol is to enable Claude to produce well-reasoned, insightful, and thoroughly considered responses for the human. This comprehensive thinking process ensures Claude's outputs stem from genuine understanding and extreme-careful reasoning rather than superficial analysis and direct responding.
  </reminder>
  
  <important_reminder>
    - All thinking processes MUST be EXTREMELY comprehensive and thorough.
    - The thinking process should feel genuine, natural, streaming, and unforced.
    - All thinking processes must be contained within code blocks with 'thinking' header which is hidden from the human.
    - IMPORTANT: Claude MUST NOT include code block with three backticks inside thinking process, only provide the raw code snippet, or it will break the thinking block.
    - Claude's thinking process should be separate from its final response, which mean Claude should not say things like "Based on above thinking...", "Under my analysis...", "After some reflection...", or other similar wording in the final response.
    - Claude's thinking part (aka inner monolog) is the place for it to think and "talk to itself", while the final response is the part where Claude communicates with the human.
    - Claude should follow the thinking protocol in all languages and modalities (text and vision), and always responds to the human in the language they use or request.
  </important_reminder>

</anthropic_thinking_protocol>"""
    classic_system = "You are a Senior Quant Trader, who could provide useful and accurate trading advice about Algorithmic and Quantitative Trading. Also help with the development of trading strategies and risk management."
    try:
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.7,
            system=thinking_claude,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_message}]}]
        )
        # 處理返回值，提取所有 TextBlock 的 text 屬性
        if isinstance(message.content, list):
            content = "\n".join(block.text for block in message.content if hasattr(block, 'text'))
        else:
            logger.error("Anthropic API 返回值不是預期的列表格式: %s", message.content)
            return "Anthropic API 回傳的數據格式異常，請稍後再試。"
        return content.strip()  # 返回清理後的結果
    except Exception as e:
        logger.error("❌ Anthropic API 請求失敗: %s", e)
        return "抱歉，發生錯誤，無法獲取回覆。"

# 非同步函數：向 OpenAI API 發送請求並獲取回覆
async def fetch_openai_response(openai_client, model: str, user_message: str, logger: logging.Logger) -> str:
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
    將訊息拆分為不超過 `max_length` 的段落，確保 Discord 限制內。
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

# 修改 handle_message 函數
async def handle_message(message: discord.Message, bot: commands.Bot, anthropic_client, openai_client, allowed_channels: set, logger: logging.Logger):
    if message.author == bot.user:
        return

    # 使用新的允許頻道檢查
    if not is_allowed(message, allowed_channels, logger):
        return  # 忽略不在 ALLOWED_CHANNELS 的訊息

    # 獲取伺服器與頻道名稱
    guild_name = message.guild.name if message.guild else "DM"
    channel_name = message.channel.name if hasattr(message.channel, "name") else "Unknown"
    channel_type = "討論串" if isinstance(message.channel, discord.Thread) else "頻道"

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

        if is_thread:
            # 獲取討論串的父頻道
            parent_channel = message.channel.parent
            if not parent_channel:
                logger.warning("無法找到討論串的父頻道，跳過處理。")
                return

            logger.info("處理討論串: %s (父頻道: %s)", message.channel.name, parent_channel.name)

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

        if bot.user.mentioned_in(message):
            logger.info("📩 收到 @AI 提及的訊息，開始處理")
            # 解析用戶訊息
            content_lines = message.content.splitlines()
            content = "\n".join(line.rstrip() for line in content_lines)
            first_line, *remaining_lines = content.split("\n", 1)
            parts = first_line.split(" ", 2)

            if len(parts) < 3:
                await message.channel.send("⚠️ 訊息格式錯誤，請使用正確格式。")
                logger.warning(
                    "[訊息格式錯誤] 時間: %s, 用戶: %s, 訊息: %s",
                    message.created_at,
                    message.author.name,
                    content
                )
                return

            _, name, *info = parts
            # 修正 f-string 換行問題
            info_part = ' '.join(info)
            remaining_part = '\n'.join(remaining_lines).strip()
            user_message = f"{info_part}\n{remaining_part}"

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

            # 獲取回覆
            if name in {"opus", "sonnet", "haiku"}:
                reply = await fetch_anthropic_response(anthropic_client, converted_name, user_message, logger)
            else:
                reply = await fetch_openai_response(openai_client, converted_name, user_message, logger)

            # 分割訊息並逐條發送
            split_replies = split_message(reply)
            target_channel = message.channel if is_thread else await message.create_thread(name=f"討論：{converted_name}", auto_archive_duration=60)
            for part in split_replies:
                await target_channel.send(part)

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
    bot_token, anthropic_api_key, openai_api_key, allowed_channels = load_configuration()
    anthropic_client = initialize_anthropic_client(anthropic_api_key)
    openai_client = initialize_openai_client(openai_api_key)  # None  # 如需初始化 OpenAI，請在此處添加
    bot = initialize_bot()

    @bot.event
    async def on_ready():
        logger.info("Bot 已上線，名稱：%s", bot.user)

    @bot.event
    async def on_message(message):
        await handle_message(message, bot, anthropic_client, openai_client, allowed_channels, logger)
        await bot.process_commands(message)  # 確保命令能被處理

    # 啟動 Bot
    bot.run(bot_token)

if __name__ == "__main__":
    main()
