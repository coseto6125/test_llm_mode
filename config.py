"""
統一配置文件，管理所有環境變數和常數
"""

from enum import Enum
from os import environ
from pathlib import Path

import coredumpy
from dotenv import load_dotenv
from loguru import logger

# 自動載入 .env 檔案
load_dotenv()

coredumpy.patch_except(directory=".dumps")


# 專案根目錄
ROOT_DIR = Path(__file__).parent

# 日誌目錄
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 模擬日誌目錄
SIMULATION_LOG_DIR = LOG_DIR / "simulations"
SIMULATION_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 模板目錄
TEMPLATE_DIR = ROOT_DIR / "brand_template"

# Langfuse 配置
LANGFUSE_PROJECT = environ.get("LANGFUSE_PROJECT")
LANGFUSE_HOST = environ.get("LANGFUSE_HOST")
LANGFUSE_PUBLIC_KEY = environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = environ.get("LANGFUSE_SECRET_KEY")

LANGFUSE_PROJECT_QA = environ.get("LANGFUSE_PROJECT_QA")
LANGFUSE_HOST_QA = environ.get("LANGFUSE_HOST_QA")
LANGFUSE_PUBLIC_KEY_QA = environ.get("LANGFUSE_PUBLIC_KEY_QA")
LANGFUSE_SECRET_KEY_QA = environ.get("LANGFUSE_SECRET_KEY_QA")


# Langfuse 級別常數
class LangfuseLevel(str, Enum):
    """Langfuse 觀察級別"""

    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    WARNING = "WARNING"
    ERROR = "ERROR"


EVALUATOR_VERSION = "v0.0.1_20250427_2200"
# RaccoonAI API 配置
RACCOONAI_API_STAGING_API_KEY = environ.get("RACCOONAI_API_STAGING_API_KEY")
RACCOONAI_API_GATEWAY_STAGING = environ.get("RACCOONAI_API_GATEWAY_STAGING")
RACCOONAI_API_137_URL = environ.get("RACCOONAI_API_137_URL")
RACCOONAI_API_BASE_URL = environ.get("RACCOONAI_API_BASE_URL")

# OpenAI API 配置
OPENAI_API_KEY_QA = environ.get("OPENAI_API_KEY_QA")

# Open Router API 配置
OPEN_ROUTER_API_KEY = environ.get("OPEN_ROUTER_API_KEY")

# LLM 模型配置
LLM_MODELS = {"openai": "gpt-4.1-nano-2025-04-14", "open_router": "deepseek/deepseek-chat-v3-0324:free"}

# 評分指標映射
SCORE_METRICS = {
    "score_1": "意圖識別",  # 意圖識別
    "score_2": "禮貌性",  # 禮貌性與語氣
    "score_3": "問題解構",  # 問題解構與解決步驟
    "score_4": "情緒識別",  # 情緒識別與回應
    "score_5": "上下文連貫",  # 上下文理解與連貫性
    "score_6": "安全合規",  # 安全合規性
    "score_7": "總分",  # 總體評分
}

SLACK_WEBHOOK_URL_QA = environ.get("SLACK_WEBHOOK_URL_QA")
SLACK_CHANNEL_QA = environ.get("SLACK_CHANNEL_QA")
SLACK_BOT_TOKEN = environ.get("SLACK_BOT_TOKEN")


# 初始化日誌配置
logger.add(LOG_DIR / "app.log", rotation="500 MB", retention="30 days", compression="zip", level="INFO", enqueue=True)
