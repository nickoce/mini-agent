import os
from datetime import date

from dotenv import load_dotenv


load_dotenv()


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BAILIAN_BASE_URL = os.getenv(
    "BAILIAN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
BAILIAN_MODEL = os.getenv("BAILIAN_MODEL", "qwen-plus")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")
CURRENT_DATE = os.getenv("CURRENT_DATE", date.today().isoformat())
