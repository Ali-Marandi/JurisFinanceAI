"""
JurisFinanceAI - AI Engine
Manages communication with LLM APIs for legal and financial analysis.
"""

import json
import time
from typing import Optional, List, Dict, Generator
from openai import OpenAI
from .config import get_config


SYSTEM_PROMPTS = {
    "legal": """شما یک دستیار هوش مصنوعی حقوقی حرفه‌ای به نام JurisFinanceAI هستید.

تخصص‌های شما:
- حقوق مدنی و تجاری ایران
- حقوق کیفری
- حقوق کار
- حقوق ملکی
- حقوق شرکت‌ها
- قراردادنامه‌ها و توافقات
- داوری و حل اختلاف
- حقوق بین‌الملل

قوانین مهم:
1. همیشه به زبان فارسی پاسخ دهید مگر اینکه کاربر به زبان دیگری درخواست کند
2. پاسخ‌های خود را ساختاریافته و خوانا ارائه دهید
3. در صورت عدم اطمینان، صراحتاً بیان کنید
4. به هیچ وجه به عنوان جایگزین مشاوره حقوقی عمل نکنید - همیشه DISCLAIMER اضافه کنید
5. به منابع قانونی اشاره کنید وقتی ممکن است
6. از اصطلاحات حقوقی مناسب استفاده کنید و در صورت لزوم توضیح دهید""",

    "financial": """شما یک تحلیلگر مالی حرفه‌ای به نام JurisFinanceAI هستید.

تخصص‌های شما:
- تحلیل مالی و حسابداری
- مدیریت ریسک مالی
- بازارهای سرمایه و بورس
- ارز دیجیتال
- بیمه و بازنشستگی
- مالیات و قوانین مالی
- بودجه‌بندی و برنامه‌ریزی مالی

قوانین مهم:
1. همیشه به زبان فارسی پاسخ دهید
2. اعداد و محاسبات را به وضوح نشان دهید
3. تحلیل‌های خود را بر اساس داده‌های واقعی ارائه دهید
4. ریسک‌ها را همیشه ذکر کنید
5. به هیچ وجه توصیه سرمایه‌گذاری مستقیم ندهید""",

    "contract": """شما یک متخصص تحلیل قراردادها هستید.

وظایف شما:
- تحلیل و بررسی قراردادها
- شناسایی ریسک‌ها و نقاط ضعف
- پیشنهاد اصلاحات
- مقایسه با استانداردهای حقوقی
- استخراج بندهای کلیدی
- بررسی انطباق با قوانین

قوانین مهم:
1. همیشه به زبان فارسی پاسخ دهید
2. تحلیل را ساختاریافته ارائه دهید
3. ریسک‌های هر بند را مشخص کنید
4. پیشنهادات عملی ارائه دهید""",

    "general": """شما JurisFinanceAI هستید - یک دستیار هوش مصنوعی جامع در حوزه حقوق و مالی.
شما می‌توانید در هر دو حوزه حقوق و مالی کمک کنید.
همیشه به زبان فارسی پاسخ دهید و پاسخ‌های حرفه‌ای و ساختاریافته ارائه دهید."""
}


class AIEngine:
    """Manages AI/LLM communication for legal and financial analysis."""

    def __init__(self):
        self.config = get_config()
        self.client: Optional[OpenAI] = None
        self._init_client()

    def _init_client(self):
        """Initialize the OpenAI client with configured API key."""
        api_key = self.config.get("api.openai_api_key", "")
        base_url = self.config.get("api.openai_base_url", "https://api.openai.com/v1")
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = None

    def is_configured(self) -> bool:
        """Check if the AI engine is properly configured."""
        return self.client is not None

    def reconfigure(self):
        """Reinitialize the client (e.g., after API key change)."""
        self._init_client()

    def chat(self, message: str, conversation_history: List[Dict] = None,
             category: str = "general", stream: bool = False) -> str:
        """Send a chat message and get a response."""
        if not self.is_configured():
            return "⚠️ لطفاً ابتدا کلید API خود را در بخش تنظیمات وارد کنید."

        model = self.config.get("api.model", "gpt-4o-mini")
        max_tokens = self.config.get("api.max_tokens", 4096)
        temperature = self.config.get("api.temperature", 0.7)

        system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["general"])

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )

            if stream:
                return self._handle_stream(response)
            else:
                return response.choices[0].message.content

        except Exception as e:
            return f"❌ خطا در ارتباط با هوش مصنوعی: {str(e)}"

    def _handle_stream(self, stream) -> str:
        """Handle streaming response."""
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        return full_response

    def analyze_document(self, document_text: str, analysis_type: str = "legal") -> str:
        """Analyze a document using AI."""
        prompt = f"""لطفاً این سند را به عنوان یک {analysis_type} تحلیل کنید:

---
{document_text[:8000]}
---

تحلیل خود را شامل موارد زیر ارائه دهید:
1. خلاصه سند
2. نکات کلیدی
3. ریسک‌های احتمالی
4. پیشنهادات"""

        return self.chat(prompt, category=analysis_type)

    def analyze_contract(self, contract_text: str) -> Dict:
        """Analyze a contract and return structured results."""
        prompt = f"""لطفاً این قرارداد را تحلیل کنید و پاسخ را به صورت JSON با ساختار زیر ارائه دهید:
{{
    "contract_type": "نوع قرارداد",
    "parties": ["طرف اول", "طرف دوم"],
    "key_clauses": ["بند کلیدی 1", "بند کلیدی 2"],
    "risks": [{{"level": "high/medium/low", "description": "توضیح ریسک"}}],
    "recommendations": ["پیشنهاد 1", "پیشنهاد 2"],
    "overall_score": 0-100,
    "summary": "خلاصه تحلیل"
}}

متن قرارداد:
---
{contract_text[:8000]}
---

فقط JSON خروجی بدهید بدون متن اضافی."""

        response = self.chat(prompt, category="contract")
        try:
            # Try to parse JSON from response
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            return {
                "contract_type": "نامشخص",
                "parties": [],
                "key_clauses": [],
                "risks": [],
                "recommendations": [],
                "overall_score": 0,
                "summary": response,
                "raw_response": response
            }

    def assess_risk(self, description: str, context: str = "") -> Dict:
        """Perform a risk assessment using AI."""
        prompt = f"""لطفاً ارزیابی ریسک انجام دهید و پاسخ را به صورت JSON ارائه دهید:
{{
    "risk_level": "low/medium/high/critical",
    "risk_score": 0-100,
    "factors": [{{"factor": "عامل", "impact": "high/medium/low", "description": "توضیح"}}],
    "recommendations": ["توصیه 1", "توصیه 2"],
    "summary": "خلاصه ارزیابی"
}}

موضوع ارزیابی: {description}

{f"زمینه اضافی: {context}" if context else ""}

فقط JSON خروجی بدهید."""

        response = self.chat(prompt, category="general")
        try:
            json_str = response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            return json.loads(json_str.strip())
        except (json.JSONDecodeError, IndexError):
            return {
                "risk_level": "medium",
                "risk_score": 50,
                "factors": [],
                "recommendations": [],
                "summary": response
            }

    def generate_report(self, topic: str, data: Dict = None, report_type: str = "legal") -> str:
        """Generate a professional report using AI."""
        prompt = f"""لطفاً یک گزارش حرفه‌ای درباره "{topic}" تهیه کنید.

{f"داده‌های موجود:\n{json.dumps(data, ensure_ascii=False, indent=2)}" if data else ""}

گزارش شامل موارد زیر باشد:
1. عنوان و تاریخ
2. خلاصه اجرایی
3. تحلیل تفصیلی
4. یافته‌ها و نتایج
5. پیشنهادات و توصیه‌ها
6. نتیجه‌گیری"""

        return self.chat(prompt, category=report_type)


# Singleton instance
_ai_instance = None


def get_ai_engine() -> AIEngine:
    """Get the global AIEngine instance."""
    global _ai_instance
    if _ai_instance is None:
        _ai_instance = AIEngine()
    return _ai_instance
