# Reference: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/openai.py

try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("إذا كنت تريد استخدام نماذج Google Gemini، يرجى تثبيت حزمة google-generativeai عن طريق تشغيل `pip install google-generativeai`، وأضف 'GEMINI_API_KEY' إلى متغيرات البيئة.")

import os
import json
import base64
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

# تعريف نموذج افتراضي للاستجابات
class DefaultFormat(BaseModel):
    response: str

# تعريف النماذج التي تدعم الاستجابات المنظمة في Gemini
GEMINI_STRUCTURED_MODELS = ['gemini-1.5-flash', 'gemini-1.5-pro']

class ChatGemini(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string="gemini-1.5-flash",  # نموذج افتراضي لـ Gemini
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        enable_cache: bool=True,
        **kwargs):
        """
        تهيئة الفئة باستخدام نموذج Gemini.
        :param model_string: اسم النموذج (مثل gemini-1.5-flash)
        :param system_prompt: المطالبة النظامية
        :param is_multimodal: دعم المدخلات متعددة الوسائط
        :param enable_cache: تفعيل التخزين المؤقت
        """
        if enable_cache:
            root = platformdirs.user_cache_dir("octotools")
            cache_path = os.path.join(root, f"cache_gemini_{model_string}.db")
            self.image_cache_dir = os.path.join(root, "image_cache")
            os.makedirs(self.image_cache_dir, exist_ok=True)
            super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        if os.getenv("GEMINI_API_KEY") is None:
            raise ValueError("يرجى تعيين متغير البيئة GEMINI_API_KEY إذا كنت تريد استخدام نماذج Google Gemini.")
        
        # تهيئة Gemini API
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_string = model_string
        self.is_multimodal = is_multimodal
        self.enable_cache = enable_cache

        if enable_cache:
            print(f"!! تم تفعيل التخزين المؤقت للنموذج: {self.model_string}")
        else:
            print(f"!! تم تعطيل التخزين المؤقت للنموذج: {self.model_string}")

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt=None, **kwargs):
        """
        إنشاء استجابة بناءً على المدخلات.
        :param content: نص أو قائمة تحتوي على نصوص أو صور
        :param system_prompt: المطالبة النظامية (اختياري)
        """
        try:
            attempt_number = self.generate.retry.statistics.get('attempt_number', 0) + 1
            if attempt_number > 1:
                print(f"محاولة {attempt_number} من 5")

            if isinstance(content, str):
                return self._generate_text(content, system_prompt=system_prompt, **kwargs)
            elif isinstance(content, list):
                if not self.is_multimodal:
                    raise NotImplementedError("التوليد متعدد الوسائط مدعوم فقط للنماذج متعددة الوسائط.")
                return self._generate_multimodal(content, system_prompt=system_prompt, **kwargs)

        except Exception as e:
            print(f"خطأ في طريقة generate: {str(e)}")
            print(f"نوع الخطأ: {type(e).__name__}")
            print(f"تفاصيل الخطأ: {e.args}")
            return {
                "error": type(e).__name__,
                "message": str(e),
                "details": getattr(e, 'args', None)
            }

    def _generate_text(
        self, prompt, system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None
    ):
        """
        إنشاء نص باستخدام Gemini API.
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        if self.enable_cache:
            cache_key = sys_prompt_arg + prompt
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        model = genai.GenerativeModel(self.model_string)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }

        if self.model_string in GEMINI_STRUCTURED_MODELS and response_format is not None:
            # دعم الاستجابات المنظمة (JSON)
            prompt_with_format = f"{sys_prompt_arg}\n{prompt}\n\nRespond in JSON format according to this schema: {response_format.schema_json()}"
            response = model.generate_content(
                prompt_with_format,
                generation_config=generation_config,
                response_mime_type="application/json"
            )
            response_text = response.text
            response_data = json.loads(response_text)
            response = response_format(**response_data)
        else:
            response = model.generate_content(
                f"{sys_prompt_arg}\n{prompt}",
                generation_config=generation_config
            )
            response = response.text

        if self.enable_cache:
            self._save_cache(cache_key, response)
        return response

    def __call__(self, prompt, **kwargs):
        """استدعاء الطريقة generate مباشرة."""
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[Union[str, dict]]:
        """
        تنسيق المدخلات متعددة الوسائط لـ Gemini.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                formatted_content.append({
                    "mime_type": "image/jpeg",  # افتراض أن الصور بصيغة JPEG
                    "data": base64.b64encode(item).decode('utf-8')
                })
            elif isinstance(item, str):
                formatted_content.append(item)
            else:
                raise ValueError(f"نوع مدخل غير مدعوم: {type(item)}")
        return formatted_content

    def _generate_multimodal(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=4000, top_p=0.99, response_format=None
    ):
        """
        إنشاء استجابة لمدخلات متعددة الوسائط.
        """
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        if self.enable_cache:
            cache_key = sys_prompt_arg + json.dumps(formatted_content)
            cache_or_none = self._check_cache(cache_key)
            if cache_or_none is not None:
                return cache_or_none

        model = genai.GenerativeModel(self.model_string)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }

        formatted_content.insert(0, sys_prompt_arg)  # إضافة المطالبة النظامية في البداية
        if self.model_string in GEMINI_STRUCTURED_MODELS and response_format is not None:
            formatted_content.append(f"\n\nRespond in JSON format according to this schema: {response_format.schema_json()}")
            response = model.generate_content(
                formatted_content,
                generation_config=generation_config,
                response_mime_type="application/json"
            )
            response_text = response.text
            response_data = json.loads(response_text)
            response_text = response_format(**response_data)
        else:
            response = model.generate_content(
                formatted_content,
                generation_config=generation_config
            )
            response_text = response.text

        if self.enable_cache:
            self._save_cache(cache_key, response_text)
        return response_text
