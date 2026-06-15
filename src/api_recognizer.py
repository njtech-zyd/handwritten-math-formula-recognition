"""
API Recognizer — 封装多模态大模型的 API 调用，识别图像中的手写数学公式。

设计为抽象接口模式：
- BaseRecognizer 定义统一接口
- DeepSeekRecognizer 提供具体实现（基于 OpenAI 兼容 SDK）
- 可扩展为其他模型实现
"""

import base64
import io
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from PIL import Image

# 默认 Prompt（用户可在 .env 中通过 DEEPSEEK_PROMPT 覆盖）
_DEFAULT_PROMPT = (
    "You are an expert at recognizing handwritten mathematical expressions. "
    "Carefully examine the handwritten formula in the image and transcribe it "
    "into LaTeX code.\n\n"
    "Rules:\n"
    "- Output ONLY the raw LaTeX code, no explanation, no markdown formatting\n"
    "- Do NOT wrap the LaTeX in $$ or $ delimiters\n"
    "- Use correct LaTeX for: \\pm, \\div, \\times, \\sqrt, \\frac, \\sum, \\int\n"
    "- Include Greek letters like \\phi, \\theta, \\alpha, \\beta where appropriate\n"
    "- Pay attention to subscripts (_{...}) and superscripts (^{...})\n"
    "- Include all numbers and operators exactly as written\n"
    "- If the expression contains an equals sign, include it"
)


def encode_image(image: Image.Image) -> str:
    """将 PIL Image 编码为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_latex(raw: str) -> str:
    """从 API 响应中提取纯 LaTeX 代码。

    移除可能出现的 markdown 包围标记：```latex, $$, \\[, \\] 等。
    """
    text = raw.strip()

    # 移除 ```latex ... ``` 或 ``` ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(latex|tex)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # 移除 $$ ... $$
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()

    # 移除 \[ ... \]
    if text.startswith(r"\[") and text.endswith(r"\]"):
        text = text[2:-2].strip()

    # 移除行内 $...$ 包围（仅当整个字符串被包围时）
    if text.startswith("$") and text.endswith("$") and text.count("$") == 2:
        text = text[1:-1].strip()

    return text


class BaseRecognizer(ABC):
    """识别器抽象基类。

    所有具体识别器（DeepSeek、本地模型等）实现此接口。
    """

    @abstractmethod
    def recognize(self, image: Image.Image) -> str:
        """识别单张图像中的数学公式。

        Args:
            image: 渲染后的手写公式图像（灰度或 RGB）。

        Returns:
            LaTeX 字符串（纯 LaTeX，不含包围标记）。
        """
        ...

    @abstractmethod
    def batch_recognize(
        self, images: list[Image.Image]
    ) -> list[str]:
        """批量识别多张图像。

        Args:
            images: 图像列表。

        Returns:
            LaTeX 字符串列表，顺序与输入一致。
        """
        ...


class DeepSeekRecognizer(BaseRecognizer):
    """DeepSeek API 识别器。

    使用 Anthropic 兼容 SDK 调用 DeepSeek 的多模态模型。
    DeepSeek 的 OpenAI 兼容端点为纯文本接口，不支持图片输入，
    因此使用 Anthropic 消息格式的端点发送图片。

    所有配置（API Key、Base URL、Model）从 .env 文件或环境变量读取。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        """初始化 DeepSeek 识别器。

        Args:
            api_key: DeepSeek API Key。如为 None，从 DEEPSEEK_API_KEY 环境变量读取。
            base_url: API 端点。如为 None，从 DEEPSEEK_BASE_URL 环境变量读取。
                      默认使用 Anthropic 兼容端点 https://api.deepseek.com/anthropic。
            model: 模型名。如为 None，从 DEEPSEEK_MODEL 环境变量读取。
            prompt: 识别提示词。如为 None，从 DEEPSEEK_PROMPT 环境变量或使用默认值。
            max_retries: 网络错误时的最大重试次数。
            timeout: 单次请求超时时间（秒，默认 120s 因为图片处理较慢）。
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com/anthropic"
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.prompt = prompt or os.getenv("DEEPSEEK_PROMPT", _DEFAULT_PROMPT)
        self.max_retries = max_retries
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。请在 .env 文件中配置，"
                "或通过 DEEPSEEK_API_KEY 环境变量传入。"
            )

    def _build_client(self):
        """延迟创建 Anthropic 客户端。"""
        import anthropic
        return anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)

    def recognize(self, image: Image.Image) -> str:
        """识别一张图像。

        Args:
            image: 手写公式图像（PIL Image）。

        Returns:
            提取出的 LaTeX 字符串。
        """
        b64 = encode_image(image)

        client = self._build_client()
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.prompt,
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64,
                                    },
                                },
                            ],
                        }
                    ],
                )
                raw = ""
                for block in response.content:
                    if block.type == "text":
                        raw += block.text
                return extract_latex(raw)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt  # 指数退避
                    print(f"[retry {attempt}/{self.max_retries}] {e} — waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"API 调用失败（已重试 {self.max_retries} 次）: {last_error}"
                    ) from last_error

        # 不会到达这里
        return ""

    def batch_recognize(self, images: list[Image.Image]) -> list[str]:
        """批量识别，逐张调用（非并发，控制 API 速率）。"""
        results = []
        for i, img in enumerate(images):
            try:
                latex = self.recognize(img)
            except Exception as e:
                print(f"[ERROR] 第 {i} 张图像识别失败: {e}")
                latex = ""
            results.append(latex)
        return results

    def refine(self, latex: str, prompt: Optional[str] = None) -> str:
        """校正/优化 LaTeX 输出。

        将 OCR 结果发送给 DeepSeek（文本端点）进行语法校正和格式优化。

        Args:
            latex: 原始 LaTeX 字符串。
            prompt: 校正指令，None 则使用默认。

        Returns:
            校正后的 LaTeX。
        """
        if not latex.strip():
            return latex

        prompt = prompt or (
            "The following is LaTeX code from an OCR system for a math formula. "
            "Fix any LaTeX syntax errors and output ONLY the corrected LaTeX. "
            "No explanations, no markdown, no $$ delimiters."
        )

        # refine 是纯文本操作，使用 OpenAI 端点（不是 Anthropic）
        from openai import OpenAI
        text_client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )
        try:
            response = text_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": latex},
                ],
                timeout=30,
            )
            raw = response.choices[0].message.content or ""
            return extract_latex(raw)
        except Exception as e:
            print(f"[refine error] {e}")
            return latex


_STROKE_PROMPT = (
    "You are an expert at recognizing handwritten mathematical expressions from "
    "an ASCII art representation. The expression is shown as a grid of characters "
    "where '#' represents ink strokes and '.' represents empty space.\n\n"
    "Read the grid carefully and transcribe the handwritten mathematical expression "
    "into LaTeX code.\n\n"
    "LaTeX rules:\n"
    "- Use \\pm for ±, \\div for ÷, \\times for ×\n"
    "- Use \\sqrt{} for square roots, \\frac{}{} for fractions\n"
    "- Use _{} for subscripts, ^{} for superscripts\n"
    "- Use \\sum, \\int, \\prod for summation, integral, product\n"
    "- Use \\alpha, \\beta, \\phi, \\theta, etc. for Greek letters\n"
    "- Use \\log, \\sin, \\cos, \\tan for math functions\n"
    "- Include all parentheses () and brackets []\n"
    "- Include all numbers and variables exactly as written\n\n"
    "Output ONLY the raw LaTeX code. No explanations, no markdown, no $$ delimiters."
)


class StrokeTextRecognizer(BaseRecognizer):
    """基于笔画文本的 DeepSeek 识别器。

    将 InkML 笔画坐标转为文本描述后，通过纯文本 API 发送给 DeepSeek。
    适用于不支持图像输入但文本能力强的模型（如 deepseek-v4-pro）。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120,
        compact_format: bool = True,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        self.prompt = prompt or os.getenv("DEEPSEEK_PROMPT", _STROKE_PROMPT)
        self.max_retries = max_retries
        self.timeout = timeout
        self.compact_format = compact_format

        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。请在 .env 文件中配置，"
                "或通过 DEEPSEEK_API_KEY 环境变量传入。"
            )

    def _build_client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def recognize(self, stroke_text: str) -> str:
        """从笔画文本识别数学公式。

        Args:
            stroke_text: 格式化的笔画轨迹文本。

        Returns:
            提取出的 LaTeX 字符串。
        """
        client = self._build_client()
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": stroke_text},
                    ],
                    timeout=self.timeout,
                )
                raw = response.choices[0].message.content or ""
                return extract_latex(raw)

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"[retry {attempt}/{self.max_retries}] {e} — waiting {wait}s")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"API 调用失败（已重试 {self.max_retries} 次）: {last_error}"
                    ) from last_error

        return ""

    def batch_recognize(self, stroke_texts: list[str]) -> list[str]:
        """批量识别。"""
        results = []
        for i, text in enumerate(stroke_texts):
            try:
                latex = self.recognize(text)
            except Exception as e:
                print(f"[ERROR] 第 {i} 个样本识别失败: {e}")
                latex = ""
            results.append(latex)
        return results
