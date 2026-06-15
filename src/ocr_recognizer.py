"""
OCR Recognizer — 使用开源 OCR 模型将公式图像直接转为 LaTeX。

基于 LaTeXify (tjoab/latex_finetuned)，一个在 MathWriting 数据集上
微调过的 TrOCR 模型，专门用于手写数学公式 → LaTeX。
可选 DeepSeek 后处理校正 LaTeX 语法错误。
"""

import os
import time
from typing import Optional

# 确保 HF 使用国内镜像，在 import transformers 前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from PIL import Image

from .api_recognizer import BaseRecognizer, DeepSeekRecognizer


_REFINE_PROMPT = (
    "You are a LaTeX expert. The following LaTeX code was produced by an OCR "
    "system that recognized a handwritten mathematical expression.\n\n"
    "Check the LaTeX for errors and fix them. Common issues:\n"
    "- Missing or extra braces {}\n"
    "- Incorrect escaping of special characters\n"
    "- Wrong commands (e.g. \\pm vs \\mp, \\div vs \\divide)\n\n"
    "Output ONLY the corrected raw LaTeX code. No explanations, no markdown, "
    "no $$ delimiters."
)


class OcrRecognizer(BaseRecognizer):
    """基于 LaTeXify (TrOCR fine-tuned) 的公式识别器。

    使用 tjoab/latex_finetuned 模型直接将手写公式图像转为 LaTeX。
    可选 DeepSeek 后处理校正 LaTeX 语法。

    该模型是 TrOCR (BEiT encoder + RoBERTa decoder) 在 MathWriting
    数据集上微调得到的，专门针对手写数学公式。
    """

    def __init__(
        self,
        use_deepseek_refine: bool = True,
        model_name: str = "tjoab/latex_finetuned",
        refine_prompt: Optional[str] = None,
        max_new_tokens: int = 200,
    ):
        self.model_name = model_name
        self.use_deepseek_refine = use_deepseek_refine
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None

        if use_deepseek_refine:
            self.refiner = DeepSeekRecognizer()
            self.refine_prompt = refine_prompt or _REFINE_PROMPT

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        print(f"[LaTeXify] 加载模型 {self.model_name}...")
        t0 = time.time()
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        print(f"[LaTeXify] 模型加载完成 ({time.time() - t0:.1f}s)")

    def recognize(self, image: Image.Image) -> str:
        self._load_model()
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self._model.generate(
                pixel_values, max_new_tokens=self.max_new_tokens
            )
            latex = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
        except Exception as e:
            print(f"[LaTeXify ERROR] {e}")
            latex = ""

        print(f"  [LaTeXify] raw: \"{latex}\"")

        # DeepSeek 后处理校正
        if self.use_deepseek_refine and latex:
            try:
                refined = self.refiner.refine(latex, self.refine_prompt)
                if refined and refined != latex:
                    print(f"  [DeepSeek] refined: \"{refined}\"")
                    latex = refined
            except Exception as e:
                print(f"[refine ERROR] {e}")

        return latex

    def batch_recognize(self, images: list[Image.Image]) -> list[str]:
        self._load_model()
        results = []
        for i, img in enumerate(images):
            try:
                latex = self.recognize(img)
            except Exception as e:
                print(f"[ERROR] 第 {i} 张图像识别失败: {e}")
                latex = ""
            results.append(latex)
        return results
