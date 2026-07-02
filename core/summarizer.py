from .llm_client import LLMClient


class Summarizer:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def summarize(self, context: str) -> str:
        return self.llm_client.summarize(self._summary_prompt(context))

    def answer(self, question: str, context: str, response_mode: str = "detailed") -> str:
        return self.llm_client.run_prompt(self._question_prompt(question, context, response_mode=response_mode))

    def _summary_prompt(self, context: str) -> str:
        return (
            "You are summarizing a captured work session from noisy OCR/audio logs.\n"
            "Do not give a generic summary. Reconstruct the most likely activity from filenames, URLs, "
            "page titles, and repeated terms. Separate facts from inferences.\n"
            "Do not expose raw log timestamps like [16:28:10] unless the user explicitly asks for exact times. "
            "If chronology matters, use plain sequence words such as first, then, later, finally.\n\n"
            "Return in this format:\n"
            "1. One-sentence gist\n"
            "2. Work sequence and concrete events, without raw timestamps\n"
            "3. Main topics and artifacts observed\n"
            "4. Action items or next steps, including inferred ones labeled as inferred\n"
            "5. Unclear/noisy parts and what extra capture would help\n"
            "6. Confidence: High/Medium/Low with a short reason\n\n"
            f"Captured context:\n{context}"
        )

    def _question_prompt(self, question: str, context: str, response_mode: str = "detailed") -> str:
        wants_details = response_mode == "detailed" or self._question_wants_details(question)
        response_instruction = (
            "Return a longer, detailed answer in natural paragraphs. Explain the answer clearly, but do not use "
            "section headers such as Evidence from capture, Likely interpretation, Missing/uncertain, Confidence, "
            "or bullet breakdowns. Do not use Markdown formatting, asterisks, numbered lists, or decorative symbols."
            if wants_details
            else
            "Return only the short answer. Do not include sections such as Evidence, Likely interpretation, "
            "Missing/uncertain, confidence, bullet breakdowns, or raw supporting details unless the user asks. "
            "Do not use Markdown formatting, asterisks, numbered lists, or decorative symbols."
        )
        return (
            "Answer the user's question using the captured session context. The context may contain OCR "
            "errors, mixed Chinese/English text, partial URLs, and repeated screen captures.\n"
            "Be useful even when the evidence is incomplete: give the best supported answer and briefly note "
            "uncertainty only if it materially changes the answer.\n\n"
            "Do not include raw capture timestamps or internal session IDs in the answer unless the user asks for them.\n\n"
            "Never wrap words in asterisks. Return plain text only.\n\n"
            f"{response_instruction}\n\n"
            f"Question: {question}\n\nCaptured context:\n{context}"
        )

    def _question_wants_details(self, question: str) -> bool:
        lowered = question.lower()
        detail_terms = [
            "why",
            "how do you know",
            "evidence",
            "source",
            "details",
            "explain",
            "uncertain",
            "confidence",
            "引用",
            "证据",
            "来源",
            "为什么",
            "解释",
            "详细",
            "不确定",
        ]
        return any(term in lowered for term in detail_terms)
