from core.summarizer import Summarizer


class FakeLLMClient:
    def summarize(self, text):
        return text

    def run_prompt(self, prompt):
        return prompt


def test_question_prompt_defaults_to_detailed_answer():
    prompt = Summarizer(FakeLLMClient()).answer("What is the video about?", "context")

    assert "Return a longer, detailed answer" in prompt
    assert "natural paragraphs" in prompt
    assert "Do not use Markdown formatting" in prompt


def test_question_prompt_allows_details_when_requested():
    prompt = Summarizer(FakeLLMClient()).answer("What is the evidence?", "context")

    assert "natural paragraphs" in prompt
    assert "Do not use Markdown formatting" in prompt


def test_question_prompt_supports_detailed_mode_without_old_sections():
    prompt = Summarizer(FakeLLMClient()).answer("What is the video about?", "context", response_mode="detailed")

    assert "Return a longer, detailed answer" in prompt
    assert "Do not use Markdown formatting" in prompt
    assert "Evidence from capture, Likely interpretation, Missing/uncertain, Confidence" in prompt
