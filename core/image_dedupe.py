from collections import deque

from PIL import Image, ImageChops, ImageStat


class FrameDeduper:
    def __init__(
        self,
        threshold: float = 0.02,
        recent_size: int = 3,
        crop_top_ratio: float = 0.55,
        crop_bottom_ratio: float = 0.92,
        sample_size: tuple[int, int] = (64, 36),
    ) -> None:
        self.threshold = threshold
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.sample_size = sample_size
        self.recent_frames = deque(maxlen=recent_size)

    def should_process(self, image: Image.Image) -> bool:
        sample = self._sample(image)
        if not self.recent_frames:
            self.recent_frames.append(sample)
            return True
        score = min(self._difference(sample, previous) for previous in self.recent_frames)
        self.recent_frames.append(sample)
        return score >= self.threshold

    def crop_region(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        top = int(height * self.crop_top_ratio)
        bottom = int(height * self.crop_bottom_ratio)
        if bottom <= top:
            top = 0
            bottom = height
        return image.crop((0, top, width, bottom))

    def _sample(self, image: Image.Image) -> Image.Image:
        return self.crop_region(image).convert("L").resize(self.sample_size)

    def _difference(self, first: Image.Image, second: Image.Image) -> float:
        diff = ImageChops.difference(first, second)
        stat = ImageStat.Stat(diff)
        return float(stat.mean[0]) / 255.0
