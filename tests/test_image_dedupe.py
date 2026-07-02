from PIL import Image, ImageDraw

from core.image_dedupe import FrameDeduper


def test_frame_deduper_skips_near_duplicate_frames():
    deduper = FrameDeduper(threshold=0.02)
    image = Image.new("RGB", (200, 100), "black")

    assert deduper.should_process(image)
    assert not deduper.should_process(image.copy())


def test_frame_deduper_processes_subtitle_region_change():
    deduper = FrameDeduper(threshold=0.02)
    first = Image.new("RGB", (200, 100), "black")
    second = first.copy()
    draw = ImageDraw.Draw(second)
    draw.rectangle((20, 70, 180, 88), fill="white")

    assert deduper.should_process(first)
    assert deduper.should_process(second)
