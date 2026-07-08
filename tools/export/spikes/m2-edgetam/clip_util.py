"""Synthetic test clip shared by capture_golden.py and e2e_loop.py.

8 frames, 640x480 (deliberately non-square to exercise the anisotropic
1024x1024 square-stretch): a red ball moving right over a static gradient
background with a gray distractor square.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

NUM_FRAMES = 8
WIDTH, HEIGHT = 640, 480
BALL_R = 60
BALL_Y = 240
BALL_X0 = 140
BALL_DX = 45  # ball center x at frame t: BALL_X0 + BALL_DX * t

# One positive click on the ball at frame 0, in original pixel space.
PROMPT_FRAME = 0
PROMPT_POINT_XY = (BALL_X0, BALL_Y)


def make_frame(t: int) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT))
    d = ImageDraw.Draw(img)
    for y in range(0, HEIGHT, 4):  # cheap vertical gradient
        g = 40 + int(120 * y / HEIGHT)
        d.rectangle([0, y, WIDTH, y + 4], fill=(g // 2, g, 160))
    d.rectangle([480, 60, 580, 160], fill=(120, 120, 120))  # static distractor
    cx = BALL_X0 + BALL_DX * t
    d.ellipse([cx - BALL_R, BALL_Y - BALL_R, cx + BALL_R, BALL_Y + BALL_R],
              fill=(220, 60, 50), outline=(255, 200, 180), width=4)
    return img


def make_clip() -> list[Image.Image]:
    return [make_frame(t) for t in range(NUM_FRAMES)]
