"""Render the README GIFs from bundled replays with the viewer's capture mode.

    uv run python scripts/make_gifs.py            # all
    uv run python scripts/make_gifs.py hop        # one

Starts ``plume viz`` on a spare port, drives headless Chromium (Playwright) through
``window.plumeCapture`` frame by frame, and writes optimised GIFs to docs/assets/.
"""

from __future__ import annotations

import io
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

OUT = Path("docs/assets")
W, H = 640, 360

# name: (query, playback fps of the GIF)
SHOTS = {
    "hop": ("replay=cargo_hop_demo.plume.json.gz&camera=chase&t0=0&t1=110&fps=0.65", 14),
    "hop_landing": ("replay=cargo_hop_demo.plume.json.gz&camera=chase&t0=496&t1=527&fps=2", 14),
    "landing": ("replay=landing_pid_full_descent.plume.json.gz&camera=chase&t0=4&t1=40&fps=2", 14),
    "compare": (
        "compare=hobby_real.plume.json.gz,hobby_sim_calibrated.plume.json.gz&camera=chase&t0=-0.4&t1=13&fps=6",
        15,
    ),
    "hop_top": ("replay=cargo_hop_demo.plume.json.gz&camera=top&t0=0&t1=525&fps=0.14", 12),
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_up(url: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"viewer did not start at {url}")


def save_gif(frames: list[Image.Image], path: Path, fps: float) -> None:
    pal = frames[len(frames) // 2].convert("P", palette=Image.ADAPTIVE, colors=96)
    q = [f.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(
        path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True
    )


def main(names: list[str]) -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "plume.cli", "viz", "--port", str(port), "--no-open"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        wait_up(base + "/api/replays")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"]
            )
            page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
            for name in names:
                query, gif_fps = SHOTS[name]
                page.goto(f"{base}/?capture=1&{query}")
                page.wait_for_function(
                    "window.plumeCapture && document.documentElement.dataset.captureReady === '1'",
                    timeout=120_000,
                )
                n = page.evaluate("window.plumeCapture.frameCount")
                frames = []
                t_start = time.time()
                for i in range(n):
                    page.evaluate(f"window.plumeCapture.seek({i})")
                    frames.append(Image.open(io.BytesIO(page.screenshot(type="png"))).copy())
                path = OUT / f"{name}.gif"
                save_gif(frames, path, gif_fps)
                print(
                    f"{name}: {n} frames in {time.time() - t_start:.0f}s -> {path} ({path.stat().st_size / 1e6:.1f} MB)"
                )
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main(sys.argv[1:] or list(SHOTS))
