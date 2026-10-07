"""Heightmap terrain: loading, sampling, and procedural generation.

Conventions
-----------
``heights[j, i]`` is the terrain height (m) at ``x = x_min + i * dx`` (east) and
``y = y_min + j * dy`` (north), so row 0 is the *southern* edge. Image files are
stored the usual way (top row = north) and flipped on load/save.

A heightmap is described by a small YAML file::

    image: demo_valley.png      # 16-bit grayscale PNG or .npy (float metres)
    x_min: -50000
    x_max: 850000
    y_min: -150000
    y_max: 150000
    z_min: -50                  # PNG only: height of pixel value 0
    z_max: 2400                 # PNG only: height of pixel value 65535
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


@dataclass
class Heightmap:
    heights: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    name: str = "terrain"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.heights = np.ascontiguousarray(self.heights, dtype=np.float64)
        if self.heights.ndim != 2 or min(self.heights.shape) < 2:
            raise ValueError("heights must be a 2-D array with at least 2x2 samples")
        if not (self.x_max > self.x_min and self.y_max > self.y_min):
            raise ValueError("terrain extent must have x_max > x_min and y_max > y_min")

    # ------------------------------------------------------------------ geometry
    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape  # (ny, nx)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.shape[1] - 1)

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / (self.shape[0] - 1)

    def contains(self, x, y) -> np.ndarray | bool:
        return (self.x_min <= x) & (x <= self.x_max) & (self.y_min <= y) & (y <= self.y_max)

    def height(self, x, y):
        """Bilinearly interpolated height; points outside the map are clamped to its edge."""
        ny, nx = self.shape
        fx = np.clip((np.asarray(x, dtype=float) - self.x_min) / self.dx, 0.0, nx - 1.000001)
        fy = np.clip((np.asarray(y, dtype=float) - self.y_min) / self.dy, 0.0, ny - 1.000001)
        i = fx.astype(int)
        j = fy.astype(int)
        tx = fx - i
        ty = fy - j
        h = self.heights
        out = (
            h[j, i] * (1 - tx) * (1 - ty)
            + h[j, i + 1] * tx * (1 - ty)
            + h[j + 1, i] * (1 - tx) * ty
            + h[j + 1, i + 1] * tx * ty
        )
        return float(out) if np.ndim(out) == 0 else out

    def gradient(self, x, y):
        """(dh/dx, dh/dy) by central differences of the interpolant."""
        ex, ey = 0.5 * self.dx, 0.5 * self.dy
        gx = (self.height(x + ex, y) - self.height(x - ex, y)) / (2 * ex)
        gy = (self.height(x, y + ey) - self.height(x, y - ey)) / (2 * ey)
        return gx, gy

    def normal(self, x: float, y: float) -> np.ndarray:
        gx, gy = self.gradient(x, y)
        n = np.array([-gx, -gy, 1.0])
        return n / np.linalg.norm(n)

    def slope_deg(self, x: float, y: float) -> float:
        n = self.normal(x, y)
        return float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))

    def sample_grid(self, x_min, x_max, y_min, y_max, nx, ny) -> np.ndarray:
        """Resample onto a regular grid with the same row convention (row 0 = south)."""
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(xs, ys)
        return self.height(X, Y)

    # ------------------------------------------------------------------ io
    def save(self, yaml_path: str | Path, fmt: str = "png") -> Path:
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            **{k: v for k, v in self.meta.items() if k not in {"image", "z_min", "z_max"}},
        }
        if fmt == "png":
            z_min, z_max = float(self.heights.min()), float(self.heights.max())
            span = max(z_max - z_min, 1e-6)
            img = np.round((self.heights - z_min) / span * 65535).astype(np.uint16)
            image_path = yaml_path.with_suffix(".png")
            Image.fromarray(img[::-1]).save(image_path)
            meta.update(image=image_path.name, z_min=z_min, z_max=z_max)
        elif fmt == "npy":
            image_path = yaml_path.with_suffix(".npy")
            np.save(image_path, self.heights[::-1].astype(np.float32))
            meta.update(image=image_path.name)
        else:
            raise ValueError(f"unknown heightmap format {fmt!r}")
        yaml_path.write_text(yaml.safe_dump(meta, sort_keys=False))
        return yaml_path

    @classmethod
    def load(cls, yaml_path: str | Path) -> Heightmap:
        yaml_path = Path(yaml_path)
        meta = yaml.safe_load(yaml_path.read_text())
        image_path = yaml_path.parent / meta["image"]
        if image_path.suffix == ".npy":
            heights = np.load(image_path).astype(np.float64)[::-1]
        else:
            img = np.asarray(Image.open(image_path))
            if img.ndim == 3:
                img = img[..., 0]
            full = 65535.0 if img.dtype == np.uint16 or img.max() > 255 else 255.0
            z_min, z_max = float(meta.get("z_min", 0.0)), float(meta.get("z_max", 1000.0))
            heights = z_min + img.astype(np.float64)[::-1] / full * (z_max - z_min)
        extra = {
            k: v
            for k, v in meta.items()
            if k not in {"name", "x_min", "x_max", "y_min", "y_max", "image", "z_min", "z_max"}
        }
        return cls(
            heights=heights,
            x_min=float(meta["x_min"]),
            x_max=float(meta["x_max"]),
            y_min=float(meta["y_min"]),
            y_max=float(meta["y_max"]),
            name=meta.get("name", yaml_path.stem),
            meta=extra,
        )


# ---------------------------------------------------------------------- procedural
def fractal_noise(shape: tuple[int, int], beta: float, rng: np.random.Generator) -> np.ndarray:
    """Spectral-synthesis noise with power spectrum ~ 1/f^beta, normalised to [-1, 1]."""
    ny, nx = shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, amp.shape)
    field_ = np.fft.irfft2(amp * np.exp(1j * phase), s=shape)
    field_ -= field_.mean()
    return field_ / (np.abs(field_).max() + 1e-12)


def flatten_sites(
    hm: Heightmap, sites: list[tuple[float, float]], radius: float, blend: float
) -> Heightmap:
    """Flatten circular plateaus (launch/landing zones) with a smooth blend to the surroundings."""
    ny, nx = hm.shape
    X, Y = np.meshgrid(np.linspace(hm.x_min, hm.x_max, nx), np.linspace(hm.y_min, hm.y_max, ny))
    h = hm.heights.copy()
    for sx, sy in sites:
        h0 = hm.height(sx, sy)
        d = np.hypot(X - sx, Y - sy)
        w = np.clip((d - radius) / max(blend, 1e-9), 0.0, 1.0)
        w = w * w * (3 - 2 * w)  # smoothstep
        h = h * w + h0 * (1 - w)
    return Heightmap(h, hm.x_min, hm.x_max, hm.y_min, hm.y_max, hm.name, dict(hm.meta))


def generate_regional_map(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    seed: int = 0,
    relief: float = 1800.0,
    base: float = 300.0,
    name: str = "procedural",
) -> Heightmap:
    """Large-scale terrain: rolling plains, a few ridgelines and valleys."""
    rng = np.random.default_rng(seed)
    nx = round((x_range[1] - x_range[0]) / resolution) + 1
    ny = round((y_range[1] - y_range[0]) / resolution) + 1
    broad = fractal_noise((ny, nx), beta=3.6, rng=rng)
    ridged = 1.0 - np.abs(fractal_noise((ny, nx), beta=2.6, rng=rng))
    detail = fractal_noise((ny, nx), beta=2.0, rng=rng)
    h = base + relief * (0.55 * broad + 0.45 * ridged**3) + 0.05 * relief * detail
    h = np.maximum(h, 0.0)
    return Heightmap(h, x_range[0], x_range[1], y_range[0], y_range[1], name)


def generate_detail_tile(
    base: Heightmap,
    center: tuple[float, float],
    half_size: float,
    resolution: float,
    seed: int = 0,
    roughness: float = 1.5,
    rock_density: float = 4e-4,
    rock_height: tuple[float, float] = (0.2, 1.2),
    name: str = "detail",
) -> Heightmap:
    """High-resolution tile of 'unprepared ground' over a base map: bumps, undulation, rocks."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    n = round(2 * half_size / resolution) + 1
    grid = base.sample_grid(cx - half_size, cx + half_size, cy - half_size, cy + half_size, n, n)
    grid = grid + roughness * fractal_noise((n, n), beta=2.4, rng=rng)
    xs = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(xs, xs)
    n_rocks = rng.poisson(rock_density * (2 * half_size) ** 2)
    for _ in range(n_rocks):
        rx, ry = rng.uniform(-half_size, half_size, 2)
        r = rng.uniform(0.8, 3.0)
        hr = rng.uniform(*rock_height)
        d2 = (X - rx) ** 2 + (Y - ry) ** 2
        grid += hr * np.exp(-d2 / (2 * (r / 2) ** 2))
    return Heightmap(grid, cx - half_size, cx + half_size, cy - half_size, cy + half_size, name)
