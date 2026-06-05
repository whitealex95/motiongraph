"""Compress the demo videos into docs/videos/ for the GitHub Pages site.

Run after (re)rendering any video that the website shows, then commit docs/.
Usage: python tools/make_web.py
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
WEB = os.path.join(ROOT, "docs", "videos")

# (source in outputs/, web filename). Keep this list in sync with docs/index.html.
VIDEOS = [
    ("dataset_walk.mp4", "dataset_walk.mp4"),
    ("dataset_jump.mp4", "dataset_jump.mp4"),
    ("exp1_mg_path.mp4", "exp1_mg.mp4"),
    ("exp1_mm_path.mp4", "exp1_mm.mp4"),
    ("exp2_mg_path_jump.mp4", "exp2_mg.mp4"),
    ("exp2_mm_path_jump.mp4", "exp2_mm.mp4"),
    ("exp2_mm_reactive.mp4", "exp2_mm_reactive.mp4"),
]


def main():
    os.makedirs(WEB, exist_ok=True)
    for src, dst in VIDEOS:
        s = os.path.join(OUT, src)
        if not os.path.exists(s):
            print(f"  MISSING {src} (render it first)")
            continue
        d = os.path.join(WEB, dst)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", s,
            "-vf", "scale='min(800,iw)':-2", "-c:v", "libx264", "-crf", "30",
            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an",
            "-movflags", "+faststart", d], check=True)
        mb = os.path.getsize(d) / 1e6
        print(f"  {src} -> docs/videos/{dst}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
