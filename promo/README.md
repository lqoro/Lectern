# Lectern ad (20 s, 1080×1920)

`lectern-ad-1080x1920.mp4` is rendered from `ad.html` against the **real redesigned app** (the
`redesign-web` UI, run with a seeded copy of the website's sample lecture), frame by frame.
Brand: dark ground, Albert Sans and the tape motif from lectern.lqoro.com. The soundtrack is the
one from the previous cut; scene timings are unchanged so it stays in sync.

Scenes: intro (0–2.6 s) · click a word (–8) · speed (–11) · export Markdown (–13) · practice quiz (–17) ·
end card with lectern.lqoro.com and Mac / Windows (Beta) (–20).

## Re-render

```bash
# 1. assets from the site into ./assets-src/site/assets  (demo.json, demo-lecture.m4a, fonts/albert-sans-latin.woff2)
# 2. check out the redesigned app and start it on a seeded library
python3 seed.py ./assets-src                  # writes ./assets-src/lectures
LECTERN_ROOT=$PWD/assets-src/lectures LECTERN_SUPPORT=$PWD/assets-src/support \
  python -m lectern                           # from a checkout of origin/redesign-web; http://127.0.0.1:8477
curl -X POST localhost:8477/api/notice        # dismiss the first-run notice
# 3. copy albert-sans-latin.woff2, icon-512.png (as icon.png) and demo.json into ./assets
node render.js --stills 3,5.5,12 stills       # quick check
node render.js ad-silent.mp4                  # full render (needs playwright + ffmpeg)
ffmpeg -i ad-silent.mp4 -i <soundtrack> -map 0:v -map 1:a -c:v copy -c:a aac -shortest ad.mp4
```
