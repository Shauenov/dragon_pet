"""pixelkit — среда для процедурной пиксельной анимации спрайтов.

Только инструменты, без персонажей и примеров:
  * маски-примитивы на сетке кадра без сглаживания (эллипс, полигон, капсула,
    трубка по кривой Безье), сдвиг и расширение масок;
  * холст RGBA: заливка масок и отдельных пикселей (без встроенного стиля —
    контур, свет и тени рисуются так, как решит автор);
  * экспорт: отдельные кадры, спрайт-листы, общий атлас, manifest.json
    и проверка, что ни один кадр не касается края ячейки.

Зависимости: numpy, Pillow (см. requirements.txt).
"""
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- размер кадра
W = H = 64
YY, XX = np.mgrid[0:H, 0:W]
PX, PY = XX + 0.5, YY + 0.5  # центры пикселей


def set_frame_size(w, h=None):
    """Сменить размер кадра (вызывать до рисования)."""
    global W, H, YY, XX, PX, PY
    W, H = w, (h or w)
    YY, XX = np.mgrid[0:H, 0:W]
    PX, PY = XX + 0.5, YY + 0.5


# ---------------------------------------------------------------- маски
def ell(cx, cy, rx, ry, ang=0.0):
    c, s = math.cos(ang), math.sin(ang)
    dx, dy = PX - cx, PY - cy
    u, v = dx * c + dy * s, -dx * s + dy * c
    return (u / max(rx, 0.3)) ** 2 + (v / max(ry, 0.3)) ** 2 <= 1.0


def poly(pts):
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon([(x - 0.5, y - 0.5) for x, y in pts], fill=1)
    return np.array(im, bool)


def capsule(x1, y1, x2, y2, r1, r2=None):
    r2 = r1 if r2 is None else r2
    vx, vy = x2 - x1, y2 - y1
    l2 = vx * vx + vy * vy or 1e-6
    t = np.clip(((PX - x1) * vx + (PY - y1) * vy) / l2, 0, 1)
    d = np.hypot(PX - (x1 + t * vx), PY - (y1 + t * vy))
    return d <= r1 + (r2 - r1) * t


def bez(p, t):
    a, b, c, d = p
    u = 1 - t
    return (u**3 * a[0] + 3 * u * u * t * b[0] + 3 * u * t * t * c[0] + t**3 * d[0],
            u**3 * a[1] + 3 * u * u * t * b[1] + 3 * u * t * t * c[1] + t**3 * d[1])


def tube(ctrl, r0, r1, n=14):
    """Сужающаяся трубка вдоль кубической кривой Безье (4 точки)."""
    m = np.zeros((H, W), bool)
    pts = [bez(ctrl, i / n) for i in range(n + 1)]
    for i in range(n):
        ra = r0 + (r1 - r0) * i / n
        rb = r0 + (r1 - r0) * (i + 1) / n
        m |= capsule(*pts[i], *pts[i + 1], ra, rb)
    return m


def at(m, dx, dy):
    """Сдвиг маски: r[y,x] = m[y+dy, x+dx] (вне холста — False)."""
    r = np.zeros_like(m)
    ys = slice(max(0, -dy), min(H, H - dy))
    xs = slice(max(0, -dx), min(W, W - dx))
    ys2 = slice(max(0, dy), min(H, H + dy))
    xs2 = slice(max(0, dx), min(W, W + dx))
    r[ys, xs] = m[ys2, xs2]
    return r


def dilate4(m):
    return m | at(m, 1, 0) | at(m, -1, 0) | at(m, 0, 1) | at(m, 0, -1)


# ---------------------------------------------------------------- математика
TAU = math.tau


def rot(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def lerp(a, b, t):
    return a + (b - a) * t


def lerp2(a, b, t):
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))


# ---------------------------------------------------------------- холст
class Canvas:
    """Пустой RGBA-кадр: заливка масок и отдельных пикселей."""

    def __init__(self):
        self.a = np.zeros((H, W, 4), np.uint8)

    def put(self, m, col, alpha=255):
        self.a[m] = (*col, alpha)

    def px(self, x, y, col, alpha=255):
        x, y = int(math.floor(x)), int(math.floor(y))
        if 0 <= x < W and 0 <= y < H:
            self.a[y, x] = (*col, alpha)


# ---------------------------------------------------------------- экспорт
def touches_edge(frame):
    al = frame[..., 3] > 0
    return bool(al[0].any() or al[-1].any() or al[:, 0].any() or al[:, -1].any())


def export(anims, out_dir, name='sprite', anchor=None, extra=None):
    """anims: список (имя, [RGBA-кадры HxWx4], fps, loop, описание).

    Пишет out_dir/frames/*.png, out_dir/sheets/*.png, atlas.png, manifest.json.
    Возвращает список кадров, касающихся края ячейки (их стоит исправить).
    """
    os.makedirs(os.path.join(out_dir, 'sheets'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'frames'), exist_ok=True)
    cols = max(len(fr) for _, fr, *_ in anims)
    atlas = Image.new('RGBA', (cols * W, len(anims) * H), (0, 0, 0, 0))
    manifest = dict(name=name, frameWidth=W, frameHeight=H,
                    anchor=list(anchor or (W // 2, H - 4)), atlas='atlas.png', animations={})
    if extra:
        manifest.update(extra)
    bad = []
    for row, (an, frames, fps, loop, desc) in enumerate(anims):
        sheet = Image.new('RGBA', (len(frames) * W, H), (0, 0, 0, 0))
        for i, fr in enumerate(frames):
            if touches_edge(fr):
                bad.append(f'{an}_{i:02d}')
            im = Image.fromarray(fr, 'RGBA')
            sheet.paste(im, (i * W, 0))
            atlas.paste(im, (i * W, row * H))
            im.save(os.path.join(out_dir, 'frames', f'{an}_{i:02d}.png'))
        sheet.save(os.path.join(out_dir, 'sheets', f'{an}.png'))
        manifest['animations'][an] = dict(sheet=f'sheets/{an}.png', row=row, frames=len(frames),
                                          fps=fps, loop=loop, description=desc)
    atlas.save(os.path.join(out_dir, 'atlas.png'))
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return bad
