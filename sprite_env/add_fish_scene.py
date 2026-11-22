"""Сценка с рыбкой из refs/fish_toss_strip.webp + refs/fish.png.

Позы дракончика берутся с листа как есть (без перерисовки). Рыбка в полёте
вынесена в отдельный спрайт и летит по дуге с вращением, поэтому бросок
получается плавным и законченным: взял → посмотрел → подбросил → следит →
поймал → обнял → довольно зажмурился.

Дописывает/заменяет анимации в уже собранном sprite_env/out
(остальные анимации берутся из out/frames без изменений).

Запуск:  python sprite_env/add_fish_scene.py
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import pixelkit as pk  # noqa: E402
from slice_refs import CELL, FLOOR, SIT_H, REFS, OUT, cut_grid, body_bottom, place  # noqa: E402


def split_body(fr):
    """Разделить кадр на тело (самый большой кусок) и всё остальное (рыбку)."""
    al = fr[..., 3] > 0
    lab, n = ndimage.label(al)
    if n <= 1:
        return fr, None
    sizes = ndimage.sum(al, lab, range(1, n + 1))
    body_id = int(np.argmax(sizes)) + 1
    body = fr.copy()
    body[..., 3] = np.where(lab == body_id, fr[..., 3], 0)
    ys, xs = np.nonzero(lab == body_id)
    body = body[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rest = al & (lab != body_id)
    rx = np.nonzero(rest.any(0))[0]
    return body, (rx.max() - rx.min() + 1 if len(rx) else None)


def fish_sprite(scale):
    a = np.array(Image.open(os.path.join(REFS, 'fish.png')).convert('RGBA'))
    ys, xs = np.nonzero(a[..., 3] > 100)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray(a, 'RGBA')
    w = max(1, round(im.width * scale))
    h = max(1, round(im.height * scale))
    return im.resize((w, h), Image.LANCZOS)


def stamp(canvas, fish, cx, cy, angle):
    """Положить рыбку центром в (cx, cy), повернув на angle градусов."""
    f = fish.rotate(angle, resample=Image.NEAREST, expand=True)
    a = np.array(f)
    a[..., 3] = np.where(a[..., 3] > 110, 255, 0)
    h, w = a.shape[:2]
    ox, oy = round(cx - w / 2), round(cy - h / 2)
    ys, xs = max(0, -oy), max(0, -ox)
    ye, xe = min(h, CELL - oy), min(w, CELL - ox)
    if ye <= ys or xe <= xs:
        return canvas
    sub = a[ys:ye, xs:xe]
    dst = canvas[oy + ys:oy + ye, ox + xs:ox + xe]
    m = sub[..., 3] > 0
    dst[m] = sub[m]
    return canvas


def load_existing():
    with open(os.path.join(OUT, 'manifest.json'), encoding='utf-8') as fh:
        man = json.load(fh)
    anims = []
    for name, a in sorted(man['animations'].items(), key=lambda kv: kv[1]['row']):
        frames = [np.array(Image.open(os.path.join(OUT, 'frames', f'{name}_{i:02d}.png')).convert('RGBA'))
                  for i in range(a['frames'])]
        anims.append([name, frames, a['fps'], a['loop'], a.get('description', '')])
    return man, anims


def main():
    strip = cut_grid('fish_toss_strip.webp', 1, 8)[0]
    k = SIT_H / body_bottom(strip[0])
    # тела без летящей рыбки (кадры 3 и 4 — лапки вверх, взгляд вверх)
    up_a, _ = split_body(strip[3])
    up_b, fish_w = split_body(strip[4])

    # рыбка того же размера, что летящая на листе
    src_w = Image.open(os.path.join(REFS, 'fish.png')).getbbox()
    fish = fish_sprite(k * fish_w / (src_w[2] - src_w[0]))

    hold, look_fish, to_mouth, catch, hug, happy = (strip[i] for i in (0, 1, 2, 5, 6, 7))
    P = lambda fr, dy=0: place(fr, k, dy)  # noqa: E731

    # траектория рыбки: от лапок вверх и обратно в лапки, 1.5 оборота
    PAW = (CELL / 2 - 4, FLOOR - 92)   # где рыбка в лапках
    TOP = PAW[1] - 120
    N = 10
    flight = []
    for i in range(N + 1):
        t = i / N
        y = PAW[1] - (PAW[1] - TOP) * 4 * t * (1 - t)
        x = PAW[0] + 10 * math.sin(t * math.pi)
        flight.append((x, y, -540 * t))

    toss = [P(hold), P(hold), P(look_fish), P(look_fish), P(to_mouth),
            P(to_mouth, 4)]                                # присел перед броском
    for i, (fx, fy, ang) in enumerate(flight[1:-1], 1):
        body = up_a if i in (1, 2, 8, 9) else up_b        # на вершине смотрит выше
        dy = -6 if i == 1 else 0                          # подскок на броске
        toss.append(stamp(P(body, dy), fish, fx, fy, ang))
    toss += [P(catch, 3), P(catch), P(to_mouth), P(hug), P(hug), P(happy), P(happy), P(happy), P(hold)]

    # «доволен рыбкой»: обнимает, жмурится, ведёт носом
    hug_loop = [P(hug), P(hug), P(happy), P(happy), P(happy), P(hug), P(hold), P(hold)]

    new = {
        'toss_fish': (toss, 8, False, 'Подбрасывает рыбку, ловит и довольно обнимает'),
        'fish_hug': (hug_loop, 3, False, 'Обнимает рыбку и довольно жмурится'),
    }

    man, anims = load_existing()
    names = [a[0] for a in anims]
    for name, (frames, fps, loop, desc) in new.items():
        if name in names:
            anims[names.index(name)] = [name, frames, fps, loop, desc]
        else:
            anims.append([name, frames, fps, loop, desc])

    # убрать старые кадры заменённых анимаций (их могло быть больше)
    for name in new:
        for f in os.listdir(os.path.join(OUT, 'frames')):
            if f.startswith(name + '_') and f[len(name) + 1:-4].isdigit():
                os.remove(os.path.join(OUT, 'frames', f))

    pk.set_frame_size(CELL)
    bad = pk.export([tuple(a) for a in anims], OUT, name=man['name'], anchor=tuple(man['anchor']))
    print('scale', round(k, 3), '| toss frames', len(toss))
    print('frames touching edge:', bad or 'none')


if __name__ == '__main__':
    main()
