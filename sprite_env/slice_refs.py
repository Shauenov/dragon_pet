"""Нарезка референсных спрайт-листов (refs/1..3.webp) на кадры без изменений рисовки.

Каждый кадр берётся с листа как есть: убирается только белый фон
(заливкой от краёв ячейки), затем кадр масштабируется так, чтобы рост
сидящего дракончика совпадал на всех листах, и ставится в ячейку
по нижнему центру (лапы на одной линии).

Запуск:  python sprite_env/slice_refs.py
Результат: sprite_env/out/{frames,sheets,atlas.png,manifest.json}
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import pixelkit as pk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, 'refs')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')

CELL = 256          # размер кадра в атласе
SIT_H = 150         # рост сидящего дракончика в кадре, px
FLOOR = CELL - 12   # линия пола (низ лап)
BG_TOL = 34         # насколько пиксель может отличаться от белого, чтобы считаться фоном


def bands(profile, min_gap):
    """Отрезки, где profile>0, слитые через щели короче min_gap."""
    on = profile > 0
    out, start, last = [], None, None
    for i, v in enumerate(on):
        if v:
            if start is None:
                start = i
            elif i - last > min_gap:
                out.append((start, last + 1))
                start = i
            last = i
    if start is not None:
        out.append((start, last + 1))
    return out


def load_sheet(name):
    src = Image.open(os.path.join(REFS, name))
    if src.mode == 'RGBA':
        a = np.array(src)
        fg = ndimage.binary_opening(a[..., 3] > 100, iterations=1)
        return a[..., :3], fg
    rgb = np.array(src.convert('RGB')).astype(np.int16)
    near_white = (255 - rgb).max(axis=2) <= BG_TOL
    # фон = белые пиксели, связанные с краем листа
    lab, _ = ndimage.label(near_white)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(edge))
    fg = ~bg
    # убрать одиночную JPEG-пыль
    fg = ndimage.binary_opening(fg, iterations=1)
    return rgb.astype(np.uint8), fg


def cut_grid(name, rows, cols):
    """Каждый связный кусок относится к ячейке сетки по центру масс
    (листы нарисованы на ровной сетке; рыбка и капельки — отдельные куски)."""
    rgb, fg = load_sheet(name)
    Hs, Ws = fg.shape
    lab, n = ndimage.label(fg)
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    cms = ndimage.center_of_mass(fg, lab, range(1, n + 1))
    cell_of = {}
    for i, (s, (cy, cx)) in enumerate(zip(sizes, cms), 1):
        if s < 12:
            continue
        r = min(rows - 1, int(cy / (Hs / rows)))
        c = min(cols - 1, int(cx / (Ws / cols)))
        cell_of.setdefault((r, c), []).append(i)
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            ids = cell_of.get((r, c))
            assert ids, f'{name}: пустая ячейка {r},{c}'
            keep = np.isin(lab, ids)
            ys, xs = np.nonzero(keep)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
            rgba[..., :3] = rgb[y0:y1, x0:x1]
            rgba[..., 3] = keep[y0:y1, x0:x1] * 255
            row.append(rgba)
        grid.append(row)
    return grid


def body_bottom(fr):
    """Низ тела = самая нижняя строка, где непрозрачных пикселей заметно много."""
    al = fr[..., 3] > 0
    cnt = al.sum(1)
    rows = np.nonzero(cnt > max(4, cnt.max() * 0.08))[0]
    return rows.max() + 1


def place(fr, scale, dy=0):
    im = Image.fromarray(fr, 'RGBA')
    w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
    im = im.resize((w, h), Image.LANCZOS)
    a = np.array(im)
    a[..., 3] = np.where(a[..., 3] > 110, 255, 0)  # жёсткий пиксельный край
    canvas = np.zeros((CELL, CELL, 4), np.uint8)
    bottom = round(body_bottom(fr) * scale)
    ox = (CELL - w) // 2
    oy = FLOOR - bottom + dy
    ys, xs = max(0, -oy), max(0, -ox)
    ye, xe = min(h, CELL - oy), min(w, CELL - ox)
    canvas[oy + ys:oy + ye, ox + xs:ox + xe] = a[ys:ye, xs:xe]
    return canvas


def main():
    s1 = cut_grid('1.webp', 5, 6)   # полёт
    s2 = cut_grid('2.webp', 8, 4)   # действия
    s3 = cut_grid('3.webp', 4, 4)   # взгляд, 16 направлений

    # масштаб листа по росту сидящего кадра на нём
    def k(fr):
        return SIT_H / body_bottom(fr)
    k1, k2, k3 = k(s1[4][5]), k(s2[0][0]), k(s3[0][0])
    print('scales', round(k1, 3), round(k2, 3), round(k3, 3))

    def seq(sheet, scale, cells, lift=0):
        return [place(sheet[r][c], scale, -lift) for r, c in cells]

    FLY = 40  # высота парения в полёте
    anims = [
        # --- лист 2
        ('idle', seq(s2, k2, [(0, 0), (0, 1), (0, 0), (0, 3), (0, 0), (0, 2), (0, 0)]), 3, True,
         'Покой: дышит, моргает, шевелит ушками и хвостом'),
        ('run_right', seq(s2, k2, [(1, i) for i in range(4)]), 8, True, 'Перебежка вправо'),
        ('run_left', seq(s2, k2, [(2, i) for i in range(4)]), 8, True, 'Перебежка влево'),
        ('wave', seq(s2, k2, [(3, 0), (3, 1), (3, 2), (3, 1), (3, 2), (3, 3)]), 5, False,
         'Приветствие: машет лапой'),
        ('jump', seq(s2, k2, [(4, 0)]) + [place(s2[4][1], k2, -25), place(s2[4][2], k2, -45),
                                         place(s2[4][2], k2, -35), place(s2[4][1], k2, -15)]
         + seq(s2, k2, [(4, 3), (4, 0)]), 7, False,
         'Прыжок: пытается взлететь, машет крыльями и приземляется'),
        ('fail', seq(s2, k2, [(5, 0), (5, 1), (5, 2), (5, 2), (5, 2), (5, 3), (0, 0)]), 3, False,
         'Неудача: прижимает ушки, сворачивается клубком и расправляется'),
        ('wait', seq(s2, k2, [(6, 0), (6, 1), (6, 0), (6, 1)]), 2, True,
         'Ожидание: умоляющие глаза, наклон головы'),
        ('inspect', seq(s2, k2, [(6, 0), (6, 2), (6, 3), (6, 2), (6, 3), (6, 0)]), 3, False,
         'Проверка результата: наклоняется и осматривает'),
        ('eat', seq(s2, k2, [(7, 0), (7, 2), (7, 3), (7, 2), (7, 3), (7, 0)]), 4, True,
         'Занятость: держит и ест рыбку'),
        ('toss_fish', seq(s2, k2, [(7, 0), (7, 1), (7, 1), (7, 0), (7, 2), (7, 3)]), 4, False,
         'Подбрасывает рыбку и ловит'),
        # --- лист 1 (полёт)
        ('fly_right', seq(s1, k1, [(0, i) for i in range(6)], FLY), 10, True, 'Полёт вправо'),
        ('fly_left', seq(s1, k1, [(1, i) for i in range(6)], FLY), 10, True, 'Полёт влево'),
        ('hover', seq(s1, k1, [(2, i) for i in range(6)], FLY), 10, True, 'Парит на месте, лицом к нам'),
        ('takeoff', [place(s1[3][i], k1, -int(FLY * i / 5)) for i in range(6)], 8, False,
         'Взлёт с места'),
        ('land', [place(s1[4][i], k1, -int(FLY * (5 - i) / 5)) for i in range(6)], 8, False,
         'Приземление'),
    ]
    # --- лист 3: 16 направлений взгляда (по строкам слева направо)
    look = [place(s3[r][c], k3) for r in range(4) for c in range(4)]
    anims.append(('look', look, 6, True, 'Взгляд: 16 направлений (кадр = направление)'))

    pk.set_frame_size(CELL)
    bad = pk.export(anims, OUT, name='Дракончик', anchor=(CELL // 2, FLOOR))
    print('frames touching edge:', bad or 'none')


if __name__ == '__main__':
    main()
