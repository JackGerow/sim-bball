"""2.5D broadcast replay of one simulated game."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from simbball.engine import _phi
from simbball.ratings import TeamModel
from simbball.teams import luminance

W, H = 1600, 900
FPS = 20
FONT_DIR = Path(r"C:\Windows\Fonts")

# 94 ft by 50 ft, seen from a sideline camera. The length runs left to right
# and stays inside the box-score gap. The width is foreshortened so the
# floor is about twice as long as it is deep, instead of nearly square.
COURT_DST = {
    (0.0, 0.0): (415.0, 292.0),
    (94.0, 0.0): (1185.0, 292.0),
    (94.0, 50.0): (1255.0, 672.0),
    (0.0, 50.0): (345.0, 672.0),
}


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if not path.exists():
        path = FONT_DIR / "arial.ttf"
    return ImageFont.truetype(str(path), size)


FONTS = {
    "score": lambda s: _font("ariblk.ttf", s),
    "bold": lambda s: _font("segoeuib.ttf", s),
    "body": lambda s: _font("segoeui.ttf", s),
    "cond": lambda s: _font("bahnschrift.ttf", s),
}


def _homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> np.ndarray:
    rows = []
    target = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        target.extend([u, v])
    solved = np.linalg.solve(np.asarray(rows, float), np.asarray(target, float))
    return np.array(
        [[solved[0], solved[1], solved[2]], [solved[3], solved[4], solved[5]], [solved[6], solved[7], 1.0]]
    )


def _project(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = matrix @ np.array([x, y, 1.0])
    point /= point[2]
    return float(point[0]), float(point[1])


def _project_many(matrix: np.ndarray, pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    arr = np.asarray(pts, float)
    hom = np.c_[arr, np.ones(len(arr))]
    out = hom @ matrix.T
    out = out[:, :2] / out[:, 2:3]
    return [(float(x), float(y)) for x, y in out]


def _q_label(quarter: int) -> str:
    if quarter <= 4:
        return f"Q{quarter}"
    if quarter == 5:
        return "OT"
    return f"{quarter - 4}OT"


def _q_banner(quarter: int) -> str:
    names = {1: "1ST QUARTER", 2: "2ND QUARTER", 3: "3RD QUARTER", 4: "4TH QUARTER"}
    if quarter in names:
        return names[quarter]
    if quarter == 5:
        return "OVERTIME"
    return f"OVERTIME {quarter - 4}"


def _fmt_clock(seconds: float) -> str:
    if seconds <= 0.08:
        return "0:00"
    total = int(math.ceil(seconds - 1e-6))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def _readable(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    if luminance(rgb) >= 120:
        return rgb
    return tuple(min(255, int(c * 0.45 + 255 * 0.55)) for c in rgb)


def _win_prob(margin: float, seconds_left: float, exp_margin: float) -> float:
    if seconds_left <= 1.0:
        if margin > 0:
            return 0.995
        if margin < 0:
            return 0.005
        return 0.5
    frac = min(1.0, seconds_left / 2880.0)
    expected = margin + exp_margin * frac
    sd = max(0.85, 12.0 * math.sqrt(frac))
    return _phi(expected / sd)


def _seconds_left(quarter: int, clock: float) -> float:
    if quarter <= 4:
        return clock + (4 - quarter) * 720.0
    return clock


def _dot_style(team: TeamModel) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    fill = team.primary
    if luminance(fill) < 48:
        fill = team.secondary if luminance(team.secondary) > 110 else (214, 218, 224)
    number = (22, 24, 28) if luminance(fill) > 155 else (255, 255, 255)
    ring = (255, 255, 255) if luminance(fill) < 185 else (16, 18, 22)
    return fill, number, ring


def _separate_dots(home: TeamModel, away: TeamModel):
    home_style = _dot_style(home)
    away_style = _dot_style(away)
    if math.dist(home_style[0], away_style[0]) >= 78:
        return home_style, away_style
    mixed = tuple(int(0.35 * c + 0.65 * s) for c, s in zip(away_style[0], away.secondary))
    number = (22, 24, 28) if luminance(mixed) > 155 else (255, 255, 255)
    ring = (255, 255, 255) if luminance(mixed) < 185 else (16, 18, 22)
    return home_style, (mixed, number, ring)


def _arc(cx: float, cy: float, radius: float, a0: float, a1: float, steps: int = 40) -> list[tuple[float, float]]:
    angles = np.linspace(a0, a1, steps)
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


def _three_point(left: bool) -> list[list[tuple[float, float]]]:
    rim_x = 5.25 if left else 88.75
    dx = math.sqrt(max(0.0, 23.75**2 - 22.0**2))
    meet = rim_x + dx if left else rim_x - dx
    baseline = 0.0 if left else 94.0
    lines = [
        [(baseline, 3.0), (meet, 3.0)],
        [(baseline, 47.0), (meet, 47.0)],
    ]
    ang = math.atan2(22.0, dx)
    if left:
        lines.append(_arc(rim_x, 25.0, 23.75, -ang, ang, 48))
    else:
        lines.append(_arc(rim_x, 25.0, 23.75, math.pi - ang, math.pi + ang, 48))
    return lines


def _stroke(draw: ImageDraw.ImageDraw, matrix: np.ndarray, pts, fill, width: int) -> None:
    pix = _project_many(matrix, pts)
    if len(pix) >= 2:
        draw.line(pix, fill=fill, width=width, joint="curve")


def _build_court(matrix_hi: np.ndarray, home_abbrev: str) -> Image.Image:
    """Wood floor, lines, and hoops at 2x resolution."""
    scale = 2
    canvas = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 0))
    inv = np.linalg.inv(matrix_hi)
    corners = _project_many(matrix_hi, [(0, 0), (94, 0), (94, 50), (0, 50)])
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    x0, x1 = max(0, int(min(xs)) - 2), min(W * scale, int(max(xs)) + 3)
    y0, y1 = max(0, int(min(ys)) - 2), min(H * scale, int(max(ys)) + 3)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    a, b, c = inv[0]
    d, e, f = inv[1]
    g, h, i = inv[2]
    denom = g * xx + h * yy + i
    cx = (a * xx + b * yy + c) / denom
    cy = (d * xx + e * yy + f) / denom
    inside = (cx >= 0) & (cx <= 94) & (cy >= 0) & (cy <= 50) & (denom > 0)
    plank = np.floor(cy / 2.15).astype(np.int32)
    light = (plank % 2) == 0
    shade = 0.74 + 0.26 * (cy / 50.0)
    rng = np.random.default_rng(7)
    noise = rng.integers(-7, 8, size=cx.shape)
    wood = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    base_r = np.where(light, 156, 124) + noise
    base_g = np.where(light, 104, 80) + noise
    base_b = np.where(light, 56, 42) + (noise // 2)
    wood[..., 0] = np.clip(base_r * shade, 0, 255)
    wood[..., 1] = np.clip(base_g * shade, 0, 255)
    wood[..., 2] = np.clip(base_b * shade, 0, 255)
    wood[..., 3] = np.where(inside, 255, 0)
    floor = Image.fromarray(wood, "RGBA")
    canvas.paste(floor, (x0, y0), floor)

    draw = ImageDraw.Draw(canvas)
    line = (244, 240, 230, 230)
    boundary = (248, 244, 236, 245)
    _stroke(draw, matrix_hi, [(0, 0), (94, 0), (94, 50), (0, 50), (0, 0)], boundary, 7)
    _stroke(draw, matrix_hi, [(47, 0), (47, 50)], line, 4)
    _stroke(draw, matrix_hi, _arc(47, 25, 6, 0, math.tau, 64), line, 4)
    _stroke(draw, matrix_hi, _arc(47, 25, 2, 0, math.tau, 32), line, 3)
    for key in (
        [(0, 17), (19, 17), (19, 33), (0, 33), (0, 17)],
        [(94, 17), (75, 17), (75, 33), (94, 33), (94, 17)],
    ):
        _stroke(draw, matrix_hi, key, line, 4)
    _stroke(draw, matrix_hi, _arc(19, 25, 6, -math.pi / 2, math.pi / 2, 36), line, 4)
    _stroke(draw, matrix_hi, _arc(75, 25, 6, math.pi / 2, math.pi * 1.5, 36), line, 4)
    _stroke(draw, matrix_hi, _arc(5.25, 25, 4, -math.pi / 2, math.pi / 2, 28), line, 4)
    _stroke(draw, matrix_hi, _arc(88.75, 25, 4, math.pi / 2, math.pi * 1.5, 28), line, 4)
    for part in _three_point(True) + _three_point(False):
        _stroke(draw, matrix_hi, part, line, 4)

    for side in ("left", "right"):
        rim_x = 5.25 if side == "left" else 88.75
        sx, sy = _project(matrix_hi, rim_x, 25)
        direction = -1 if side == "left" else 1
        bx = sx + direction * 34
        draw.line([(bx, sy - 48), (bx, sy + 34)], fill=(236, 238, 242, 240), width=10)
        draw.ellipse([sx - 34, sy - 15, sx + 34, sy + 15], outline=(214, 98, 38, 255), width=7)
        net = (236, 236, 236, 170)
        draw.line([(sx - 22, sy + 4), (sx, sy + 30)], fill=net, width=3)
        draw.line([(sx + 22, sy + 4), (sx, sy + 30)], fill=net, width=3)
        draw.line([(sx, sy + 2), (sx, sy + 30)], fill=net, width=3)

    logo_font = FONTS["score"](160)
    mark = Image.new("RGBA", (W * scale, H * scale), (0, 0, 0, 0))
    mark_draw = ImageDraw.Draw(mark)
    cx, cy = _project(matrix_hi, 47, 25)
    bbox = mark_draw.textbbox((0, 0), home_abbrev, font=logo_font)
    mark_draw.text(
        (cx - (bbox[2] - bbox[0]) / 2, cy - (bbox[3] - bbox[1]) / 2),
        home_abbrev,
        font=logo_font,
        fill=(255, 248, 236, 42),
    )
    canvas = Image.alpha_composite(canvas, mark)
    return canvas.resize((W, H), Image.Resampling.LANCZOS)


def _background(home: TeamModel, away: TeamModel, court: Image.Image) -> Image.Image:
    base = Image.new("RGB", (W, H), (7, 10, 18))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow)
    draw.ellipse([-180, -80, 620, 820], fill=away.primary + (70,))
    draw.ellipse([980, -80, 1760, 820], fill=home.primary + (70,))
    rng = np.random.default_rng(11)
    for _ in range(48):
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, H))
        r = int(rng.integers(1, 3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, int(rng.integers(18, 55))))
    glow = glow.filter(ImageFilter.GaussianBlur(36))
    base.paste(glow, (0, 0), glow)
    # Soft shadow under the floor, then the court.
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    floor = _project_many(
        _homography([(0, 0), (94, 0), (94, 50), (0, 50)], [COURT_DST[k] for k in ((0, 0), (94, 0), (94, 50), (0, 50))]),
        [(0, 0), (94, 0), (94, 50), (0, 50)],
    )
    shifted = [(x + 8, y + 16) for x, y in floor]
    shadow_draw.polygon(shifted, fill=(0, 0, 0, 160))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    base.paste(shadow, (0, 0), shadow)
    base.paste(court, (0, 0), court)
    return base


def _fit(draw: ImageDraw.ImageDraw, text: str, font_fn, max_width: float, size: int, minimum: int = 12):
    font = font_fn(size)
    while size > minimum and draw.textlength(text, font=font) > max_width:
        size -= 1
        font = font_fn(size)
    return font


def _clip(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    trimmed = text
    while trimmed and draw.textlength(trimmed + "…", font=font) > max_width:
        trimmed = trimmed[:-1]
    return trimmed + "…"


def _panel(
    team: TeamModel,
    box: list[list[float]],
    lineup: list[int],
    highlight: int | None,
    width: int = 360,
) -> Image.Image:
    starters = [i for i, player in enumerate(team.players) if player.starter]
    bench = [i for i, player in enumerate(team.players) if not player.starter]
    row_h = 30
    height = 58 + 22 + 16 + len(starters) * row_h + 20 + len(bench) * row_h + 14
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=12, fill=(10, 16, 28, 228))
    draw.rectangle([0, 0, 5, height], fill=team.primary + (255,))
    title_font = _fit(draw, team.city.upper(), FONTS["bold"], width - 28, 22, 14)
    ink = (236, 240, 244)
    draw.text((16, 10), team.city.upper(), font=title_font, fill=ink)
    head_font = FONTS["cond"](11)
    muted = (150, 160, 176)
    for label, right in (("MIN", 172), ("PTS", 204), ("FG", 252), ("REB", 290), ("AST", 324), ("PF", 350)):
        tw = draw.textlength(label, font=head_font)
        draw.text((right - tw, 38), label, font=head_font, fill=muted)
    name_font = FONTS["bold"](13)
    stat_font = FONTS["cond"](13)
    y = 56

    def section(label: str, indexes: list[int]) -> None:
        nonlocal y
        draw.text((16, y), label, font=FONTS["cond"](11), fill=team.secondary if luminance(team.secondary) > 80 else team.primary)
        y += 16
        for idx in indexes:
            player = team.players[idx]
            stats = box[idx]
            if highlight == idx:
                draw.rounded_rectangle([10, y - 2, width - 10, y + row_h - 4], radius=6, fill=team.primary + (90,))
            name_fill = ink if idx in lineup else (132, 140, 154)
            jersey = str(player.jersey)
            draw.text((16, y + 3), jersey, font=FONTS["bold"](12), fill=team.secondary if luminance(team.secondary) > 90 else (255, 255, 255))
            name = _clip(draw, player.last_upper, name_font, 92)
            draw.text((42, y + 2), name, font=name_font, fill=name_fill)
            minutes = int(round(stats[0] / 60.0))
            values = [
                str(minutes),
                str(int(stats[13])),
                f"{int(stats[1])}-{int(stats[2])}",
                str(int(stats[7])),
                str(int(stats[8])),
                str(int(stats[12])),
            ]
            for value, right in zip(values, (172, 204, 252, 290, 324, 350)):
                tw = draw.textlength(value, font=stat_font)
                draw.text((right - tw, y + 3), value, font=stat_font, fill=ink if idx in lineup else muted)
            y += row_h

    section("STARTERS", starters)
    section("BENCH", bench)
    return image


def _empty_box(team: TeamModel) -> list[list[float]]:
    return [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in team.players]


def _roles(indices: list[int], team: TeamModel) -> list[int]:
    if len(indices) < 5:
        indices = list(indices) + [indices[0]] * (5 - len(indices))
    handler = max(indices, key=lambda i: team.players[i].usage)
    big = max(indices, key=lambda i: team.players[i].dreb)
    if big == handler:
        others = [i for i in indices if i != handler]
        big = max(others, key=lambda i: team.players[i].dreb) if others else handler
    rest = [i for i in indices if i not in (handler, big)]
    while len(rest) < 3:
        rest.append(handler)
    return [handler, rest[0], rest[1], rest[2], big]


def _attack_right(quarter: int, offense_home: bool) -> bool:
    home_right = quarter % 2 == 1
    return home_right if offense_home else (not home_right)


def _spot(dx: float, dy: float, attack_right: bool) -> tuple[float, float]:
    if attack_right:
        return (min(92.5, max(1.5, 88.75 - dx)), min(48.5, max(1.5, 25 + dy)))
    return (min(92.5, max(1.5, 5.25 + dx)), min(48.5, max(1.5, 25 + dy)))


def _formation(phase: str, attack_right: bool) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if phase == "transition":
        off = [(8, 0), (24, -14), (28, 14), (40, -8), (46, 10)]
        deff = [(14, 2), (18, -10), (36, 12), (50, -10), (58, 8)]
    elif phase == "putback":
        off = [(7, 1), (12, -10), (12, 10), (16, -2), (9, 6)]
        deff = [(3, 0), (6, -7), (6, 7), (10, -3), (4, 6)]
    elif phase == "ft":
        off = [(15.5, 0), (12, -9), (12, 9), (20, -16), (22, 14)]
        deff = [(3.5, -8), (3.5, 8), (7.5, -8), (7.5, 8), (18, 0)]
    else:
        off = [(24, 0), (18, -19), (18, 19), (12, -8), (11, 10)]
        deff = [(14, 1), (10, -13), (10, 13), (5, -6), (5, 7)]
    return [_spot(*p, attack_right) for p in off], [_spot(*p, attack_right) for p in deff]


ZONE_OFFSET = {
    "paint": (5.2, 0.0),
    "mid_left": (14.0, -8.0),
    "mid_right": (14.0, 8.0),
    "elbow": (16.0, 0.0),
    "corner_left": (2.6, -21.0),
    "corner_right": (2.6, 21.0),
    "wing_left": (15.0, -18.0),
    "wing_right": (15.0, 18.0),
    "top": (23.2, 0.0),
}


def _zone_xy(zone: str, attack_right: bool) -> tuple[float, float]:
    return _spot(*ZONE_OFFSET.get(zone, (16, 0)), attack_right)


def _rim(attack_right: bool) -> tuple[float, float]:
    return (88.75, 25.0) if attack_right else (5.25, 25.0)


def _end_positions(play: dict, home: TeamModel, away: TeamModel) -> dict[tuple[str, int], tuple[float, float]]:
    off = play["off"]
    defense = "A" if off == "H" else "H"
    attack_right = _attack_right(play["q"], off == "H")
    phase = play.get("phase") or "halfcourt"
    if play["type"] in {"ft"}:
        phase = "ft"
    if play["type"] in {"intro", "quarter", "final"}:
        # Teams lined up in their own halves for the tip and the breaks.
        positions = {}
        # Keep both fives in their own half, clear of the center score card.
        for side, team, spots in (
            ("A", away, [(16, 25), (10, 8), (10, 42), (24, 14), (24, 36)]),
            ("H", home, [(78, 25), (84, 8), (84, 42), (70, 14), (70, 36)]),
        ):
            lineup = play["lineup"][side][:5]
            for idx, spot in zip(_roles(lineup, team), spots):
                positions[(side, idx)] = spot
        return positions
    off_team = home if off == "H" else away
    def_team = away if off == "H" else home
    off_roles = _roles(play["lineup"][off], off_team)
    def_roles = _roles(play["lineup"][defense], def_team)
    off_spots, def_spots = _formation(phase, attack_right)
    positions = {}
    for idx, spot in zip(off_roles, off_spots):
        positions[(off, idx)] = spot
    for idx, spot in zip(def_roles, def_spots):
        positions[(defense, idx)] = spot
    shooter = play.get("shooter")
    if play["type"] in {"shot", "foul"} and shooter is not None and play.get("zone"):
        positions[(off, shooter)] = _zone_xy(play["zone"], attack_right)
    rebound = play.get("rebound")
    if rebound and play["type"] == "shot" and not play.get("made"):
        rim = _rim(attack_right)
        positions[rebound] = ((positions.get(rebound, rim)[0] + rim[0]) / 2, (positions.get(rebound, rim)[1] + rim[1]) / 2)
    return positions


def _smooth(t: float) -> float:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return t * t * (3 - 2 * t)


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    s = _smooth(t)
    return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s)


def _ball_point(play: dict, positions: dict, t: float) -> tuple[float, float, float]:
    """Court x, y and a screen-space arc height."""
    off = play["off"]
    attack_right = _attack_right(play["q"], off == "H")
    rim = _rim(attack_right)
    shooter = play.get("shooter")
    shooter_at = positions.get((off, shooter), rim) if shooter is not None else rim
    assist = play.get("assist")
    assist_at = positions.get((off, assist), shooter_at) if assist is not None else shooter_at

    def mix(a, b, local, arc):
        local = _smooth(local)
        return (a[0] + (b[0] - a[0]) * local, a[1] + (b[1] - a[1]) * local, math.sin(math.pi * local) * arc)

    kind = play["type"]
    if kind in {"intro", "quarter", "final"}:
        return (47.0, 25.0, 18 + 8 * math.sin(t * math.pi))
    if kind == "sub":
        return 47.0, 25.0, 0.0
    if kind == "ft":
        if t < 0.25:
            return shooter_at[0], shooter_at[1], 0
        local = (t - 0.25) / 0.55
        if play.get("made") or t < 0.8:
            x, y, arc = mix(shooter_at, rim, min(1.0, local), 26)
            return x, y, arc
        rebound = play.get("rebound")
        target = positions.get(rebound, rim) if rebound else (rim[0], 8 if rim[1] > 25 else 42)
        return mix(rim, target, (t - 0.8) / 0.2, 16)
    if kind == "turnover":
        if play["text"] == "STEAL" and play.get("actor"):
            stealer = positions.get(tuple(play["actor"]), shooter_at)
            return mix(shooter_at, stealer, t, 20)
        return shooter_at[0], shooter_at[1], 4 * math.sin(t * math.pi)
    if kind == "foul":
        return mix(shooter_at, _lerp(shooter_at, rim, 0.35), t, 12)
    # Shot.
    release = 0.18
    if assist is not None and t < release:
        return mix(assist_at, shooter_at, t / release, 14)
    flight_end = 0.68 if play.get("made") else 0.62
    if t < flight_end:
        local = 0 if t < release else (t - release) / (flight_end - release)
        arc = 52 if play.get("zone") in {"top", "wing_left", "wing_right", "corner_left", "corner_right"} else 36
        return mix(shooter_at, rim, local, arc)
    if play.get("made"):
        # Drop through the rim.
        drop = _spot(1.2, 0, attack_right)
        return mix(rim, drop, (t - flight_end) / (1 - flight_end), 4)
    rebound = play.get("rebound")
    target = positions.get(rebound, _spot(8, 6, attack_right)) if rebound else _spot(2, 18, attack_right)
    return mix(rim, target, (t - flight_end) / (1 - flight_end), 18)


def _frames_for(play: dict) -> int:
    kind = play["type"]
    if kind == "intro":
        return 28
    if kind == "final":
        return 42
    if kind == "quarter":
        return 12
    if kind == "sub":
        return 7
    if kind == "ft":
        return 7
    if kind == "foul":
        return 7
    if kind == "turnover":
        return 9
    if kind == "shot":
        return 12 if play.get("made") else 11
    return 10


def _draw_scoreboard(draw, away, home, score, quarter, clock, shot_clock, fouls, wp):
    board_w = 860
    left = (W - board_w) // 2
    top = 16
    height = 102
    gap = 8
    side_w = 348
    mid_w = board_w - side_w * 2 - gap * 2
    away_box = [left, top, left + side_w, top + height]
    mid_box = [away_box[2] + gap, top, away_box[2] + gap + mid_w, top + height]
    home_box = [mid_box[2] + gap, top, mid_box[2] + gap + side_w, top + height]
    draw.rounded_rectangle(away_box, radius=10, fill=away.primary)
    draw.rounded_rectangle(home_box, radius=10, fill=home.primary)
    draw.rounded_rectangle(mid_box, radius=10, fill=(16, 20, 30))

    def side_text(box, team, points, foul_n, name_on_left):
        score_font = FONTS["score"](50)
        score_text = str(int(points))
        ink = team.ink
        sw = draw.textlength(score_text, font=score_font)
        if name_on_left:
            name_font = _fit(draw, team.name.upper(), FONTS["bold"], box[2] - box[0] - sw - 40, 20, 12)
            draw.text((box[0] + 14, box[1] + 16), team.name.upper(), font=name_font, fill=ink)
            draw.text((box[2] - 14 - sw, box[1] + 32), score_text, font=score_font, fill=ink)
            foul_x = box[0] + 14
        else:
            name_font = _fit(draw, team.name.upper(), FONTS["bold"], box[2] - box[0] - sw - 40, 20, 12)
            draw.text((box[0] + 14, box[1] + 32), score_text, font=score_font, fill=ink)
            nw = draw.textlength(team.name.upper(), font=name_font)
            draw.text((box[2] - 14 - nw, box[1] + 16), team.name.upper(), font=name_font, fill=ink)
            foul_x = box[2] - 132
        small = FONTS["cond"](12)
        draw.text((foul_x, box[3] - 22), f"FOULS {foul_n}", font=small, fill=ink)
        for dot in range(5):
            cx = foul_x + 62 + dot * 12
            cy = box[3] - 14
            filled = dot < min(5, foul_n)
            draw.ellipse([cx, cy - 3, cx + 7, cy + 4], fill=ink if filled else None, outline=ink, width=1)
        if foul_n >= 5:
            draw.text((foul_x + 62, box[1] + 62), "BONUS", font=FONTS["bold"](10), fill=(255, 214, 80) if luminance(team.primary) < 140 else (40, 30, 0))

    side_text(away_box, away, score[1], fouls[1], True)
    side_text(home_box, home, score[0], fouls[0], False)
    q_font = FONTS["bold"](16)
    clock_font = FONTS["score"](32)
    q_text = _q_label(quarter)
    clock_text = _fmt_clock(clock)
    qw = draw.textlength(q_text, font=q_font)
    draw.text(((mid_box[0] + mid_box[2]) / 2 - qw / 2, mid_box[1] + 8), q_text, font=q_font, fill=(176, 186, 198))
    cw = draw.textlength(clock_text, font=clock_font)
    draw.text(((mid_box[0] + mid_box[2]) / 2 - cw / 2, mid_box[1] + 26), clock_text, font=clock_font, fill=(245, 247, 250))
    shot = f"SHOT {int(shot_clock)}"
    sw = draw.textlength(shot, font=FONTS["bold"](12))
    pill = [(mid_box[0] + mid_box[2]) / 2 - sw / 2 - 8, mid_box[3] - 28, (mid_box[0] + mid_box[2]) / 2 + sw / 2 + 8, mid_box[3] - 8]
    draw.rounded_rectangle(pill, radius=4, fill=(232, 176, 46))
    draw.text((pill[0] + 8, pill[1] + 1), shot, font=FONTS["bold"](12), fill=(20, 16, 8))

    # Win probability.
    bar_top = top + height + 10
    away_pct = int(round((1 - wp) * 100))
    home_pct = 100 - away_pct
    pct_font = FONTS["score"](22)

    draw.text((left, bar_top - 2), f"{away_pct}%", font=pct_font, fill=_readable(away.primary))
    home_label = f"{home_pct}%"
    hw = draw.textlength(home_label, font=pct_font)
    draw.text((left + board_w - hw, bar_top - 2), home_label, font=pct_font, fill=_readable(home.primary))
    label = "WIN PROBABILITY"
    lab_font = FONTS["cond"](11)
    lw = draw.textlength(label, font=lab_font)
    draw.text(((W - lw) / 2, bar_top + 4), label, font=lab_font, fill=(150, 158, 170))
    bar_x = left + 78
    bar_w = board_w - 156
    bar_y = bar_top + 22
    away_w = int(bar_w * (1 - wp))
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 8], radius=4, fill=home.primary)
    if away_w > 2:
        draw.rounded_rectangle([bar_x, bar_y, bar_x + max(away_w, 8), bar_y + 8], radius=4, fill=away.primary)
    knob = bar_x + away_w
    draw.ellipse([knob - 7, bar_y - 4, knob + 7, bar_y + 12], fill=(245, 247, 250))
    draw.ellipse([knob - 4, bar_y - 1, knob + 4, bar_y + 9], fill=(20, 24, 32))


def _nameplate(play, home, away, t: float) -> Image.Image | None:
    if play["type"] not in {"shot", "ft", "turnover", "foul"}:
        return None
    side, idx = play["actor"]
    team = home if side == "H" else away
    player = team.players[idx]
    action = play["text"]
    result = play.get("result") or ""
    show_result = t >= 0.62 and result in {"MADE", "MISSED", "BLOCKED", "STEAL", "TURNOVER"}
    right = result if show_result else action
    if show_result and result == "MADE":
        right_fill = (22, 140, 78)
        right_ink = (255, 255, 255)
    elif show_result and result == "MISSED":
        right_fill = (176, 48, 52)
        right_ink = (255, 255, 255)
    elif show_result and result == "BLOCKED":
        right_fill = (214, 122, 36)
        right_ink = (20, 14, 8)
    else:
        right_fill = (232, 184, 56)
        right_ink = (24, 18, 8)
    left_font = FONTS["bold"](16)
    right_font = FONTS["bold"](16)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left_text = f"{player.jersey}  {player.last_upper}"
    lw = dummy.textlength(left_text, font=left_font)
    rw = dummy.textlength(right, font=right_font)
    height = 32
    gap = 4
    image = Image.new("RGBA", (int(lw + rw + 36 + gap), height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([0, 0, lw + 18, height - 1], radius=6, fill=team.primary + (245,))
    draw.text((8, 5), left_text, font=left_font, fill=team.ink)
    x2 = lw + 18 + gap
    draw.rounded_rectangle([x2, 0, x2 + rw + 16, height - 1], radius=6, fill=right_fill + (245,))
    draw.text((x2 + 8, 5), right, font=right_font, fill=right_ink)
    return image


def _banner(team: TeamModel, phase: str) -> Image.Image:
    label = {"transition": "transition", "putback": "second chance", "ft": "free throw"}.get(phase, "half court")
    title = f"{team.name.upper()} BALL"
    image = Image.new("RGBA", (520, 42), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = FONTS["bold"](18)
    sub = FONTS["body"](16)
    tw = draw.textlength(title, font=font)
    sw = draw.textlength(label, font=sub)
    width = int(tw + sw + 36)
    image = Image.new("RGBA", (width, 42), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    ink = team.ink
    draw.rounded_rectangle([0, 0, tw + 20, 41], radius=6, fill=team.primary + (240,))
    draw.text((10, 8), title, font=font, fill=ink)
    draw.text((tw + 28, 9), label, font=sub, fill=(210, 216, 224))
    return image


def _center_card(lines: list[tuple[str, object, tuple]]) -> Image.Image:
    """lines of (text, font, color)."""
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = 0
    measured = []
    for text, font, color in lines:
        bbox = dummy.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], max(bbox[3] - bbox[1], getattr(font, "size", 18))
        width = max(width, tw)
        measured.append((text, font, color, tw, th))
    height = 24 + sum(th + 14 for _, _, _, _, th in measured)
    image = Image.new("RGBA", (int(width + 56), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([0, 0, image.width - 1, image.height - 1], radius=14, fill=(8, 12, 20, 215))
    y = 16
    for text, font, color, tw, th in measured:
        draw.text(((image.width - tw) / 2, y), text, font=font, fill=color)
        y += th + 14
    return image


def _paste(base: Image.Image, overlay: Image.Image, xy: tuple[int, int]) -> None:
    base.paste(overlay, xy, overlay)


def _player_radius(court_y: float) -> int:
    return int(9 + 13 * (court_y / 50.0))


def render_game(
    home: TeamModel,
    away: TeamModel,
    plays: list[dict],
    exp_margin: float,
    periods: list[tuple[int, int]],
    out_path: Path,
    stills_dir: Path | None = None,
    max_events: int | None = None,
    arena: str = "",
    n_sims: int = 10000,
    mode_count: int = 0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src = [(0.0, 0.0), (94.0, 0.0), (94.0, 50.0), (0.0, 50.0)]
    dst = [COURT_DST[p] for p in src]
    matrix = _homography(src, dst)
    matrix_hi = _homography(src, [(x * 2, y * 2) for x, y in dst])
    court = _build_court(matrix_hi, home.abbrev)
    # Away glow on the left, home glow on the right, matching the scoreboard.
    base = _background(home, away, court)
    home_dot, away_dot = _separate_dots(home, away)
    dots = {"H": home_dot, "A": away_dot}
    teams = {"H": home, "A": away}

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    log_path = out_path.with_suffix(".ffmpeg.log")
    log_file = open(log_path, "wb")
    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
            "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "veryfast", "-movflags", "+faststart", str(out_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=log_file,
    )
    events = plays if max_events is None else plays[:max_events]
    if max_events is not None and plays and plays[-1]["type"] == "final" and events[-1]["type"] != "final":
        events = events + [plays[-1]]
    prev_pos: dict[tuple[str, int], tuple[float, float]] = {}
    prev_box = {"H": _empty_box(home), "A": _empty_box(away)}
    prev_fouls = (0, 0)
    still_targets = {"intro": None, "action": None, "final": None}
    frame_index = 0
    try:
        for play in events:
            end_pos = _end_positions(play, home, away)
            if play["type"] in {"intro", "quarter", "final"}:
                start_pos = dict(end_pos)
            else:
                start_pos = {key: prev_pos.get(key, (47.0, 48.5)) for key in end_pos}
            nframes = _frames_for(play)
            off_team = teams[play["off"]]
            plate = None
            banner = None
            if play["type"] in {"shot", "ft", "turnover", "foul"}:
                banner = _banner(off_team, play.get("phase") or "halfcourt")
            actor = play.get("actor")
            highlight = actor[1] if actor and actor[0] else None
            # Panels for the pre-resolve and post-resolve box.
            panels_before = (
                _panel(away, prev_box["A"], play["lineup"]["A"], highlight if actor and actor[0] == "A" else None),
                _panel(home, prev_box["H"], play["lineup"]["H"], highlight if actor and actor[0] == "H" else None),
            )
            panels_after = (
                _panel(away, play["box"]["A"], play["lineup"]["A"], highlight if actor and actor[0] == "A" else None),
                _panel(home, play["box"]["H"], play["lineup"]["H"], highlight if actor and actor[0] == "H" else None),
            )
            card = None
            if play["type"] == "intro":
                card = _center_card(
                    [
                        (f"{n_sims:,} SIMULATIONS", FONTS["cond"](16), (232, 184, 56)),
                        (f"{away.city.upper()} {away.name.upper()}", FONTS["bold"](26), away.secondary if luminance(away.primary) < 90 else away.primary),
                        ("AT", FONTS["cond"](14), (160, 168, 180)),
                        (f"{home.city.upper()} {home.name.upper()}", FONTS["bold"](26), home.secondary if luminance(home.primary) < 90 else home.primary),
                        (arena or f"{n_sims:,} simulated games", FONTS["body"](15), (186, 194, 206)),
                        (
                            f"Most common score · {mode_count:,} of {n_sims:,}" if mode_count else f"Most common score in {n_sims:,} sims",
                            FONTS["body"](14),
                            (150, 158, 170),
                        ),
                    ]
                )
            elif play["type"] == "quarter":
                card = _center_card(
                    [
                        (_q_banner(play["q"]), FONTS["score"](36), (245, 247, 250)),
                        (f"{away.abbrev}  {play['score_after'][1]}    {play['score_after'][0]}  {home.abbrev}", FONTS["bold"](22), (232, 184, 56)),
                    ]
                )
            elif play["type"] == "sub":
                card = _center_card([(play.get("sub_text") or "SUBSTITUTION", FONTS["bold"](18), (245, 247, 250))])
            elif play["type"] == "final":
                bits = []
                labels = ["1", "2", "3", "4"]
                for index, (hpts, apts) in enumerate(periods):
                    label = labels[index] if index < 4 else ("OT" if index == 4 else f"OT{index - 3}")
                    bits.append(f"Q{label} {apts}-{hpts}")
                card = _center_card(
                    [
                        ("FINAL", FONTS["score"](26), (232, 184, 56)),
                        (f"{away.abbrev}   {play['score_after'][1]}", FONTS["score"](42), _readable(away.primary)),
                        (f"{home.abbrev}   {play['score_after'][0]}", FONTS["score"](42), _readable(home.primary)),
                        ("  ·  ".join(bits), FONTS["cond"](15), (186, 194, 206)),
                        (
                            f"Most common score · {mode_count:,} of {n_sims:,}" if mode_count else f"Most common score in {n_sims:,} sims",
                            FONTS["body"](14),
                            (150, 158, 170),
                        ),
                    ]
                )
            for step in range(nframes):
                t = 0 if nframes == 1 else step / (nframes - 1)
                resolved = t >= 0.62
                clock = play["clock0"] + (play["clock1"] - play["clock0"]) * t
                if play["type"] == "shot":
                    shot_clock = 24 + (play["shot_clock"] - 24) * min(1.0, t / 0.7)
                elif play["type"] == "ft":
                    shot_clock = 24
                else:
                    shot_clock = play["shot_clock"]
                score = play["score_after"] if resolved else play["score_before"]
                fouls = play["fouls"] if resolved else prev_fouls
                margin = score[0] - score[1]
                wp = _win_prob(margin, _seconds_left(play["q"], clock), exp_margin)
                frame = base.copy()
                draw = ImageDraw.Draw(frame)
                panels = panels_after if resolved else panels_before
                _paste(frame, panels[0], (10, 176))
                _paste(frame, panels[1], (W - 10 - panels[1].width, 176))
                # Players back to front.
                placed = []
                for key, spot_end in end_pos.items():
                    spot = _lerp(start_pos[key], spot_end, t)
                    side, idx = key
                    sx, sy = _project(matrix, spot[0], spot[1])
                    placed.append((spot[1], sx, sy, side, idx))
                placed.sort()
                for i in range(len(placed)):
                    for j in range(i):
                        dx = placed[i][1] - placed[j][1]
                        dy = placed[i][2] - placed[j][2]
                        dist = math.hypot(dx, dy) or 1.0
                        if dist < 30:
                            push = (30 - dist) / 2
                            placed[i] = (
                                placed[i][0],
                                placed[i][1] + dx / dist * push,
                                placed[i][2] + dy / dist * push,
                                placed[i][3],
                                placed[i][4],
                            )
                number_font_cache: dict[int, ImageFont.FreeTypeFont] = {}
                for depth, sx, sy, side, idx in placed:
                    fill, number, ring = dots[side]
                    radius = _player_radius(depth)
                    draw.ellipse([sx - radius, sy - radius * 0.28, sx + radius, sy + radius * 0.42], fill=(28, 16, 8))
                    draw.ellipse([sx - radius, sy - radius, sx + radius, sy + radius], fill=ring)
                    inset = 2
                    draw.ellipse(
                        [sx - radius + inset, sy - radius + inset, sx + radius - inset, sy + radius - inset],
                        fill=fill,
                    )
                    jersey = str(teams[side].players[idx].jersey)
                    size = radius if len(jersey) == 1 else max(9, radius - 2)
                    font = number_font_cache.get(size) or FONTS["bold"](size)
                    number_font_cache[size] = font
                    bbox = draw.textbbox((0, 0), jersey, font=font)
                    draw.text(
                        (sx - (bbox[2] - bbox[0]) / 2, sy - (bbox[3] - bbox[1]) / 2 - 1),
                        jersey,
                        font=font,
                        fill=number,
                    )
                bx, by, lift = _ball_point(play, end_pos, t)
                bsx, bsy = _project(matrix, bx, by)
                bsy -= lift
                br = 7 + int(4 * (by / 50.0))
                draw.ellipse([bsx - br, bsy - br, bsx + br, bsy + br], fill=(232, 110, 28))
                draw.ellipse([bsx - br + 2, bsy - br + 2, bsx - 1, bsy - 1], fill=(255, 196, 120))
                if play["type"] == "shot" and play.get("made") and t > 0.64:
                    pop = f"+{play['points']}"
                    fade = 1 - (t - 0.64) / 0.36
                    pop_font = FONTS["score"](28)
                    rx, ry = _project(matrix, *_rim(_attack_right(play["q"], play["off"] == "H")))
                    draw.text((rx - 10, ry - 36 - (1 - fade) * 16), pop, font=pop_font, fill=(255, 255, 255))
                _draw_scoreboard(draw, away, home, score, play["q"], clock, max(0, shot_clock), fouls, wp)
                if play["type"] in {"shot", "ft", "turnover", "foul"}:
                    plate = _nameplate(play, home, away, t)
                    if plate and actor:
                        anchor = end_pos.get(tuple(actor))
                        if anchor:
                            ax, ay = _project(matrix, *anchor)
                            px = int(min(max(340, ax - plate.width / 2), W - 340 - plate.width))
                            py = int(ay - _player_radius(anchor[1]) - plate.height - 12)
                            if py < 176:
                                py = int(ay + _player_radius(anchor[1]) + 8)
                            _paste(frame, plate, (px, py))
                    if banner:
                        _paste(frame, banner, ((W - banner.width) // 2, H - 58))
                if card is not None and play["type"] != "shot":
                    card_y = 390 if play["type"] == "sub" else 348
                    _paste(frame, card, ((W - card.width) // 2, card_y))
                draw.text((22, H - 28), "sim-bball", font=FONTS["cond"](14), fill=(140, 148, 160))
                proc.stdin.write(frame.tobytes())
                if stills_dir is not None:
                    if play["type"] == "intro" and still_targets["intro"] is None and step == min(8, nframes - 1):
                        still_targets["intro"] = frame.copy()
                    if (
                        still_targets["action"] is None
                        and play["type"] == "shot"
                        and frame_index > 80
                        and step == nframes - 1
                    ):
                        still_targets["action"] = frame.copy()
                    if play["type"] == "final" and step == min(10, nframes - 1):
                        still_targets["final"] = frame.copy()
                frame_index += 1
            prev_pos = end_pos
            prev_box = play["box"]
            prev_fouls = play["fouls"]
    finally:
        if proc.stdin:
            proc.stdin.close()
        code = proc.wait()
        log_file.close()
    if code != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg exited {code}\n{detail}")
    if stills_dir is not None:
        stills_dir.mkdir(parents=True, exist_ok=True)
        for name, image in still_targets.items():
            if image is not None:
                image.convert("RGB").save(stills_dir / f"still_{name}.jpg", quality=90)
