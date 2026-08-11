# Rendering images as terminal art

Notes on how `img2quad.py`, `img2half.py` and the animation sprites work, and the traps that
cost the most time. The short version: **a terminal cell is not a pixel**, and almost every
bad-looking conversion comes from pretending it is.

## 1. What a terminal cell actually is

One cell carries three things:

| | |
|---|---|
| a glyph | one character |
| a foreground colour | applied to the glyph's "ink" |
| a background colour | applied to everything else |

So a cell is **two colours separated by a shape**. That is the entire budget. You cannot put
four different colours in a cell; you can put two, divided along whatever boundary the glyph
provides.

## 2. Subpixel families

Unicode gives block characters that carve a cell into a grid. Pick one and the cell becomes
that many subpixels:

| family | subpixels per cell | glyphs | subpixel shape on screen |
|---|---|---|---|
| half block | 1 x 2 | `▀ ▄ █` | **square** |
| quadrant | 2 x 2 | `▘▝▖▗▀▄▌▐▚▞▛▜▙▟█` | 1 wide : 2 tall |
| sextant | 2 x 3 | `🬀`-`🬻` | 1 wide : 1.33 tall |
| braille | 2 x 4 | `⠀`-`⣿` | **square** |

A 32 x 12 cell canvas is therefore 32 x 24 half-block pixels, or 64 x 24 quadrant pixels.

**Choosing between them:**

- **Half block** for photographs and imported art. Square pixels, and fg/bg give you two
  *independent* full-RGB pixels per cell — this is what `chafa`, `viu` and `timg` use.
- **Quadrant** for procedural sprites. Twice the spatial resolution in both axes, at the cost
  of the two colours being shared across four subpixels.
- **Braille** for line plots and curves. Highest resolution, but all eight dots share one
  colour and the glyphs render as visibly separated dots, not solid ink.

## 3. The aspect trap

A terminal cell is roughly **twice as tall as it is wide**. That propagates:

```
half block subpixel : cell_w x cell_h/2  ->  square
quadrant subpixel   : cell_w/2 x cell_h/2 -> 1:2, twice as tall as wide
```

So with quadrant encoding, **a square source image rendered at N x N subpixels comes out
stretched 2x vertically**. Either author pre-squashed (in Aseprite: `Sprite > Properties >
Pixel Aspect Ratio = 1:2`), or resample with the correction built in, as `img2quad.load()`
does:

```python
height = round(width * image.height / image.width / 2)   # note the / 2
```

Anything meant to look *round* needs the same correction. `Canvas.disc()` draws an ellipse
twice as wide as it is tall so it lands on screen as a circle.

## 4. The encoder: two colours per cell, chosen by error

This is the part that matters most and the part most converters get wrong.

The naive approach picks **one** colour per cell — usually the average, or a majority vote —
and uses the glyph only to mark which subpixels are covered. Every cell straddling a boundary
loses one of its two colours. Fine detail disappears: a one-pixel highlight next to a large
flat area gets outvoted and vanishes.

The better approach searches all 16 quadrant masks and picks the one reproducing the block
with the least error:

```python
for mask in range(16):
    front = [p for i, p in enumerate(quad) if mask >> i & 1]
    back  = [p for i, p in enumerate(quad) if not mask >> i & 1]
    fg, bg = mean(front), mean(back)
    score = sum_squared_error(quad, mask, fg, bg)
```

A block containing **at most two distinct colours reproduces exactly**, whatever the shape of
the boundary between them. Measured on the walking sprite across 24 frames:

| | one colour per cell | two colours per cell |
|---|---|---|
| RMSE per channel | 49.44 | **9.95** |
| cells reproduced exactly | 70.6% | **95.8%** |

Sixteen candidates per cell sounds expensive and is not: only a few hundred *distinct* 2x2
blocks exist across a whole animation, so `functools.cache` on the block tuple absorbs it. A
24-frame cycle renders in ~27 ms cold, ~18 ms warm.

## 5. Transparency

Cells with any transparent subpixel must keep a **transparent background** — emit a foreground
colour only. Give them a background and the art paints an opaque rectangle around itself,
which looks fine standalone and wrong the moment it sits beside anything.

That means edge cells carry only one colour. Interior cells get two. It is the right trade:
the edge is where transparency matters and where a second colour matters least.

```python
if any(p is None for p in quad):
    return f"\x1b[0m\x1b[38;2;{fg}m{glyph}"      # one colour, transparent background
return f"\x1b[38;2;{fg}m\x1b[48;2;{bg}m{glyph}"  # two colours, opaque
```

## 6. Gotchas that cost real time

**Gradients are expensive.** Flat-colour art compresses ~7x better than gradients: the walking
robot is 4.5 KB gzipped, a logo with a smooth sheen is 21.8 KB. More colours per cell also
means more blocks needing three or four, which the two-colour encoder can only approximate —
adding shading took our exact-reproduction rate from 95.8% down to 90.2%.

**Monochrome fallback is nearly useless.** Strip colour from a two-colour render and every
fully-covered cell collapses to `█`, because the glyph was carrying *colour* boundaries, not
coverage. Do not judge a render from a colour-stripped dump — it will look broken when it
is not. (I did this repeatedly and misdiagnosed working output as buggy.)

**Escapes dominate the byte count.** Emitting colour for every cell regardless of whether it
changed took a 24-frame cycle to 182 KB raw; suppressing repeats cut it to 114 KB, and gzip
from 4.5 KB to 3.5 KB.

**Pre-rendered art cannot reflow.** A sprite is a raster. When the terminal narrows, the only
options are clip, drop, or swap to a smaller asset — no layout engine helps. Decide the size
tiers up front.

**Author at the target size.** Pixel art does not survive resampling: a 2-pixel limb that
loses half a pixel stops reading as a limb, and a 1-pixel antenna disappears. Scaling a design
down produces mush; re-authoring at the smaller size does not.

## 7. Sizes for loading screen greetings

The loading screen leaves two different widths beside its run summary, so a greeting ships at
two sizes. They are authored separately, never scaled -- see the last point in section 6.

| slot | terminal | cells | subpixels |
|---|---|---|---|
| `small` | 80-119 columns | **24 x 12** | 48 x 24 |
| `wide` | 120+ columns | **64 x 12** | 128 x 24 |

Anything that scrolls must advance a whole number of its own pattern period across the cycle,
or it visibly jumps when the loop wraps. The greeting must also close: frame N flows into
frame 0.

## 8. Tools here

| | |
|---|---|
| `gif2anim.py` | GIF, image, or procedural sprite to a shipped greeting |
| `anim_view.py` | look at a greeting, on its own or beside a mock run summary |
| `sprites/` | procedural greetings; `canvas.py` holds the drawing surface and encoder |

Build one, then look at it:

```bash
# from an image: scaled to fit and padded, so the mark keeps its proportions
uv run python tools/gif2anim.py logo.png --size wide --name my-wide --out /tmp/out

# from a procedural sprite, which already draws at the target grid
uv run python tools/gif2anim.py --from-module tools.sprites.walker --size wide \
    --name walker-wide --out /tmp/out

uv run python tools/anim_view.py /tmp/out/my-wide.anim          # play it
uv run python tools/anim_view.py my-wide --beside               # as the screen shows it
uv run python tools/anim_view.py --all                          # compare everything shipped
```

`--beside` is the view worth trusting: a greeting can look fine alone and wrong in place --
too tall for the summary box, or so narrow the gap swallows it.

Feed the converter a PNG with a real alpha channel. GIF's 1-bit transparency gives hard edges,
and a baked-in background leaves a dark fringe on every anti-aliased pixel.
