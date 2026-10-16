#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["Pillow"]
# ///
# Self-contained: `uv run tools/plotgen/gen_atlas.py` (or any PEP 723 runner).
"""Generate pre-rasterized Thai glyph atlas and metrics JSON from THSarabunNew-Bold.

Covers:
  - ASCII 0x20..0x7E
  - Thai 0x0E01..0x0E4E
  - Thai digits 0x0E50..0x0E59
  - Fallback box for missing glyphs

Outputs:
  - crates/tltk-core/data/glyph_atlas.png
  - crates/tltk-core/data/glyph_metrics.json
"""

import json
import os
import sys
from PIL import Image, ImageFont, ImageDraw

FONT_CANDIDATES = [
    os.environ.get("TLTK_THAI_FONT", ""),
    os.path.expanduser("~/.local/share/fonts/THSarabunNew-Bold-NoPUA.ttf"),
    os.path.expanduser("~/.fonts/THSarabunNew Bold.ttf"),
    "/usr/share/fonts/truetype/thai/THSarabunNew-Bold.ttf",
]

def find_font():
    for p in FONT_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Could not locate THSarabunNew-Bold font file")

def is_combining(cp):
    # Combining marks: sit above or below the base consonant with zero advance
    return cp == 0x0E31 or (0x0E34 <= cp <= 0x0E3A) or (0x0E47 <= cp <= 0x0E4E)

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    out_dir = os.path.join(repo_root, "crates/tltk-core/data")
    os.makedirs(out_dir, exist_ok=True)
    
    font_path = find_font()
    print(f"Using font: {font_path}")
    font_size = 21  # Matches matplotlib 14pt TH Sarabun New Bold at 80dpi
    font = ImageFont.truetype(font_path, size=font_size)
    
    atlas_w, atlas_h = 256, 128
    atlas_im = Image.new("L", (atlas_w, atlas_h), 0)
    curr_x, curr_y = 1, 1
    row_h = 0
    
    metrics = {}
    
    # 1. Fallback box glyph (codepoint 0)
    box_w, box_h = 8, 10
    box_im = Image.new("L", (box_w, box_h), 0)
    box_draw = ImageDraw.Draw(box_im)
    box_draw.rectangle([(0, 0), (box_w - 1, box_h - 1)], outline=255, fill=0)
    atlas_im.paste(box_im, (curr_x, curr_y))
    metrics["0"] = [box_w + 1, 0, -box_h, box_w, box_h, curr_x, curr_y]
    curr_x += box_w + 1
    row_h = max(row_h, box_h)
    
    # 2. Characters to rasterize
    codepoints = (
        list(range(0x20, 0x7F)) +
        list(range(0x0E01, 0x0E4F)) +
        list(range(0x0E50, 0x0E5A))
    )
    
    baseline_temp = 5 + font.getmetrics()[0]  # true baseline: origin + font ascent
    for cp in codepoints:
        ch = chr(cp)
        temp = Image.new("L", (48, 56), 0)
        d = ImageDraw.Draw(temp)
        # anchor="ls": x=origin, y=BASELINE — off_x/off_y then measure true
        # bearings, including how high the font designs each combining mark.
        d.text((5, baseline_temp), ch, font=font, fill=255, anchor="ls")
        bbox = temp.getbbox()
        
        comb = is_combining(cp)
        adv = 0 if comb else int(round(font.getlength(ch)))
        
        if bbox is None:
            # Empty glyph (e.g. space)
            metrics[str(cp)] = [adv, 0, 0, 0, 0, 0, 0]
            continue
            
        gw = bbox[2] - bbox[0]
        gh = bbox[3] - bbox[1]
        
        if curr_x + gw + 1 >= atlas_w:
            curr_x = 1
            curr_y += row_h + 1
            row_h = 0
            
        if curr_y + gh + 1 >= atlas_h:
            raise RuntimeError(f"Atlas overflow at codepoint U+{cp:04X} ({ch})")
            
        row_h = max(row_h, gh)
        glyph_crop = temp.crop(bbox)
        atlas_im.paste(glyph_crop, (curr_x, curr_y))
        
        off_x = bbox[0] - 5
        off_y = bbox[1] - baseline_temp
        
        metrics[str(cp)] = [adv, off_x, off_y, gw, gh, curr_x, curr_y]
        curr_x += gw + 1
        
    png_path = os.path.join(out_dir, "glyph_atlas.png")
    json_path = os.path.join(out_dir, "glyph_metrics.json")
    
    atlas_im.save(png_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))
        
    print(f"Successfully generated:")
    print(f"  {png_path} ({os.path.getsize(png_path)} bytes)")
    print(f"  {json_path} ({os.path.getsize(json_path)} bytes)")
    print(f"  Total glyphs: {len(metrics)}, max y used: {curr_y + row_h}/{atlas_h}")

if __name__ == "__main__":
    main()
