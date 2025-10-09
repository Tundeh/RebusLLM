import argparse, csv, json, re, pathlib
from PIL import Image


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_hex_color(s: str):
    s = s.strip().lstrip("#")
    if len(s) == 3: s = "".join(c*2 for c in s)
    if len(s) != 6: raise ValueError("Bad hex color")
    return tuple(int(s[i:i+2], 16) for i in (0,2,4))

def resize_pad(im: Image.Image, target_w: int, target_h: int, bg_rgb=(255,255,255)) -> Image.Image:
    im = im.convert("RGB")
    iw, ih = im.size
    scale = min(target_w/iw, target_h/ih)
    nw, nh = max(1, int(round(iw*scale))), max(1, int(round(ih*scale)))
    im_resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg_rgb)
    canvas.paste(im_resized, ((target_w - nw)//2, (target_h - nh)//2))
    return canvas

def resize_crop(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    im = im.convert("RGB")
    iw, ih = im.size
    scale = max(target_w/iw, target_h/ih)
    nw, nh = max(1, int(round(iw*scale))), max(1, int(round(ih*scale)))
    im_resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - target_w)//2; top = (nh - target_h)//2
    return im_resized.crop((left, top, left+target_w, top+target_h))

def resize_stretch(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return im.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", default="data/raw/metadata.csv")
    ap.add_argument("--img_dir", default="data/raw/images")
    ap.add_argument("--out_csv", default="data/processed/benchmark.csv")
    ap.add_argument("--manifest", default="data/processed/manifest.jsonl")
    ap.add_argument("--target_size", type=int, nargs=2, metavar=("W","H"), default=[512,512])
    ap.add_argument("--mode", choices=["pad","crop","stretch"], default="pad")
    ap.add_argument("--bg", default="#FFFFFF")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    target_w, target_h = args.target_size
    bg_rgb = parse_hex_color(args.bg)

    data_root = pathlib.Path("data")
    out_dir = data_root / f"processed/images_{target_w}x{target_h}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "pad":
        resizer = lambda im: resize_pad(im, target_w, target_h, bg_rgb)
    elif args.mode == "crop":
        resizer = lambda im: resize_crop(im, target_w, target_h)
    else:
        resizer = lambda im: resize_stretch(im, target_w, target_h)

    # index images by stem
    img_dir = pathlib.Path(args.img_dir)
    index = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS}

    artifacts = data_root / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    missing, corrupt, rows = [], [], []

    with open(args.raw_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        assert {"id","answer","difficulty"} <= set(rdr.fieldnames), "CSV must have id,answer,difficulty"

        for row in rdr:
            rid = str(row["id"]).strip()
            src = index.get(rid)
            if not src:
                for ext in IMG_EXTENSIONS:
                    p = img_dir / f"{rid}{ext}"
                    if p.exists(): src = p; break
            if not src:
                missing.append(rid); continue

            try:
                with Image.open(src) as im_ctx:
                    im_ctx.load()
                    im = im_ctx.convert("RGB")
            except Exception:
                corrupt.append(rid); continue

            orig_w, orig_h = im.size
            dst = out_dir / f"{rid}.png"
            if args.overwrite or not dst.exists():
                resizer(im).save(dst, format="PNG", optimize=True)

            diff = norm_text(row["difficulty"])
            diff_map = {"e":"easy","m":"medium","h":"hard"}
            diff = diff_map.get(diff, diff)
            assert diff in {"easy","medium","hard"}, f"bad difficulty={diff} id={rid}"

            rows.append({
                "id": int(rid),
                "image": str(dst.relative_to(data_root)),
                "orig_image": str(src.relative_to(data_root)),
                "answer": norm_text(row["answer"]),
                "difficulty": diff,
                "aliases": json.dumps([]),
                "orig_w": orig_w, "orig_h": orig_h,
                "w": target_w, "h": target_h,
                "resize_mode": args.mode
            })

    if missing: (artifacts/"missing_images.txt").write_text("\n".join(missing), encoding="utf-8")
    if corrupt: (artifacts/"corrupt_images.txt").write_text("\n".join(corrupt), encoding="utf-8")

    # benchmark.csv
    headers = ["id","image","orig_image","answer","difficulty","aliases","orig_w","orig_h","w","h","resize_mode"]
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["id"]))

    # manifest.jsonl (points to processed images)
    pathlib.Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as out:
        for r in sorted(rows, key=lambda x: x["id"]):
            out.write(json.dumps({
                "id": r["id"], "image": r["image"], "gold": r["answer"],
                "difficulty": r["difficulty"], "w": r["w"], "h": r["h"]
            })+"\n")

if __name__ == "__main__":
    main()