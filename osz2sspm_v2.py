#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# osu standard playfield coordinates
OSU_W, OSU_H = 512.0, 384.0

# hitobject type bitmasks
OSU_HITCIRCLE = 1
OSU_SLIDER = 2
OSU_SPINNER = 8

DIFF_TO_GROUP = {
    "na": 0x00,
    "easy": 0x01,
    "medium": 0x02,
    "hard": 0x03,
    "logic": 0x04,
    "tasukete": 0x05,
    # docs list up to Tasukete; community sometimes uses more, but v2 spec lists these. :contentReference[oaicite:4]{index=4}
}


@dataclass
class OsuMeta:
    title: str = "Unknown Title"
    artist: str = "Unknown Artist"
    creator: str = "Unknown Mapper"
    version: str = "Converted"
    audio_filename: Optional[str] = None
    background_filename: Optional[str] = None
    mode: int = 0


def extract_osz(osz_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(osz_path, "r") as z:
        z.extractall(dest)


def parse_osu(osu_path: Path) -> Tuple[OsuMeta, List[Tuple[int, int, int, int]]]:
    section = None
    meta = OsuMeta()
    hitobjects: List[Tuple[int, int, int, int]] = []

    def kv(line: str) -> Optional[Tuple[str, str]]:
        if ":" not in line:
            return None
        k, v = line.split(":", 1)
        return k.strip(), v.strip()

    with osu_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                continue

            if section == "General":
                pair = kv(line)
                if not pair:
                    continue
                k, v = pair
                if k == "AudioFilename":
                    meta.audio_filename = v
                elif k == "Mode":
                    try:
                        meta.mode = int(v)
                    except ValueError:
                        meta.mode = 0

            elif section == "Metadata":
                pair = kv(line)
                if not pair:
                    continue
                k, v = pair
                if k == "Title":
                    meta.title = v
                elif k == "Artist":
                    meta.artist = v
                elif k == "Creator":
                    meta.creator = v
                elif k == "Version":
                    meta.version = v

            elif section == "Events":
                # Background: 0,0,"filename",xOffset,yOffset or 0,0,filename,xOffset,yOffset
                if line.startswith("0,") and meta.background_filename is None:
                    m = re.match(r'0,0,(?:"([^"]*)"|([^,]*))', line)
                    if m:
                        meta.background_filename = (m.group(1) or m.group(2) or "").strip() or None

            elif section == "HitObjects":
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        x = int(float(parts[0]))
                        y = int(float(parts[1]))
                        t = int(float(parts[2]))
                        typ = int(parts[3])
                        hitobjects.append((x, y, t, typ))
                    except ValueError:
                        pass

    return meta, hitobjects

def parse_osu_slider_end_events(osu_path: Path) -> List[Tuple[int, int, int]]:
    """
    Returns list of slider end events: (end_time_ms, end_x, end_y)

    Computes duration using:
      - latest red timing point at slider start (beatLength)
      - latest green timing point at slider start (SV multiplier)
      - Difficulty SliderMultiplier
      - slider pixelLength and repeat count
    End position is approximated using the last control point; for even repeats, end returns to head.
    """
    section = None

    slider_multiplier = 1.0
    timing_points: List[Tuple[int, float, bool]] = []  # (time, beatLength, uninherited_is_red)

    slider_lines: List[Tuple[int, int, int, int, int, float, Optional[Tuple[int, int]]]] = []
    # (x,y,t,typ,repeat,pixelLength,last_point_xy)

    def kv(line: str):
        if ":" not in line:
            return None
        k, v = line.split(":", 1)
        return k.strip(), v.strip()

    with osu_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
                continue

            if section == "Difficulty":
                pair = kv(line)
                if not pair:
                    continue
                k, v = pair
                if k == "SliderMultiplier":
                    try:
                        slider_multiplier = float(v)
                    except ValueError:
                        slider_multiplier = 1.0

            elif section == "TimingPoints":
                parts = line.split(",")
                # time, beatLength, meter, sampleSet, sampleIndex, volume, uninherited, effects
                if len(parts) >= 7:
                    try:
                        t = int(float(parts[0]))
                        beat_len = float(parts[1])
                        uninherited = int(parts[6])  # 1 = red, 0 = green
                        timing_points.append((t, beat_len, uninherited == 1))
                    except ValueError:
                        pass

            elif section == "HitObjects":
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                try:
                    x = int(float(parts[0]))
                    y = int(float(parts[1]))
                    t = int(float(parts[2]))
                    typ = int(parts[3])
                except ValueError:
                    continue

                is_slider = (typ & OSU_SLIDER) != 0
                if not is_slider:
                    continue

                # slider params at parts[5]
                # format: CurveType|x1:y1|x2:y2...,repeat,pixelLength,...
                obj_params = parts[5]
                slider_fields = obj_params.split("|")
                last_point = None
                if len(slider_fields) >= 2:
                    # last control point is last x:y token
                    for token in reversed(slider_fields[1:]):
                        if ":" in token:
                            try:
                                px, py = token.split(":")
                                last_point = (int(float(px)), int(float(py)))
                            except ValueError:
                                last_point = None
                            break

                try:
                    repeat = int(parts[6]) if len(parts) >= 7 else 1
                except ValueError:
                    repeat = 1
                try:
                    pixel_len = float(parts[7]) if len(parts) >= 8 else 0.0
                except ValueError:
                    pixel_len = 0.0

                if repeat < 1:
                    repeat = 1

                slider_lines.append((x, y, t, typ, repeat, pixel_len, last_point))

    # Helper: latest red and green timing point <= time
    timing_points.sort(key=lambda p: p[0])

    def get_red_beatlength(at_time: int) -> float:
        beat = 500.0  # default 120 BPM
        for tp_t, beat_len, is_red in timing_points:
            if tp_t > at_time:
                break
            if is_red and beat_len > 0:
                beat = beat_len
        return beat

    def get_sv_multiplier(at_time: int) -> float:
        sv = 1.0
        for tp_t, beat_len, is_red in timing_points:
            if tp_t > at_time:
                break
            if (not is_red) and beat_len < 0:
                # osu formula: SV multiplier = -100 / beatLength
                sv = (-100.0 / beat_len) if beat_len != 0 else 1.0
        if sv <= 0:
            sv = 1.0
        return sv

    events: List[Tuple[int, int, int]] = []
    for x, y, t, _typ, repeat, pixel_len, last_point in slider_lines:
        beat_len = get_red_beatlength(t)
        sv = get_sv_multiplier(t)

        # duration per span (ms)
        # velocity px per beat = slider_multiplier * 100 * sv
        vel = slider_multiplier * 100.0 * sv
        if vel <= 0:
            continue

        duration_one = (pixel_len / vel) * beat_len
        total_duration = duration_one * repeat
        end_t = int(round(t + total_duration))

        # end position (approx):
        # if repeat is odd => ends at last point, even => returns to head
        if (repeat % 2) == 0:
            end_x, end_y = x, y
        else:
            if last_point is None:
                end_x, end_y = x, y
            else:
                end_x, end_y = last_point

        events.append((end_t, end_x, end_y))

    return events



def pick_best_osu(folder: Path) -> Path:
    osu_files = list(folder.rglob("*.osu"))
    if not osu_files:
        raise SystemExit("No .osu files found inside the .osz")

    best = None
    best_count = -1
    for p in osu_files:
        meta, hos = parse_osu(p)
        if meta.mode != 0:
            continue
        if len(hos) > best_count:
            best = p
            best_count = len(hos)

    if best is None:
        raise SystemExit("No osu!standard (Mode:0) .osu found in the set.")
    return best


# --- coordinate mapping ---
def osu_to_square01(x: float, y: float) -> Tuple[float, float]:
    """
    Convert osu (512x384) into square-normalized [0..1] without stretching:
    - uniform scale based on width
    - vertical letterbox centering
    """
    nx = (x / OSU_W) - 0.5
    ny = (y / OSU_W) - (OSU_H / (2.0 * OSU_W))
    return (nx + 0.5, ny + 0.5)


def stretch_minmax(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Stretch so leftmost/rightmost/topmost/bottommost fill the whole square.
    """
    if not points:
        return points
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    rx = max_x - min_x
    ry = max_y - min_y
    if rx == 0:
        rx = 1.0
    if ry == 0:
        ry = 1.0

    out = []
    for x, y in points:
        nx = (x - min_x) / rx
        ny = (y - min_y) / ry
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
        out.append((nx, ny))
    return out


def square01_to_ss_xy(sx: float, sy: float) -> Tuple[float, float]:
    """
    Sound Space+ positions in v2 "quantum" mode use float32 where:
    0 = left/top, 1 = center, 2 = right/bottom :contentReference[oaicite:5]{index=5}
    We'll map [0..1] -> [0..2].
    """
    sx = min(1.0, max(0.0, sx))
    sy = min(1.0, max(0.0, sy))
    return (sx * 2.0, sy * 2.0)


def hitobjects_to_notes(
    hitobjects: List[Tuple[int, int, int, int]],
    include_sliders: bool,
    include_spinners: bool,
    include_slider_ends: bool = False,
    slider_end_events: Optional[List[Tuple[int, int, int]]] = None,  # (end_time_ms, end_x, end_y)
) -> List[Tuple[int, float, float]]:
    """
    Return list of (time_ms, x_float, y_float)
    """
    base: List[Tuple[int, float, float]] = []

    for x, y, t, typ in hitobjects:
        is_circle = (typ & OSU_HITCIRCLE) != 0
        is_slider = (typ & OSU_SLIDER) != 0
        is_spinner = (typ & OSU_SPINNER) != 0

        if is_circle or (include_sliders and is_slider):
            sx, sy = osu_to_square01(float(x), float(y))
            base.append((t, sx, sy))
        elif include_spinners and is_spinner:
            base.append((t, 0.5, 0.5))

    # Add slider ends (extra tap at slider end time)
    if include_slider_ends and slider_end_events:
        for end_t, end_x, end_y in slider_end_events:
            sx, sy = osu_to_square01(float(end_x), float(end_y))
            base.append((int(end_t), sx, sy))

    base.sort(key=lambda n: n[0])

    stretched = stretch_minmax([(sx, sy) for _t, sx, sy in base])

    out: List[Tuple[int, float, float]] = []
    for (t, orig_sx, orig_sy), (nsx, nsy) in zip(base, stretched):
    # If this was a spinner (center marker), force exact center AFTER scaling
        if orig_sx == 0.5 and orig_sy == 0.5:
            rx, ry = square01_to_ss_xy(0.5, 0.5)
        else:
            rx, ry = square01_to_ss_xy(nsx, nsy)

        out.append((int(t), float(rx), float(ry)))
    return out



# --- SSPM v2 writing helpers (per basils-garden/types spec) ---
def u16_str(s: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) > 65535:
        b = b[:65535]
    return struct.pack("<H", len(b)) + b


def build_marker_definitions() -> bytes:
    """
    v2 marker definitions:
    - 1 byte count
    - each def: uint16len id + 1 byte value count + list of type IDs + 0x00 terminator :contentReference[oaicite:6]{index=6}
    We define ONLY one definition: ssp_note with one value: Position (type 0x07).
    """
    out = bytearray()
    out += struct.pack("<B", 1)  # one definition
    out += u16_str("ssp_note")   # notes should be first :contentReference[oaicite:7]{index=7}
    out += struct.pack("<B", 1)  # one value
    out += struct.pack("<B", 0x07)  # position type :contentReference[oaicite:8]{index=8}
    out += struct.pack("<B", 0x00)  # end of value list :contentReference[oaicite:9]{index=9}
    return bytes(out)


def build_markers(notes: List[Tuple[int, float, float]]) -> bytes:
    """
    Each marker:
    - uint32 time_ms
    - 1 byte marker type
    - marker data per definition :contentReference[oaicite:10]{index=10}

    Marker type: since our only definition is #0 (ssp_note), marker type = 0.

    Position encoding (type 0x07):
    - 1 byte: 0x00 int grid / 0x01 quantum float :contentReference[oaicite:11]{index=11}
    - if quantum: float32 x, float32 y :contentReference[oaicite:12]{index=12}
    """
    out = bytearray()
    for t, x, y in notes:
        out += struct.pack("<I", t)
        out += struct.pack("<B", 0)      # marker type = 0 (ssp_note)
        out += struct.pack("<B", 0x01)   # quantum / float positions :contentReference[oaicite:13]{index=13}
        out += struct.pack("<ff", float(x), float(y))
    return bytes(out)


def write_sspm_v2(
    out_path: Path,
    map_id: str,
    map_name: str,
    song_name: str,
    mappers: List[str],
    difficulty_group: int,
    audio_bytes: bytes,
    cover_bytes: bytes,
    notes: List[Tuple[int, float, float]],
) -> None:
    """
    Build an SSPM v2 file from scratch using the documented layout. :contentReference[oaicite:14]{index=14}
    """
    has_audio = 1 if audio_bytes else 0
    has_cover = 1 if cover_bytes else 0

    marker_defs = build_marker_definitions()
    markers = build_markers(notes)

    # Static metadata expects SHA1 of marker and marker definitions blocks. :contentReference[oaicite:15]{index=15}
    sha1 = hashlib.sha1(marker_defs + markers).digest()

    last_ms = notes[-1][0] if notes else 0
    note_count = len(notes)
    marker_count = len(notes)  # we only write notes

    # Custom data block: 2 bytes field count, repeated fields... :contentReference[oaicite:16]{index=16}
    custom_data = struct.pack("<H", 0)  # 0 fields

    # Strings block per spec :contentReference[oaicite:17]{index=17}
    strings = bytearray()
    strings += u16_str(map_id)
    strings += u16_str(map_name)
    strings += u16_str(song_name)
    strings += struct.pack("<H", len(mappers))
    for m in mappers:
        strings += u16_str(m)

    # ---- Build file with placeholder pointers ----
    buf = bytearray()

    # Header signature + version + reserved :contentReference[oaicite:18]{index=18}
    buf += b"SS+m"                     # 0x53 53 2b 6d :contentReference[oaicite:19]{index=19}
    buf += struct.pack("<H", 2)        # version 2 :contentReference[oaicite:20]{index=20}
    buf += b"\x00\x00\x00\x00"         # reserved must be zero :contentReference[oaicite:21]{index=21}

    # Static metadata :contentReference[oaicite:22]{index=22}
    buf += sha1                        # 20 bytes
    buf += struct.pack("<I", last_ms)
    buf += struct.pack("<I", note_count)
    buf += struct.pack("<I", marker_count)
    buf += struct.pack("<B", difficulty_group)
    buf += struct.pack("<H", 0)        # rating (unknown) -> 0
    buf += struct.pack("<B", has_audio)
    buf += struct.pack("<B", has_cover)
    buf += struct.pack("<B", 0)        # requires mod -> 0

    # Pointers (10 * u64) :contentReference[oaicite:23]{index=23}
    ptr_off = len(buf)
    buf += b"\x00" * (8 * 10)

    # Strings
    buf += bytes(strings)

    # Track block offsets
    def set_ptr(idx: int, off: int, length: int) -> None:
        # idx order per spec :contentReference[oaicite:24]{index=24}
        # 0 custom off, 1 custom len, 2 audio off, 3 audio len, 4 cover off, 5 cover len,
        # 6 markerdefs off, 7 markerdefs len, 8 markers off, 9 markers len
        struct.pack_into("<Q", buf, ptr_off + idx * 8, off)
        struct.pack_into("<Q", buf, ptr_off + (idx + 1) * 8, length)

    # Custom data block
    custom_off = len(buf)
    buf += custom_data
    custom_len = len(custom_data)

    # Audio block (optional) :contentReference[oaicite:25]{index=25}
    audio_off = len(buf)
    buf += audio_bytes
    audio_len = len(audio_bytes)

    # Cover block (optional) :contentReference[oaicite:26]{index=26}
    cover_off = len(buf)
    buf += cover_bytes
    cover_len = len(cover_bytes)

    # Marker definitions block :contentReference[oaicite:27]{index=27}
    mdef_off = len(buf)
    buf += marker_defs
    mdef_len = len(marker_defs)

    # Marker block :contentReference[oaicite:28]{index=28}
    mk_off = len(buf)
    buf += markers
    mk_len = len(markers)

    # Fill pointers
    # custom
    struct.pack_into("<Q", buf, ptr_off + 0 * 8, custom_off)
    struct.pack_into("<Q", buf, ptr_off + 1 * 8, custom_len)
    # audio
    struct.pack_into("<Q", buf, ptr_off + 2 * 8, audio_off if has_audio else 0)
    struct.pack_into("<Q", buf, ptr_off + 3 * 8, audio_len if has_audio else 0)
    # cover
    struct.pack_into("<Q", buf, ptr_off + 4 * 8, cover_off if has_cover else 0)
    struct.pack_into("<Q", buf, ptr_off + 5 * 8, cover_len if has_cover else 0)
    # markerdefs
    struct.pack_into("<Q", buf, ptr_off + 6 * 8, mdef_off)
    struct.pack_into("<Q", buf, ptr_off + 7 * 8, mdef_len)
    # markers
    struct.pack_into("<Q", buf, ptr_off + 8 * 8, mk_off)
    struct.pack_into("<Q", buf, ptr_off + 9 * 8, mk_len)

    out_path.write_bytes(bytes(buf))


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert .osz/.osu to SSPM v2 (Sound Space Plus) without templates")
    ap.add_argument("input", type=Path, help="Path to .osz or .osu")
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output .sspm path")
    ap.add_argument("--difficulty", type=str, default="hard", help="na/easy/medium/hard/logic/tasukete")
    ap.add_argument("--include-sliders", action="store_true", help="Convert slider heads as notes")
    ap.add_argument("--include-spinners", action="store_true", help="Convert spinner starts as notes")
    ap.add_argument("--map-id", type=str, default=None, help="Override map ID (default: derived from title)")
    ap.add_argument("--include-slider-ends", action="store_true", help="Also convert slider ENDs into notes (in addition to slider heads).")
    ap.add_argument("--cover", type=Path, default=None, help="Optional path to cover image (overrides beatmap background).")

    args = ap.parse_args()

    inp: Path = args.input
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    diff = args.difficulty.strip().lower()
    if diff not in DIFF_TO_GROUP:
        raise SystemExit(f"Invalid difficulty '{diff}'. Choose: {', '.join(DIFF_TO_GROUP.keys())}")

    temp_ctx: Optional[tempfile.TemporaryDirectory] = None
    try:
        workdir = inp.parent
        osu_path = inp

        if inp.suffix.lower() == ".osz":
            temp_ctx = tempfile.TemporaryDirectory()
            extracted = Path(temp_ctx.name)
            extract_osz(inp, extracted)
            workdir = extracted
            osu_path = pick_best_osu(extracted)
        elif inp.suffix.lower() != ".osu":
            raise SystemExit("Input must be .osz or .osu")

        meta, hitobjects = parse_osu(osu_path)
        if meta.mode != 0:
            raise SystemExit(f"Unsupported Mode:{meta.mode}. This supports osu!standard (Mode:0) only.")

        if not meta.audio_filename:
            raise SystemExit("No AudioFilename found in the .osu.")
        audio_path = workdir / meta.audio_filename
        if not audio_path.exists():
            raise SystemExit(f"Audio file not found: {audio_path}")

        audio_bytes = audio_path.read_bytes()
        
        slider_end_events = None
        if args.include_slider_ends:
            slider_end_events = parse_osu_slider_end_events(osu_path)
        
        
        notes = hitobjects_to_notes(
            hitobjects,
            include_sliders=args.include_sliders,
            include_spinners=args.include_spinners,
            include_slider_ends=args.include_slider_ends,
            slider_end_events=slider_end_events,
        )
        if not notes:
            raise SystemExit("No notes generated.")

        cover_bytes = b""
        if args.cover is not None:
            if not args.cover.exists():
                raise SystemExit(f"Cover file not found: {args.cover}")
            cover_bytes = args.cover.read_bytes()
        elif meta.background_filename:
            cover_path = workdir / meta.background_filename
            if cover_path.exists():
                cover_bytes = cover_path.read_bytes()

        map_id = args.map_id or re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{meta.artist}-{meta.title}-{meta.version}")[:64]
        map_name = f"{meta.title} [{meta.version}]"
        song_name = f"{meta.artist} - {meta.title}"
        mappers = [meta.creator]

        write_sspm_v2(
            out_path=args.out,
            map_id=map_id,
            map_name=map_name,
            song_name=song_name,
            mappers=mappers,
            difficulty_group=DIFF_TO_GROUP[diff],
            audio_bytes=audio_bytes,
            cover_bytes=cover_bytes,
            notes=notes,
        )

        print("✅ Wrote:", args.out)
        print("✅ Notes:", len(notes))
        print("✅ Audio bytes:", len(audio_bytes))
        if cover_bytes:
            print("✅ Cover bytes:", len(cover_bytes))
        return 0

    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
