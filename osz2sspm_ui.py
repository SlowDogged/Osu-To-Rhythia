import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import tempfile
import re

# Import your existing converter script (must be in same folder)
import osz2sspm_v2 as conv


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("osz2sspm v2 - UI")
        self.geometry("760x420")
        self.minsize(760, 420)

        self.input_path: Path | None = None
        self.temp_ctx: tempfile.TemporaryDirectory | None = None
        self.workdir: Path | None = None

        # list of tuples: (display_name, osu_path, meta)
        self.osu_entries: list[tuple[str, Path, conv.OsuMeta]] = []

        # UI vars
        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()
        self.var_set_diff_group = tk.StringVar(value="hard")
        self.var_include_sliders = tk.BooleanVar(value=True)
        self.var_include_spinners = tk.BooleanVar(value=False)
        self.var_include_slider_ends = tk.BooleanVar(value=False)
        self.var_selected_osu = tk.StringVar()  # display string for combobox
        self.var_cover = tk.StringVar()  # optional custom cover image path

        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        # Row: input
        row1 = ttk.Frame(frm)
        row1.pack(fill="x")
        ttk.Label(row1, text="Input (.osz or .osu):", width=18).pack(side="left")
        ttk.Entry(row1, textvariable=self.var_input).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row1, text="Browse…", command=self.pick_input).pack(side="left")

        # Row: difficulty inside osz
        row2 = ttk.Frame(frm)
        row2.pack(fill="x")
        ttk.Label(row2, text="Chart in set:", width=18).pack(side="left")
        self.cmb_osu = ttk.Combobox(row2, textvariable=self.var_selected_osu, state="readonly")
        self.cmb_osu.pack(side="left", fill="x", expand=True, padx=6)
        self.cmb_osu.bind("<<ComboboxSelected>>", lambda e: self._auto_output_name())

        # Row: output
        row3 = ttk.Frame(frm)
        row3.pack(fill="x")
        ttk.Label(row3, text="Output (.sspm):", width=18).pack(side="left")
        ttk.Entry(row3, textvariable=self.var_output).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row3, text="Save As…", command=self.pick_output).pack(side="left")

        # Row: optional cover image
        row_cover = ttk.Frame(frm)
        row_cover.pack(fill="x")
        ttk.Label(row_cover, text="Cover image (optional):", width=18).pack(side="left")
        ttk.Entry(row_cover, textvariable=self.var_cover).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row_cover, text="Browse…", command=self.pick_cover).pack(side="left")

        # Options
        opt = ttk.LabelFrame(frm, text="Options")
        opt.pack(fill="x", pady=10)

        opt_row1 = ttk.Frame(opt)
        opt_row1.pack(fill="x", padx=10, pady=6)

        ttk.Label(opt_row1, text="SSPM difficulty group:", width=20).pack(side="left")
        cmb = ttk.Combobox(
            opt_row1,
            textvariable=self.var_set_diff_group,
            state="readonly",
            values=list(conv.DIFF_TO_GROUP.keys()),
            width=14,
        )
        cmb.pack(side="left")

        opt_row2 = ttk.Frame(opt)
        opt_row2.pack(fill="x", padx=10, pady=6)
        ttk.Checkbutton(opt_row2, text="Include slider heads as notes", variable=self.var_include_sliders).pack(side="left")
        ttk.Checkbutton(opt_row2, text="Include spinner starts as notes", variable=self.var_include_spinners).pack(side="left", padx=14)
        ttk.Checkbutton(opt_row2, text="Include slider ends as notes", variable=self.var_include_slider_ends).pack(side="left", padx=14)


        # Convert button
        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x", pady=10)

        ttk.Button(btn_row, text="Convert", command=self.convert, width=18).pack(side="left")
        ttk.Button(btn_row, text="Clear", command=self.clear, width=18).pack(side="left", padx=10)

        # Log box
        self.txt = tk.Text(frm, height=10, wrap="word")
        self.txt.pack(fill="both", expand=True, pady=8)
        self._log("Ready. Pick a .osz or .osu to begin.")

    def _log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def _cleanup_temp(self):
        if self.temp_ctx is not None:
            try:
                self.temp_ctx.cleanup()
            except Exception:
                pass
        self.temp_ctx = None
        self.workdir = None

    def _on_close(self):
        self._cleanup_temp()
        self.destroy()

    def clear(self):
        self._cleanup_temp()
        self.input_path = None
        self.osu_entries = []
        self.var_input.set("")
        self.var_output.set("")
        self.var_cover.set("")
        self.var_selected_osu.set("")
        self.cmb_osu["values"] = []
        self.txt.delete("1.0", "end")
        self._log("Cleared.")

    def pick_input(self):
        fp = filedialog.askopenfilename(
            title="Select .osz or .osu",
            filetypes=[("osu files", "*.osz *.osu"), ("All files", "*.*")],
        )
        if not fp:
            return
        self.load_input(Path(fp))

    def load_input(self, inp: Path):
        self._cleanup_temp()
        self.input_path = inp
        self.var_input.set(str(inp))

        if not inp.exists():
            messagebox.showerror("Error", f"Input not found:\n{inp}")
            return

        suffix = inp.suffix.lower()
        self.osu_entries = []

        try:
            if suffix == ".osz":
                self.temp_ctx = tempfile.TemporaryDirectory()
                extracted = Path(self.temp_ctx.name)
                conv.extract_osz(inp, extracted)
                self.workdir = extracted
                self._scan_osu_files(extracted)
            elif suffix == ".osu":
                self.workdir = inp.parent
                meta, _ = conv.parse_osu(inp)
                if meta.mode != 0:
                    raise RuntimeError(f"Unsupported Mode:{meta.mode} (needs Mode:0).")
                disp = f"{meta.artist} - {meta.title} [{meta.version}]  (from .osu)"
                self.osu_entries = [(disp, inp, meta)]
            else:
                raise RuntimeError("Input must be .osz or .osu")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._cleanup_temp()
            return

        values = [d for (d, _, _) in self.osu_entries]
        self.cmb_osu["values"] = values

        if values:
            self.var_selected_osu.set(values[0])
            self._auto_output_name()

        self._log(f"Loaded: {inp}")
        if suffix == ".osz":
            self._log(f"Found {len(values)} osu!standard difficulty file(s) inside the set.")
        else:
            self._log("Using the selected .osu file directly.")

    def _scan_osu_files(self, folder: Path):
        osu_files = list(folder.rglob("*.osu"))
        if not osu_files:
            raise RuntimeError("No .osu files found inside the .osz")

        entries = []
        for p in osu_files:
            meta, hos = conv.parse_osu(p)
            if meta.mode != 0:
                continue
            # make a nice label
            disp = f"{meta.artist} - {meta.title} [{meta.version}]  ({len(hos)} objects)"
            entries.append((disp, p, meta))

        if not entries:
            raise RuntimeError("No osu!standard (Mode:0) .osu found in the set.")

        # sort: most objects first (often “harder”)
        entries.sort(key=lambda t: int(re.search(r"\((\d+) objects\)$", t[0]).group(1)), reverse=True)
        self.osu_entries = entries

    def pick_output(self):
        fp = filedialog.asksaveasfilename(
            title="Save .sspm as",
            defaultextension=".sspm",
            filetypes=[("SSPM files", "*.sspm")],
        )
        if not fp:
            return
        self.var_output.set(fp)

    def pick_cover(self):
        fp = filedialog.askopenfilename(
            title="Select cover image (optional)",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.webp"), ("All files", "*.*")],
        )
        if fp:
            self.var_cover.set(fp)

    def _auto_output_name(self):
        # auto-suggest output name near the input file
        if not self.input_path or not self.osu_entries:
            return
        disp = self.var_selected_osu.get()
        match = next((t for t in self.osu_entries if t[0] == disp), None)
        if not match:
            return

        _, osu_path, meta = match

        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{meta.artist}-{meta.title}-{meta.version}")[:64]
        out_dir = self.input_path.parent
        suggested = out_dir / f"{safe}.sspm"
        self.var_output.set(str(suggested))

    def convert(self):
        if not self.input_path:
            messagebox.showerror("Error", "Pick an input .osz/.osu first.")
            return

        outp = Path(self.var_output.get().strip())
        if not str(outp):
            messagebox.showerror("Error", "Choose an output .sspm path.")
            return

        disp = self.var_selected_osu.get()
        match = next((t for t in self.osu_entries if t[0] == disp), None)
        if not match:
            messagebox.showerror("Error", "Select a chart/difficulty to export.")
            return

        _, osu_path, meta = match

        diff_group_key = self.var_set_diff_group.get().strip().lower()
        if diff_group_key not in conv.DIFF_TO_GROUP:
            messagebox.showerror("Error", f"Invalid SSPM difficulty group: {diff_group_key}")
            return

        try:
            # parse
            meta, hitobjects = conv.parse_osu(osu_path)
            if meta.mode != 0:
                raise RuntimeError(f"Unsupported Mode:{meta.mode}. Needs Mode:0.")

            if not self.workdir:
                raise RuntimeError("Internal error: missing workdir.")
            if not meta.audio_filename:
                raise RuntimeError("No AudioFilename found in the .osu.")

            audio_path = self.workdir / meta.audio_filename
            if not audio_path.exists():
                raise RuntimeError(f"Audio file not found: {audio_path}")
            
            slider_end_events = None
            if self.var_include_slider_ends.get():
                slider_end_events = conv.parse_osu_slider_end_events(osu_path)


            audio_bytes = audio_path.read_bytes()

            notes = conv.hitobjects_to_notes(
                hitobjects,
                include_sliders=self.var_include_sliders.get(),
                include_spinners=self.var_include_spinners.get(),
                include_slider_ends=self.var_include_slider_ends.get(),
                slider_end_events=slider_end_events,
            )
            if not notes:
                raise RuntimeError("No notes generated.")

            # Cover: custom path overrides beatmap background
            cover_bytes = b""
            cover_path_str = self.var_cover.get().strip()
            if cover_path_str:
                cover_path = Path(cover_path_str)
                if cover_path.exists():
                    cover_bytes = cover_path.read_bytes()
            elif meta.background_filename:
                bg_path = self.workdir / meta.background_filename
                if bg_path.exists():
                    cover_bytes = bg_path.read_bytes()

            map_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{meta.artist}-{meta.title}-{meta.version}")[:64]
            map_name = f"{meta.title} [{meta.version}]"
            song_name = f"{meta.artist} - {meta.title}"
            mappers = [meta.creator]

            # ensure parent directory exists
            outp.parent.mkdir(parents=True, exist_ok=True)

            conv.write_sspm_v2(
                out_path=outp,
                map_id=map_id,
                map_name=map_name,
                song_name=song_name,
                mappers=mappers,
                difficulty_group=conv.DIFF_TO_GROUP[diff_group_key],
                audio_bytes=audio_bytes,
                cover_bytes=cover_bytes,
                notes=notes,
            )

            self._log("")
            self._log(f"✅ Wrote: {outp}")
            self._log(f"✅ Chart: {meta.version}")
            self._log(f"✅ Notes: {len(notes)}")
            self._log(f"✅ Audio bytes: {len(audio_bytes)}")
            if cover_bytes:
                self._log(f"✅ Cover bytes: {len(cover_bytes)}")

            messagebox.showinfo("Done", f"Exported:\n{outp}\n\nChart: {meta.version}\nNotes: {len(notes)}")

        except Exception as e:
            messagebox.showerror("Convert failed", str(e))
            self._log(f"❌ Error: {e}")


if __name__ == "__main__":
    App().mainloop()
