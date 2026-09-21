"""
Image Captioning AI  (Computer Vision + NLP)
--------------------------------------------
Two pre-trained encoder-decoder models, both COCO-style captioners:

1) ViT + GPT-2  (nlpconnect/vit-gpt2-image-captioning)
   Encoder: Vision Transformer turns the picture into visual features.
   Decoder: GPT-2 reads them via cross-attention and writes the caption.

2) BLIP         (Salesforce/blip-image-captioning-base)
   Vision-language model pre-trained on ~129M image-text pairs, with a
   bootstrapping step that cleans noisy web captions. Usually more accurate.

Use the dropdown to pick a model, or choose "Compare both" to see the two
captions side by side.

Setup (once):
    py -m pip install torch transformers pillow

Run the GUI:       python image_captioning.py
Run in terminal:   python image_captioning.py photo1.jpg photo2.png

Each model downloads about 1 GB the first time it is used (internet needed).
After that it works offline.
"""

import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageOps, ImageTk

MAX_NEW_TOKENS = 30   # maximum caption length in tokens
NUM_BEAMS = 4         # beam search width (higher = slower, often better)
NUM_CAPTIONS = 3      # best caption + 2 alternatives


def tidy(text):
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else text + "."


def open_image(path):
    img = Image.open(path)
    return ImageOps.exif_transpose(img)  # respect phone-camera rotation


# ----------------------------------------------------------------- Models
class ViTGPT2Captioner:
    """Pre-trained ViT image encoder + GPT-2 text decoder."""
    MODEL = "nlpconnect/vit-gpt2-image-captioning"

    def load(self):
        # Imported here so the window can open before the heavy libraries load.
        import torch
        from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL).to(self.device)
        self.model.eval()
        self.processor = ViTImageProcessor.from_pretrained(self.MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)

    def caption(self, image):
        """Return a list of captions for a PIL image, best one first."""
        image = image.convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(
                pixel_values, max_length=16, num_beams=NUM_BEAMS,
                num_return_sequences=NUM_CAPTIONS)
        return [tidy(t) for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]


class BlipCaptioner:
    """BLIP vision-language model (ViT encoder + BERT-style text decoder)."""
    MODEL = "Salesforce/blip-image-captioning-base"

    def load(self):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(self.MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(self.MODEL).to(self.device)
        self.model.eval()

    def caption(self, image):
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            ids = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS,
                num_return_sequences=NUM_CAPTIONS)
        return [tidy(t) for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


def make_captioners():
    return {"ViT + GPT-2": ViTGPT2Captioner(), "BLIP": BlipCaptioner()}


COMPARE = "Compare both"


# -------------------------------------------------------------------- GUI
class CaptionApp:
    BG = "#1e1e2e"
    PANEL = "#313244"
    TEXT = "#cdd6f4"
    MUTED = "#a6adc8"
    ACCENT = "#89b4fa"
    GOOD = "#a6e3a1"
    BAD = "#f38ba8"
    PREVIEW_W, PREVIEW_H = 520, 380

    def __init__(self, root, captioners):
        self.root = root
        self.captioners = captioners            # {display name: captioner}
        self.names = list(captioners)
        self.loaded = set()
        self.events = queue.Queue()
        self.photo = None          # keep a reference so Tk doesn't discard the image
        self.busy = False
        self.image = None          # last image, so changing model can re-run it

        root.title("Image Captioning AI")
        root.configure(bg=self.BG)
        root.resizable(False, False)
        self.choice = tk.StringVar(value=self.names[0])
        self._build()

        # Preload the default model in the background so the first click is fast
        self.busy = True
        threading.Thread(target=self._worker, args=(None, [self.names[0]]), daemon=True).start()
        root.after(100, self._poll)

    def _build(self):
        tk.Label(self.root, text="IMAGE CAPTIONING AI", font=("Segoe UI", 20, "bold"),
                 bg=self.BG, fg=self.TEXT).pack(pady=(14, 0))
        tk.Label(self.root, text="Pre-trained vision encoder  →  transformer text decoder",
                 font=("Segoe UI", 10), bg=self.BG, fg=self.MUTED).pack()

        self.status = tk.Label(self.root, text="Starting...", font=("Segoe UI", 11),
                               bg=self.BG, fg=self.ACCENT, wraplength=520, justify="center")
        self.status.pack(pady=8)

        frame = tk.Frame(self.root, bg=self.PANEL, width=self.PREVIEW_W, height=self.PREVIEW_H)
        frame.pack(padx=20)
        frame.pack_propagate(False)
        self.preview = tk.Label(frame, text="No image selected", bg=self.PANEL,
                                fg=self.MUTED, font=("Segoe UI", 12))
        self.preview.pack(expand=True)

        row = tk.Frame(self.root, bg=self.BG)
        row.pack(pady=12)
        tk.Label(row, text="Model:", font=("Segoe UI", 11), bg=self.BG,
                 fg=self.TEXT).pack(side="left", padx=(0, 6))
        self.menu = tk.OptionMenu(row, self.choice, *(self.names + [COMPARE]))
        self.menu.config(font=("Segoe UI", 11), bg=self.PANEL, fg=self.TEXT,
                         activebackground=self.PANEL, activeforeground=self.TEXT,
                         relief="flat", highlightthickness=0, width=13)
        self.menu["menu"].config(bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 11))
        self.menu.pack(side="left", padx=(0, 14))

        self.button = tk.Button(row, text="Choose Image", state="disabled",
                                font=("Segoe UI", 12, "bold"), bg=self.ACCENT, fg="#11111b",
                                relief="flat", padx=18, pady=6, cursor="hand2",
                                command=self.choose_image)
        self.button.pack(side="left")

        self.caption_label = tk.Label(self.root, text="", font=("Segoe UI", 14, "bold"),
                                      bg=self.BG, fg=self.GOOD, wraplength=520, justify="left")
        self.caption_label.pack(padx=20)

        self.alt_label = tk.Label(self.root, text="", font=("Segoe UI", 10),
                                  bg=self.BG, fg=self.MUTED, wraplength=520, justify="left")
        self.alt_label.pack(padx=20, pady=(6, 18))

    # -- background worker: loads any missing model, then captions
    def _selected_names(self):
        return self.names if self.choice.get() == COMPARE else [self.choice.get()]

    def _worker(self, image, names):
        try:
            for name in names:
                if name not in self.loaded:
                    self.events.put(("status", f"Loading {name}... (first use downloads ~1 GB)"))
                    self.captioners[name].load()
                    self.loaded.add(name)
            if image is None:
                self.events.put(("ready", None))
                return
            self.events.put(("status", "Generating caption..."))
            results = {n: self.captioners[n].caption(image) for n in names}
            self.events.put(("caption", results))
        except ImportError:
            self.events.put(("error",
                             "Missing packages. Open Command Prompt and run:\n"
                             "py -m pip install torch transformers pillow"))
        except Exception as exc:  # e.g. no internet on first download
            self.events.put(("error", f"Something went wrong: {exc}"))

    # -- UI events (queue is polled on the Tk thread, which keeps Tk thread-safe)
    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle(kind, payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, kind, payload):
        if kind == "status":
            self.status.config(text=payload, fg=self.ACCENT)
        elif kind == "ready":
            self.busy = False
            self.status.config(text="Model ready. Choose an image to caption.", fg=self.GOOD)
            self.button.config(state="normal")
        elif kind == "caption":
            self.busy = False
            self.status.config(text="Done.", fg=self.GOOD)
            if len(payload) == 1:
                caps = next(iter(payload.values()))
                self.caption_label.config(text=caps[0])
                alts = caps[1:]
                self.alt_label.config(
                    text=("Other possible captions:\n" + "\n".join(f"• {a}" for a in alts))
                    if alts else "")
            else:
                self.caption_label.config(
                    text="\n\n".join(f"{n}:\n{caps[0]}" for n, caps in payload.items()))
                self.alt_label.config(text="")
            self.button.config(state="normal")
        elif kind == "error":
            self.busy = False
            self.status.config(text=payload, fg=self.BAD)
            self.button.config(state="normal")

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if path:
            self.load_image(path)

    def load_image(self, path):
        if self.busy:
            return
        try:
            image = open_image(path)
        except Exception as exc:
            self.status.config(text=f"Could not open that file: {exc}", fg=self.BAD)
            return

        self.image = image
        preview = image.copy()
        preview.thumbnail((self.PREVIEW_W - 10, self.PREVIEW_H - 10))
        self.photo = ImageTk.PhotoImage(preview.convert("RGB"))
        self.preview.config(image=self.photo, text="")

        self.busy = True
        self.button.config(state="disabled")
        self.caption_label.config(text="")
        self.alt_label.config(text="")
        self.status.config(text="Working...", fg=self.ACCENT)
        threading.Thread(target=self._worker, args=(image, self._selected_names()),
                         daemon=True).start()


# -------------------------------------------------------------------- Main
def run_cli(paths):
    captioners = make_captioners()
    for name, cap in captioners.items():
        print(f"Loading {name} (first use downloads ~1 GB)...")
        cap.load()
    for path in paths:
        image = open_image(path)
        print(f"\n{path}")
        for name, cap in captioners.items():
            caps = cap.caption(image)
            print(f"  [{name}] {caps[0]}")
            for alt in caps[1:]:
                print(f"      (alt) {alt}")


def main():
    if len(sys.argv) > 1:
        run_cli(sys.argv[1:])
        return
    root = tk.Tk()
    CaptionApp(root, make_captioners())
    root.mainloop()


if __name__ == "__main__":
    main()
