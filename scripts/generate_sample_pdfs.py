from pathlib import Path

try:
    import fitz
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf", "-q"])
    import fitz

sample = Path(__file__).resolve().parent.parent / "sample-data"

for txt in sample.glob("*.txt"):
    doc = fitz.open()
    text = txt.read_text(encoding="utf-8")
    chunks: list[str] = []
    buf: list[str] = []
    for para in text.split("\n\n"):
        buf.append(para)
        if sum(len(x) for x in buf) > 1400:
            chunks.append("\n\n".join(buf))
            buf = []
    if buf:
        chunks.append("\n\n".join(buf))
    if not chunks:
        chunks = [text]

    for i, chunk in enumerate(chunks, 1):
        page = doc.new_page()
        page.insert_text((50, 50), f"{txt.stem} — page {i}", fontsize=11)
        rect = fitz.Rect(50, 80, 560, 740)
        page.insert_textbox(rect, chunk, fontsize=10, fontname="helv")

    out = sample / f"{txt.stem}.pdf"
    doc.save(out)
    doc.close()
    print(f"wrote {out.name}")

print("done")
