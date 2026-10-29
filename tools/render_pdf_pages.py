"""QA rasterizer using bundled pypdfium2; no source documents are edited."""
import argparse
from pathlib import Path
import pypdfium2 as pdfium
from pypdf import PdfReader
p=argparse.ArgumentParser();p.add_argument('pdf');p.add_argument('output');args=p.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
pdf=pdfium.PdfDocument(args.pdf)
for i in range(len(pdf)):pdf[i].render(scale=1.6).to_pil().save(out/f'page-{i+1}.png')
reader=PdfReader(args.pdf)
for i,page in enumerate(reader.pages):
 text=page.extract_text() or ''
 print(f'PAGE {i+1}: {len(text)} chars | {text[:100].replace(chr(10), " ")}')
(out/'extracted_text.txt').write_text('\n'.join(f'PAGE {i+1}\n'+(pg.extract_text() or '') for i,pg in enumerate(reader.pages)),encoding='utf-8')
