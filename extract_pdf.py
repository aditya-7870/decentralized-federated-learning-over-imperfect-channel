"""Simple PDF text extractor using raw PDF parsing."""
import re
import sys

with open('CN_Project_Report_DFL_Imperfect_Channels (1).pdf', 'rb') as f:
    data = f.read()

# Extract text streams from PDF (simple raw extraction)
text_blocks = re.findall(b'BT(.+?)ET', data, re.DOTALL)
texts = []
for block in text_blocks:
    # Extract string literals - handle both () and <> strings
    strings = re.findall(rb'\(([^)\\]*(?:\\.[^)\\]*)*)\)', block)
    for s in strings:
        try:
            # Decode escape sequences
            t = s.replace(b'\\n', b' ').replace(b'\\r', b' ')
            t = t.replace(b'\\t', b' ').replace(b'\\(', b'(').replace(b'\\)', b')')
            t = t.decode('latin-1').strip()
            if len(t) > 2:
                texts.append(t)
        except Exception:
            pass

output = '\n'.join(texts)
sys.stdout.buffer.write(output.encode('utf-8', errors='replace'))
sys.stdout.buffer.write(b'\n')
