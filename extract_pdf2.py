"""Extract readable text from the PDF using zlib decompression and text stream parsing."""
import re, zlib, sys

with open('CN_Project_Report_DFL_Imperfect_Channels (1).pdf', 'rb') as f:
    data = f.read()

streams = re.findall(b'stream\r?\n(.*?)\r?\nendstream', data, re.DOTALL)

all_text = []
for stream in streams:
    try:
        decompressed = zlib.decompress(stream)
        text = decompressed.decode('latin-1', errors='replace')
        # Find text inside Tj / TJ operators
        # Extract strings in () that look like real words
        chunks = re.findall(r'\(([^)]{2,})\)', text)
        cleaned = []
        for chunk in chunks:
            # Remove PDF kerning numbers like -4( or )18(
            clean = re.sub(r'\)-?\d+\(', '', chunk)
            clean = re.sub(r'[^\x20-\x7E\n]', '', clean)
            clean = clean.strip()
            if len(clean) > 3 and re.search(r'[a-zA-Z]{3,}', clean):
                cleaned.append(clean)
        if cleaned:
            all_text.extend(cleaned)
    except Exception:
        pass

# Join and print
output = '\n'.join(all_text)
sys.stdout.buffer.write(output.encode('utf-8', errors='replace'))
