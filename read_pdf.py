import pymupdf, sys
doc = pymupdf.open('CN_Project_Report_DFL_Imperfect_Channels (1).pdf')
text = '\n'.join([p.get_text() for p in doc])

# Get large chunk around the methodology architecture section
idx = text.find('Methodology Architecture')
print(text[idx:idx+6000].encode('ascii','replace').decode())
