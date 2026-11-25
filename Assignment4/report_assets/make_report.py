"""Convert report.md to submission/report.pdf (Markdown -> HTML -> PDF)."""
import re
import shutil
import subprocess
from pathlib import Path

import markdown
from weasyprint import HTML

ASSETS = Path(__file__).resolve().parent
TMP = ASSETS / 'tmp'
PDF = ASSETS.parent / 'submission' / 'report.pdf'
MAX_PAGES = 2

TMP.mkdir(exist_ok=True)
body = markdown.markdown((ASSETS / 'report.md').read_text(encoding='utf-8'), extensions=['tables'])
# Paragraphs that are only bold, only an image, or only italic are figure/table titles, figures and captions.
body = re.sub(r'<p>(<strong>[^<]*</strong>)</p>', r'<p class="title">\1</p>', body)
body = re.sub(r'<p>(<img [^>]*>)</p>', r'<p class="figure">\1</p>', body)
body = re.sub(r'<p>(<em>[^<]*</em>)</p>', r'<p class="caption">\1</p>', body)

# <base> makes report.css and the figures resolve against report_assets/, not tmp/.
html_path = TMP / 'report.html'
html_path.write_text(f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<base href="../">
<title>CMPUT 328 Assignment 4 Report</title>
<meta name="author" content="Laurie Gantuangco">
<link rel="stylesheet" href="report.css">
</head>
<body>
{body}
</body>
</html>
''', encoding='utf-8')

document = HTML(filename=str(html_path)).render()
document.write_pdf(PDF)

for old in TMP.glob('report_page-*.png'):
    old.unlink()
if shutil.which('pdftoppm'):
    subprocess.run(['pdftoppm', '-r', '80', '-png', str(PDF), str(TMP / 'report_page')], check=True)

pages = len(document.pages)
print(f'{PDF} ({pages} pages)')
if pages > MAX_PAGES:
    raise SystemExit(f'Report is {pages} pages; the limit is {MAX_PAGES}.')
