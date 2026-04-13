#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/extract_fonon.py
فنون المحاماة والترافع — PDF → JSON (عبر OCR)

يقرأ PDF من data/raw/ ويُنتج JSON في data/processed/

المتطلبات:
    pip install pymupdf pytesseract pillow
    Tesseract مع حزمة العربية (ara.traineddata)

الاستخدام:
    python scripts/extract_fonon.py
"""

import re
import json
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROCESSED

INPUT_FILE  = DATA_RAW       / "legal_templates.pdf"
OUTPUT_FILE = DATA_PROCESSED / "fonon.json"
DPI = 250

# ── عناوين الأقسام ──────────────────────────────────────────────────────
TITLES = [
    "طلب نسخة مصدقة",
    "دعوى تصديق عقد الزواج الخارجي",
    "دعوى تصديق الطلاق الخارجي",
    "حجة أذن بالزواج من زوجة ثانية",
    "اساسيات وأنواع دعاوى الطلاق",
    "دعوى الطلاق الرجعي . الطلاق العادي",
    "دعوى الطلاق الخلعي ( المخالعة )",
    "دعوى التفريق للضرر",
    "دعوى تثبيت رجعة الزوج لزوجته خلال فترة العدة",
    "حق المطلقة بالسكنى",
    "دعوى التعويض عن الطلاق التعسفي + نفقة العدة",
    "دعوى المطالبة بالمهر المؤجل ( المؤخر )",
    "دعوى النفقة للزوجة أو للزوجة والاطفال",
    "دعوى زيادة النفقة للزوجة والاطفال",
    "دعوى المطاوعة",
    "دعوى نشوز الزوجة",
    "دعوى اثاث الزوجية",
    "دعوى تأييد الحضانة",
    "دعوى تسليم الاطفال",
    "دعوى مشاهدة الاطفال",
    "دعوى أسقاط الحضانة",
    "حجة الوصاية",
    "حالات تعيين ولي مؤقت على القاصر",
    "حجة الوصاية المؤقتة",
    "حجة اذن بالسفر",
    "حجة الحجر والقيمومة على المفقود",
    "دعوى وفاة المفقود",
    "حجة الولادة",
    "دعوى اثبات النسب",
    "حجة التخارج",
    "القسام الشرعي",
    "نقل الراتب التقاعدي من متقاعد متوفي الى ورثته",
    "صورة قيد 57 مترجمة لخارج العراق",
    "اصدار جواز سفر بدل ضائع او تالف",
    "الاعتراض على الحكم الغيابي",
    "تقرير خبرة قضائية / دعوى نفقة للزوجة والاطفال",
    "تقرير خبراء قضائيين / دعوى اثاث زوجية",
    "تقرير خبرة قضائية / دعوى التعويض عن الطلاق التعسفي ونفقة العدة",
    "كيفية اقامة الدعوى الشرعية",
]


def classify_category(title: str) -> str:
    t = title
    if any(x in t for x in ['زواج', 'عقد الزواج']): return 'زواج'
    if any(x in t for x in ['طلاق', 'مخالعة', 'رجعة', 'تفريق']): return 'طلاق'
    if any(x in t for x in ['نفقة', 'مهر', 'سكنى', 'تعويض', 'مطاوعة', 'نشوز']): return 'نفقة_ومستحقات'
    if any(x in t for x in ['حضانة', 'أطفال', 'اطفال', 'مشاهدة', 'تسليم']): return 'حضانة'
    if any(x in t for x in ['اثاث', 'زوجية']): return 'اثاث_زوجية'
    if any(x in t for x in ['وصاية', 'قاصر', 'ولي']): return 'وصاية'
    if any(x in t for x in ['حجر', 'مفقود', 'وفاة']): return 'احوال_خاصة'
    if any(x in t for x in ['نسب', 'ولادة', 'تخارج', 'قسام']): return 'احوال_مدنية'
    if any(x in t for x in ['جواز', 'سفر', 'قيد', 'تقاعد']): return 'وثائق_رسمية'
    if 'تقرير' in t: return 'تقارير_قضائية'
    return 'اجراءات'


def normalize(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'[\s\.،:()]+', ' ', text)
    return text.strip()


_NORM_TITLES = {normalize(t): t for t in TITLES}


def match_title(line: str) -> str | None:
    n = normalize(line)
    if n in _NORM_TITLES:
        return _NORM_TITLES[n]
    for nt, original in _NORM_TITLES.items():
        if len(n) >= 8 and (n in nt or nt in n) and abs(len(n) - len(nt)) <= 6:
            return original
    return None


def ocr_pdf(pdf_path: Path, dpi: int = DPI) -> list:
    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    pages_text = []
    for i, page in enumerate(doc):
        print(f'  OCR صفحة {i+1}/{total}...', end='\r')
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = Image.open(io.BytesIO(pix.tobytes('png')))
        text = pytesseract.image_to_string(img, lang='ara', config='--oem 3 --psm 6')
        pages_text.append(text)
    doc.close()
    print(f'\n  تم: {total} صفحة')
    return pages_text


def clean_lines(pages: list) -> list:
    lines = []
    for page_text in pages:
        for ln in page_text.split('\n'):
            ln = ln.strip()
            if not ln or re.fullmatch(r'\d{1,3}', ln):
                continue
            if 'فنون المحاماة' in ln and ('اعداد' in ln or 'الطائي' in ln):
                continue
            lines.append(ln)
    return lines


_RE_START  = re.compile(r'^(السيد\s+قاضي\s+محكمة|م\s*/\s*(حجة|نسخة|تقرير))', re.U)
_RE_END    = re.compile(r'^مع\s+(الشكر|وافر)\s+(والتقدير|الشكر)', re.U)
_RE_ATTACH = re.compile(r'^المرفقات[\s.:]*$', re.U)


def parse(lines: list) -> list:
    results = []
    tid = 0
    cur_title = None
    formal, attachments = [], []
    state = 'outside'

    def flush():
        nonlocal cur_title
        if not cur_title:
            return
        text = ' '.join(ln for ln in formal if ln).strip()
        atts = [a.strip() for a in attachments if a.strip()]
        if text or atts:
            results.append({
                'template_id': tid,
                'title': cur_title,
                'text': text,
                'attachments': atts,
                'metadata': {
                    'type': 'legal_template',
                    'source': 'فنون المحاماة والترافع أمام المحاكم الشرعية',
                    'category': classify_category(cur_title),
                }
            })

    for ln in lines:
        matched = match_title(ln)
        if matched:
            flush()
            tid += 1
            cur_title = matched
            formal, attachments = [], []
            state = 'before'
            continue
        if cur_title is None:
            continue
        if state == 'before':
            if _RE_START.match(ln):
                state = 'formal'
                formal.append(ln)
        elif state == 'formal':
            if _RE_ATTACH.match(ln):
                state = 'attach'
            elif _RE_END.match(ln):
                formal.append(ln)
                state = 'after_end'
            else:
                formal.append(ln)
        elif state == 'after_end':
            if _RE_ATTACH.match(ln):
                state = 'attach'
        elif state == 'attach':
            attachments.append(ln)

    flush()
    return results


def main():
    print('=' * 50)
    print('  فنون المحاماة — استخراج بالـ OCR')
    print('=' * 50)

    try:
        import pytesseract
        ver = pytesseract.get_tesseract_version()
        print(f'\n✓ Tesseract: {ver}')
    except Exception:
        print('\n✗ Tesseract غير موجود!')
        print('  حمّله من: https://github.com/UB-Mannheim/tesseract/wiki')
        sys.exit(1)

    print(f'\n[1] OCR على {INPUT_FILE} ...')
    pages = ocr_pdf(INPUT_FILE)

    print('\n[2] تنظيف النص...')
    lines = clean_lines(pages)
    print(f'    ← {len(lines)} سطر')

    print('\n[3] تحليل الصياغات...')
    results = parse(lines)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'\n✓ {len(results)} صياغة → {OUTPUT_FILE}')
    missing = [t for t in TITLES if t not in {r['title'] for r in results}]
    if missing:
        print(f'⚠ مفقودة ({len(missing)}): {missing[:3]}...')


if __name__ == '__main__':
    main()
