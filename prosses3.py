#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_fonon_ocr.py
يستخدم OCR (Tesseract) لقراءة فنون المحاماة واستخراج 39 صياغة

المتطلبات:
  pip install pymupdf pytesseract pillow
  + تثبيت Tesseract مع حزمة العربية:
    https://github.com/UB-Mannheim/tesseract/wiki
    أثناء التثبيت: اختر "Additional script data" → Arabic
"""

import fitz
import pytesseract
from PIL import Image
import re
import json
import io
import os

# ══════════════════════════════════════════════════
# ⚙️  إعدادات — عدّلها حسب جهازك
# ══════════════════════════════════════════════════
PDF_PATH  = r"C:\Users\NEW\OneDrive\Desktop\Project Assistant\فنون_المحاماة.pdf"
JSON_OUT  = r"C:\Users\NEW\OneDrive\Desktop\Project Assistant\fonon_data.json"
DEBUG_TXT = r"C:\Users\NEW\OneDrive\Desktop\Project Assistant\fonon_ocr_debug.txt"

# مسار Tesseract (عدّله إذا ثبّتته في مكان مختلف)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# دقة تحويل الصفحة إلى صورة (كلما كانت أعلى كان OCR أدق)
DPI = 250


# ══════════════════════════════════════════════════
# عناوين الأقسام الـ 39 (من الفهرست)
# ══════════════════════════════════════════════════
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


def classify(title: str) -> str:
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


# ══════════════════════════════════════════════════
# 1. تحويل PDF → نص بالـ OCR
# ══════════════════════════════════════════════════
def ocr_pdf(pdf_path: str, dpi: int = DPI) -> list[str]:
    """
    يُرجع قائمة: [نص_صفحة_1, نص_صفحة_2, ...]
    """
    doc = fitz.open(pdf_path)
    total = len(doc)
    pages_text = []

    for i, page in enumerate(doc):
        print(f'  OCR صفحة {i+1}/{total}...', end='\r')

        # تحويل الصفحة إلى صورة
        mat = fitz.Matrix(dpi / 72, dpi / 72)   # scale factor
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img_bytes = pix.tobytes('png')
        img = Image.open(io.BytesIO(img_bytes))

        # OCR باللغة العربية
        text = pytesseract.image_to_string(
            img,
            lang='ara',
            config='--oem 3 --psm 6'   # PSM 6 = فقرات موحدة
        )
        pages_text.append(text)

    doc.close()
    print(f'\n  تم: {total} صفحة')
    return pages_text


# ══════════════════════════════════════════════════
# 2. تنظيف النص
# ══════════════════════════════════════════════════
def clean_lines(pages: list[str]) -> list[str]:
    lines = []
    for page_text in pages:
        for ln in page_text.split('\n'):
            ln = ln.strip()
            if not ln:
                continue
            # حذف أرقام الصفحات
            if re.fullmatch(r'\d{1,3}', ln):
                continue
            # حذف رأس الصفحة الثابت
            if 'فنون المحاماة' in ln and ('اعداد' in ln or 'الطائي' in ln):
                continue
            lines.append(ln)
    return lines


# ══════════════════════════════════════════════════
# 3. مطابقة العناوين (مع مرونة لأخطاء OCR)
# ══════════════════════════════════════════════════
def normalize(text: str) -> str:
    """يُزيل الحركات والتشكيل والمسافات الزائدة"""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # حذف الحركات
    text = re.sub(r'[أإآ]', 'ا', text)                  # توحيد الألف
    text = re.sub(r'ة', 'ه', text)                       # توحيد التاء المربوطة
    text = re.sub(r'ى', 'ي', text)                       # توحيد الياء
    text = re.sub(r'[\s\.\،:()]+', ' ', text)            # توحيد المسافات
    return text.strip()


_NORM_TITLES = {normalize(t): t for t in TITLES}


def match_title(line: str) -> str | None:
    n = normalize(line)
    # مطابقة تامة
    if n in _NORM_TITLES:
        return _NORM_TITLES[n]
    # مطابقة جزئية: إذا كان السطر قصيراً وأحد العناوين يحتويه
    for nt, original in _NORM_TITLES.items():
        if len(n) >= 8 and (n in nt or nt in n) and abs(len(n) - len(nt)) <= 6:
            return original
    return None


# ══════════════════════════════════════════════════
# 4. أنماط بداية ونهاية الصياغة
# ══════════════════════════════════════════════════
_RE_START  = re.compile(r'^(السيد\s+قاضي\s+محكمة|م\s*/\s*(حجة|نسخة|تقرير))', re.U)
_RE_END    = re.compile(r'^مع\s+(الشكر|وافر)\s+(والتقدير|الشكر)', re.U)
_RE_ATTACH = re.compile(r'^المرفقات[\s.:]*$', re.U)


# ══════════════════════════════════════════════════
# 5. تحليل الأقسام
# ══════════════════════════════════════════════════
def parse(lines: list[str]) -> list[dict]:
    results = []
    tid = 0
    cur_title = None
    formal = []
    attachments = []
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
                    'category': classify(cur_title),
                }
            })

    for ln in lines:
        # ── هل هذا السطر عنوان قسم جديد؟ ──────────────────────
        matched = match_title(ln)
        if matched:
            flush()
            tid += 1
            cur_title = matched
            formal = []
            attachments = []
            state = 'before'
            continue

        if cur_title is None:
            continue

        # ── آلة الحالات ──────────────────────────────────────────
        if state == 'before':
            if _RE_START.match(ln):
                state = 'formal'
                formal.append(ln)
            # الكلام الشرحي العامي قبل الصياغة → يُتجاهل

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


# ══════════════════════════════════════════════════
# 6. تقرير الجودة
# ══════════════════════════════════════════════════
def report(results: list):
    print(f'\n{"═"*55}')
    print(f'  صياغات مستخرجة : {len(results)} من أصل {len(TITLES)}')
    print(f'{"═"*55}')

    extracted = {r['title'] for r in results}
    missing = [t for t in TITLES if t not in extracted]
    if missing:
        print(f'\n  ⚠  مفقودة ({len(missing)}):')
        for m in missing:
            print(f'     - {m}')

    print('\n── عيّنة أول 3 صياغات ──')
    for r in results[:3]:
        print(f"\n  [{r['template_id']:2d}] {r['title']}")
        print(f"       نص    : {r['text'][:100]}...")
        print(f"       مرفقات: {r['attachments']}")
        print(f"       تصنيف : {r['metadata']['category']}")
    print()


# ══════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 55)
    print('  فنون المحاماة — استخراج بالـ OCR')
    print('=' * 55)

    # ── تحقق من Tesseract ─────────────────────────────────────
    try:
        ver = pytesseract.get_tesseract_version()
        print(f'\n✓ Tesseract: {ver}')
    except Exception:
        print('\n✗ Tesseract غير موجود أو المسار خاطئ!')
        print('  1. حمّل من: https://github.com/UB-Mannheim/tesseract/wiki')
        print('  2. أثناء التثبيت اختر: Additional scripts → Arabic')
        print('  3. عدّل pytesseract.tesseract_cmd في أعلى الكود')
        exit(1)

    # ── تحقق من حزمة العربية ──────────────────────────────────
    langs = pytesseract.get_languages()
    if 'ara' not in langs:
        print('\n✗ حزمة العربية غير مثبتة في Tesseract!')
        print('  حمّل ara.traineddata من:')
        print('  https://github.com/tesseract-ocr/tessdata/blob/main/ara.traineddata')
        print('  وضعه في مجلد tessdata داخل مجلد Tesseract')
        exit(1)
    print('✓ حزمة العربية متوفرة')

    # ── OCR ──────────────────────────────────────────────────
    print(f'\n[1] تحويل PDF إلى نص بالـ OCR (DPI={DPI}) ...')
    print('    (قد يستغرق بضع دقائق)')
    pages = ocr_pdf(PDF_PATH)

    # ── تنظيف ──────────────────────────────────────────────
    print('\n[2] تنظيف النص...')
    lines = clean_lines(pages)
    print(f'    ← {len(lines)} سطر')

    # حفظ نص OCR للفحص
    with open(DEBUG_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'    ← نص OCR الخام: {DEBUG_TXT}')

    # ── تحليل ──────────────────────────────────────────────
    print('\n[3] تحليل الصياغات...')
    results = parse(lines)

    # ── حفظ JSON ────────────────────────────────────────────
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'\n✓ تم: {len(results)} صياغة → {JSON_OUT}')
    report(results)

    # ── نصيحة إذا كانت النتائج ناقصة ──────────────────────
    if len(results) < len(TITLES):
        missing_count = len(TITLES) - len(results)
        print(f'💡 لتحسين النتائج ({missing_count} مفقودة):')
        print('   - جرّب DPI=300 بدلاً من 250')
        print('   - افتح fonon_ocr_debug.txt وابحث عن بعض العناوين يدوياً')
        print('   - إذا وجدت العنوان بصيغة مختلفة، أرسله لأضيفه للقاموس')