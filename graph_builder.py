#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_builder.py
بناء شبكة العلاقات من Qdrant — مفاتيح طبيعية

الإصلاحات المبنية على فحص البيانات الحقيقية:

  [1] link_articles → same-law فقط بشكل افتراضي
      cross-law فقط إذا ذُكر اسم القانون صراحةً في النص
      (النسخة القديمة كانت تربط بالقانونين دائماً)

  [2] link_by_proximity → حُذف بالكامل
      السبب: bab في laws = "الباب الثامن" لكل 308 مادة (وهمي)
             bab في osoul = "باب تمهيدي" لكل 495 مادة (وهمي)
             fasl في laws = فارغ لكل 308 مادة
             fasl في osoul: مبعثر (max_gap=250 في بعض الفصول)
      → البديل: الجوار الرقمي المباشر article_num ± 1

  [3] link_fonon_to_articles → حُذفت "دعوى" من TOPIC_OSOUL
      السبب: تظهر في 26/28 قالب → لا قيمة دلالية
      وحُذف الشرط "دعوى in title" العام

تشغيل مرة واحدة:
  python graph_builder.py
"""

import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

GRAPH_FILE = Path(__file__).parent / "graph.json"
COL_LAWS   = "legal_laws"
COL_OSOUL  = "legal_osoul"
COL_FONON  = "legal_fonon"


# ══════════════════════════════════════════════════════════════
# أدوات مساعدة
# ══════════════════════════════════════════════════════════════

def fetch_all(client: QdrantClient, collection: str) -> list:
    all_points, offset = [], None
    while True:
        results, next_offset = client.scroll(
            collection_name=collection, offset=offset,
            limit=100, with_payload=True, with_vectors=False,
        )
        all_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset
    print(f"  <- {collection}: {len(all_points)} نقطة")
    return all_points


def make_key(payload: dict) -> str:
    source = payload.get("source", "")
    if source == "fonon":
        return f"fonon_{payload.get('template_id', '')}"
    return f"{source}_{payload.get('article_num', '')}"


def extract_refs(text: str) -> list:
    """
    يستخرج أرقام المواد المذكورة في النص.
    يتطلب سياقاً صريحاً لتجنب false positives.
    """
    nums = set()
    for p in [r'المادة\s+(\d+)', r'المواد\s+(\d+)']:
        for m in re.finditer(p, str(text)):
            n = int(m.group(1))
            if 1 <= n <= 600:
                nums.add(n)
    # م/137 أو م.137 (يتطلب / أو . لتجنب الأرقام العادية)
    for m in re.finditer(r'\bم\s*[./]\s*(\d+)\b', str(text)):
        n = int(m.group(1))
        if 1 <= n <= 600:
            nums.add(n)
    return list(nums)


def add_edge(graph: dict, src: str, dst: str):
    """إضافة رابط مع تجنب التكرار (أحادي الاتجاه — الاتجاهية تُحدد في مكان الاستدعاء)"""
    if src and dst and src != dst:
        if dst not in graph.get(src, []):
            graph.setdefault(src, []).append(dst)


# ══════════════════════════════════════════════════════════════
# خرائط المواضيع
# ══════════════════════════════════════════════════════════════

TOPIC_LAWS = {
    "زواج":   [5, 6, 7, 8, 9, 10, 40, 41, 42],
    "مهر":    [52, 53, 54, 55, 56, 57, 58],
    "طلاق":   [85, 86, 87, 88, 89, 90, 91],
    "خلع":    [95, 96, 97, 98, 99, 100],
    "تفريق":  [105, 106, 107, 108, 109, 110],
    "نسب":    [120, 121, 122, 123, 124, 125],
    "حضانة":  [137, 138, 139, 140, 141, 146, 154],
    "رؤية":   [154, 155, 156],
    "نفقة":   [72, 73, 74, 75, 76, 77, 78],
    "وصاية":  [160, 161, 162, 163],
    "ولاية":  [163, 164, 165, 166, 167, 168, 169, 170],
    "وصية":   [200, 201, 202, 203, 204, 205],
    "ميراث":  [265, 270, 275, 280, 285, 290, 295, 300],
}

# "دعوى" محذوفة: تظهر في 26/28 قالب → تولّد روابط مزيفة لكل شيء
TOPIC_OSOUL = {
    "تبليغ":   [10, 11, 12, 13, 14, 15],
    "إثبات":   [65, 66, 67, 68, 69, 70],
    "استئناف": [150, 151, 152, 153],
    "تنفيذ":   [280, 281, 282, 283],
    "اختصاص":  [30, 31, 32, 33, 34],
    "طعن":     [170, 171, 172, 173],
    "خبرة":    [72, 73, 74, 75],
    "حجز":     [290, 291, 292, 293],
}


# ══════════════════════════════════════════════════════════════
# بناء الشبكة
# ══════════════════════════════════════════════════════════════

def build_graph(client: QdrantClient) -> dict:
    graph = {}

    # ── [1] جلب البيانات ──────────────────────────────────────
    print("\n[1] جلب البيانات من Qdrant...")
    laws_points  = fetch_all(client, COL_LAWS)
    osoul_points = fetch_all(client, COL_OSOUL)
    fonon_points = fetch_all(client, COL_FONON)

    for p in laws_points + osoul_points + fonon_points:
        key = make_key(p.payload)
        if key:
            graph.setdefault(key, [])

    laws_by_num  = {int(p.payload["article_num"]): make_key(p.payload)
                    for p in laws_points  if p.payload.get("article_num") is not None}
    osoul_by_num = {int(p.payload["article_num"]): make_key(p.payload)
                    for p in osoul_points if p.payload.get("article_num") is not None}

    print(f"  laws_by_num : {len(laws_by_num)} مادة")
    print(f"  osoul_by_num: {len(osoul_by_num)} مادة")

    # ── [2] روابط مادة ← مادة (same-law بشكل افتراضي) ─────────
    #
    #  cross-law فقط إذا ذُكر اسم القانون الآخر صراحةً في النص.
    #  النسخة القديمة كانت تربط بالقانونين دائماً مما يولّد
    #  روابط مزيفة: laws_54 → osoul_447 لأن النص ذكر الرقم 447
    #  بدون أي علاقة بأصول المحاكمات.
    # ───────────────────────────────────────────────────────────
    print("\n[2] بناء روابط مادة <- مادة (same-law)...")

    OSOUL_NAMES = ["أصول المحاكمات", "قانون الأصول", "الإجراءات المدنية"]
    LAWS_NAMES  = ["الأحوال الشخصية", "قانون الأحوال"]

    def link_same_law(points, self_idx, other_idx, other_names):
        for p in points:
            src_key = make_key(p.payload)
            text    = p.payload.get("original_text", "")
            cur_num = p.payload.get("article_num")
            cross   = any(name in text for name in other_names)
            for ref in extract_refs(text):
                if cur_num and ref == int(cur_num):
                    continue
                if ref in self_idx:
                    # self_idx له الأولوية دائماً
                    add_edge(graph, src_key, self_idx[ref])
                    add_edge(graph, self_idx[ref], src_key)
                elif cross and ref in other_idx:
                    # cross-law فقط إذا الرقم غائب من self وذُكر القانون الآخر صراحةً
                    add_edge(graph, src_key, other_idx[ref])
                    add_edge(graph, other_idx[ref], src_key)

    link_same_law(laws_points,  laws_by_num,  osoul_by_num, OSOUL_NAMES)
    link_same_law(osoul_points, osoul_by_num, laws_by_num,  LAWS_NAMES)

    # ── [3] روابط صياغة ← مواد (bidirectional) ────────────────
    print("\n[3] بناء روابط الصياغات بالمواد...")

    for p in fonon_points:
        fkey = make_key(p.payload)
        # فحص النص الكامل — ليس العنوان فقط
        full_text = " ".join(filter(None, [
            p.payload.get("title", ""),
            p.payload.get("category", ""),
            p.payload.get("formal_text", ""),
            p.payload.get("intro_notes", ""),
        ]))

        for topic, nums in TOPIC_LAWS.items():
            if topic in full_text:
                for num in nums:
                    if num in laws_by_num:
                        add_edge(graph, fkey, laws_by_num[num])
                        add_edge(graph, laws_by_num[num], fkey)

        for topic, nums in TOPIC_OSOUL.items():
            if topic in full_text:
                for num in nums:
                    if num in osoul_by_num:
                        add_edge(graph, fkey, osoul_by_num[num])
                        add_edge(graph, osoul_by_num[num], fkey)

    # ── [4] الجوار الرقمي المباشر (article_num ± 1) ───────────
    #
    #  سبب الاختيار:
    #    laws : bab = "الباب الثامن" لكل 308 مادة ← وهمي
    #           fasl = فارغ لكل 308 مادة
    #    osoul: bab = "باب تمهيدي" لكل 495 مادة ← وهمي
    #           fasl: مبعثر (max_gap=250 في بعض الفصول)
    #  → الجوار الرقمي يعكس البنية الطبيعية للقانون
    #    بدون الاعتماد على metadata معطوبة.
    # ───────────────────────────────────────────────────────────
    print("\n[4] بناء روابط الجوار الرقمي (article_num ± 1)...")

    for by_num in (laws_by_num, osoul_by_num):
        for num, key in by_num.items():
            for nb in (num - 1, num + 1):
                if nb in by_num:
                    add_edge(graph, key, by_num[nb])
                    add_edge(graph, by_num[nb], key)

    return graph


# ══════════════════════════════════════════════════════════════
# تقرير
# ══════════════════════════════════════════════════════════════

def report(graph: dict):
    total_nodes = len(graph)
    total_edges = sum(len(v) for v in graph.values())
    isolated    = sum(1 for v in graph.values() if not v)

    print(f"\n{'='*50}")
    print(f"  إجمالي العقد   : {total_nodes}")
    print(f"  إجمالي الروابط : {total_edges}")
    print(f"  عقد معزولة     : {isolated}")
    print(f"  متوسط الروابط  : {total_edges / max(total_nodes, 1):.1f} لكل عقدة")

    top = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    print(f"\n  أكثر المواد ارتباطاً:")
    for key, neighbors in top:
        print(f"    {key:20} <- {len(neighbors)} رابط")

    print(f"\n  عينة من الروابط:")
    shown = 0
    for key, neighbors in graph.items():
        if neighbors and shown < 4:
            print(f"    {key} -> {neighbors[:3]}")
            shown += 1
    print('=' * 50)


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + '=' * 50)
    print("  بناء شبكة العلاقات من Qdrant")
    print('=' * 50)

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

    graph = build_graph(client)

    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"\n  تم الحفظ في: {GRAPH_FILE}")
    report(graph)
    print("\n  جاهز\n")