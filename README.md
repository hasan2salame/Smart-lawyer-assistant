# Syrian Legal AI Assistant 🏛️
### مساعد قانوني ذكي للمحامين السوريين

نظام استرجاع وتوليد معزّز (RAG) متخصص في قانون الأحوال الشخصية وأصول المحاكمات المدنية السورية.

---

## ما يفعله النظام

| المسار | المدخل | المخرج |
|--------|--------|--------|
| **صياغة** | "بدي صياغة دعوى طلاق خلعي" | قالب دعوى جاهز + تعبئة تلقائية |
| **مرفقات** | "شو الأوراق المطلوبة لدعوى النفقة؟" | قائمة المستندات المطلوبة |
| **قانوني** | "ما هي شروط الحضانة؟" | إجابة من المواد القانونية + مصادر |

---

## المعمارية

```
طلب المحامي
      │
      ▼
┌──────────────────┐
│ Intent Classifier │  Waterfall: Rules → Groq 8b
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│              Hybrid RAG Pipeline             │
│                                              │
│  Dense(laws)  ──┐                            │
│  Dense(osoul) ──┤                            │
│                 ├─► RRF ─► Graph ─► Rerank  │
│  BM25(laws)   ──┤                            │
│  BM25(osoul)  ──┘                            │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│    LLM Layer     │  handlers منفصلة لكل intent
│  Groq LLaMA 70b  │  Streaming حقيقي
└──────────────────┘
         │
         ▼
┌──────────────────┐
│  FastAPI + SQLite│  sessions دائمة + تعدد نوافذ
└──────────────────┘
```

---

## البيانات

| المصدر | المحتوى | العدد |
|--------|---------|-------|
| `legal_laws` | قانون الأحوال الشخصية السوري 1953 | 308 مادة |
| `legal_osoul` | قانون أصول المحاكمات المدنية 2016 | 495 مادة |
| `legal_fonon` | قوالب فنون المحاماة (PDF) | 28 قالب |

**شبكة العلاقات:** 831 عقدة — 2884 رابط

---

## التشغيل

### الطريقة 1 — Docker (موصى به)

```bash
# استنساخ المشروع
git clone https://github.com/hasan2salame/Smart-lawyer-assistant.git
cd Smart-lawyer-assistant

# إعداد المفاتيح
# عدّل .env بمفاتيحك الحقيقية

# تشغيل
docker compose up
```

افتح المتصفح على: **http://localhost:8000**

---

### الطريقة 2 — Python مباشرة

```bash
# إنشاء البيئة الافتراضية
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# تثبيت المكتبات
pip install -r requirements.txt

# تشغيل السيرفر
python main.py
# يفتح المتصفح تلقائياً على http://localhost:8000
```

---

## الإعداد الأول (مرة واحدة فقط)

لو عندك البيانات الخام (txt + pdf):

```bash
# استخراج البيانات
python scripts/extract_laws.py
python scripts/extract_osoul.py
python scripts/extract_fonon.py    # يحتاج Tesseract

# رفع لـ Qdrant
python scripts/upload_laws.py
python scripts/upload_osoul.py
python scripts/upload_fonon.py

# بناء شبكة العلاقات
python scripts/build_graph.py
```

---

## هيكل المشروع

```
syrian-legal-assistant/
│
├── config.py                    ← الإعدادات المركزية
├── main.py                      ← FastAPI entry point
├── pipeline.py                  ← المنسّق
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── raw/                     ← المصادر الأصلية
│   └── processed/               ← JSON + graph.json
│
├── retrieval/                   ← طبقة الاسترجاع
│   ├── __init__.py              ← HybridRetriever
│   ├── dense.py                 ← Cohere Embeddings
│   ├── bm25.py                  ← BM25 Sparse Search
│   ├── rrf.py                   ← Rank Fusion
│   ├── reranker.py              ← Cohere Rerank
│   └── graph/
│       ├── builder.py           ← بناء الشبكة
│       └── retriever.py         ← توسيع النتائج
│
├── nlp/
│   ├── classifier.py            ← Intent Classifier
│   └── llm.py                   ← LLM Layer
│
├── scripts/                     ← سكريبتات الإعداد
├── frontend/                    ← واجهة المستخدم
└── tests/
    ├── test_dense.py
    ├── test_bm25.py
    ├── test_rrf.py
    ├── test_graph.py
    ├── results/                 ← خرج الاختبارات
    └── comparisons/             ← تحليل مقارن
```

---

## API

| Method | Endpoint | الوصف |
|--------|----------|-------|
| `GET` | `/` | الواجهة |
| `GET` | `/health` | حالة السيرفر |
| `POST` | `/session/new` | جلسة جديدة |
| `GET` | `/sessions` | المحادثات السابقة |
| `DELETE` | `/session/{id}` | حذف محادثة |
| `POST` | `/ask` | سؤال (دعم Streaming) |

---

## التقنيات

| التقنية | الاستخدام |
|---------|----------|
| FastAPI | REST API + Streaming |
| Qdrant Cloud | Vector Database |
| Cohere | Embeddings + Rerank |
| Groq LLaMA | توليد الردود |
| LlamaIndex | Dense Search |
| BM25Okapi | Sparse Search |
| SQLite | المحادثات |
| Docker | النشر |

---

## الترخيص

MIT License — انظر [LICENSE](LICENSE)
