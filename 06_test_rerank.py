import os, cohere, time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding

load_dotenv()

# ── إعداد ─────────────────────────────────────────────────────────────────
co = cohere.Client(os.getenv("COHERE_API_KEY"))
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=60)

Settings.embed_model = CohereEmbedding(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-multilingual-v3.0",
    input_type="search_query",
)
Settings.llm = None

def get_retriever(collection, top_k=10):
    store = QdrantVectorStore(client=qdrant, collection_name=collection)
    index = VectorStoreIndex.from_vector_store(store)
    return index.as_retriever(similarity_top_k=top_k)

retriever_laws  = get_retriever("legal_laws",  top_k=10)
retriever_osoul = get_retriever("legal_osoul", top_k=10)
retriever_fonon = get_retriever("legal_fonon", top_k=10)

SEP = "═" * 60

# ══════════════════════════════════════════════════════════════
def rerank(query: str, nodes: list, top_n: int = 4) -> list:
    """يُعيد ترتيب النتائج بدقة فائقة عبر استغلال كامل السياق والميتا-داتا"""
    if not nodes: return []
    
    rich_documents = []
    for n in nodes:
        meta = n.node.metadata
        
        # 1. جلب الهوية (رقم المادة أو العنوان)
        identity = meta.get("article_str") or meta.get("title") or ""
        category = meta.get("category", "")
        
        # 2. جلب النص الكامل (رفع القيد لضمان الدقة في المسار الثاني)
        # نستخدم 2000 حرف لضمان شمولية المواد القانونية الطويلة
        full_text = (meta.get("original_text") or meta.get("formal_text") or n.node.get_content())[:2000]
        
        # 3. بناء السياق المدمج
        # نضيف الهوية دائماً لأنها تعطي وزناً كبيراً في البحث القانوني
        context = f"المعرف: {identity} | التصنيف: {category} | النص: {full_text}"
            
        rich_documents.append(context)

    # استخدام موديل Cohere Rerank v3.0 (الأقوى حالياً للمحتوى العربي)
    response = co.rerank(
        query=query,
        documents=rich_documents,
        top_n=top_n,
        model="rerank-multilingual-v3.0",
    )
    
    reranked = []
    for r in response.results:
        node = nodes[r.index]
        node.score = r.relevance_score
        reranked.append(node)
    return reranked
def compare_to_file(query: str, retriever, label: str, file_handle):
    """يقارن النتائج ويكتبها مباشرة في ملف"""
    file_handle.write(f"\n{SEP}\n 🟢 {label}\n السؤال: {query}\n{SEP}\n")

    nodes = retriever.retrieve(query)

    file_handle.write("\n🔹 قبل Rerank (أفضل 4 حسب المتجهات):\n")
    for i, n in enumerate(nodes[:4], 1):
        meta  = n.node.metadata
        score = round(n.score or 0, 3)
        title = meta.get("article_str") or meta.get("title", "بدون عنوان")
        file_handle.write(f"    [{i}] score={score:5.3f} | {title}\n")

    reranked = rerank(query, nodes, top_n=4)
    
    file_handle.write("\n🎯 بعد Rerank (أفضل 4 حسب الفهم الذكي المدمج):\n")
    for i, n in enumerate(reranked, 1):
        meta  = n.node.metadata
        score = round(n.score or 0, 3)
        title = meta.get("article_str") or meta.get("title", "بدون عنوان")
        text  = (meta.get("original_text") or meta.get("text") or meta.get("formal_text") or n.node.get_content())[:150]
        file_handle.write(f"    [{i}] score={score:5.3f} | {title}\n")
        file_handle.write(f"         النص: {text.replace(chr(10), ' ')}...\n")

# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    output_filename = "test_results_combined.txt"
    
    print(f"🚀 جاري الفحص بالذكاء المدمج وحفظ النتائج في {output_filename}...")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"تقرير اختبار نظام البحث القانوني (النسخة المدمجة) - {time.ctime()}\n")
        
        compare_to_file("صياغة دعوى طلاق خلعي مخالعة", retriever_fonon, "مسار 1 — صياغة وفنون", f)
        compare_to_file("شروط الحضانة وأسباب سقوطها عن الأم", retriever_laws, "مسار 2 — أحوال شخصية", f)
        compare_to_file("إجراءات تبليغ المدعى عليه ومواعيد تقديم الجواب", retriever_osoul, "مسار 3 — أصول محاكمات", f)

    print("✅ تم الانتهاء! افتح الملف الآن لرؤية النتائج العالية في جميع المسارات.")