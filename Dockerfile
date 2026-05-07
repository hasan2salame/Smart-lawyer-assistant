# Dockerfile
# المساعد القانوني السوري

FROM python:3.11-slim

# متغيرات البيئة
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# تثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ الكود
COPY . .

# إنشاء مجلدات البيانات
RUN mkdir -p data/raw data/processed

# المنفذ
EXPOSE 8000

# تشغيل السيرفر
# ملاحظة: داخل Docker لا يفتح المتصفح — افتح http://localhost:8000 يدوياً
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
