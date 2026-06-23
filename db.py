from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# رابط قاعدة البيانات (ملف محلي باسم compliance.db سيتم إنشاؤه تلقائياً)
SQLALCHEMY_DATABASE_URL = "sqlite:///./compliance.db"

# إنشاء محرك قاعدة البيانات
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# إنشاء مصنع الجلسات للتواصل مع قاعدة البيانات
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# الفئة الأساسية التي سترث منها جميع الجداول
Base = declarative_base()

# دالة لفتح وإغلاق الجلسة مع كل عملية
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()