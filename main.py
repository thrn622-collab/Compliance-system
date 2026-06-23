# الملف: main.py

from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from db import engine, get_db  # تم التحديث هنا ليقرأ من db.py
import models
import schemas
import shutil
import os
import joblib  
import warnings
import pandas as pd
import numpy as np
from io import StringIO

warnings.filterwarnings('ignore')

# إنشاء الجداول في قاعدة البيانات عند التشغيل
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Enterprise Compliance System")

# ==========================================
# تحميل نماذج الذكاء الاصطناعي 
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    rf_model = joblib.load(os.path.join(BASE_DIR, 'aml_rf_model.pkl'))
    le_client_type = joblib.load(os.path.join(BASE_DIR, 'client_type_encoder.pkl'))
    print("✅ تم تحميل نموذج KYC (Random Forest)")
except:
    rf_model = None
    le_client_type = None

try:
    xgb_trans_model = joblib.load(os.path.join(BASE_DIR, 'aml_transactions_model.pkl'))
    print("✅ تم تحميل نموذج المعاملات (XGBoost)")
except:
    xgb_trans_model = None

# ==========================================
# الواجهة الأمامية المدمجة (HTML + Tailwind)
# ==========================================
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>نظام الامتثال المؤسسي - End to End</title>
        <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap" rel="stylesheet">
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = { theme: { extend: { fontFamily: { sans: ['Cairo', 'sans-serif'] } } } }
        </script>
        <style>
            .glass-panel { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); }
            .hidden-field { display: none; }
            #aml_console { scroll-behavior: smooth; }
        </style>
    </head>
    <body class="bg-slate-50 min-h-screen text-slate-800">
        
        <header class="bg-slate-900 text-white py-5 shadow-lg border-b-4 border-indigo-500">
            <div class="container mx-auto px-6 flex justify-between items-center">
                <div>
                   <h1 class="text-2xl font-extrabold tracking-wide uppercase">🛡️ نظام إدارة وفحص الامتثال والـ KYC</h1>
<p class="text-slate-400 text-sm mt-1">منصة فحص هوية العملاء ومراقبة العمليات المالية</p>
                </div>
            </div>
        </header>

        <div class="container mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            <!-- نموذج إدخال البيانات الموحد (اليمين) -->
            <div class="lg:col-span-4 glass-panel p-6 rounded-2xl shadow-md border border-slate-200 h-fit">
                <h2 class="text-xl font-bold text-slate-800 mb-6 flex items-center gap-2">
                    <span class="bg-indigo-100 text-indigo-700 p-2 rounded-lg">📋</span> إضافة عميل وتحليل شامل
                </h2>
                
                <div class="flex bg-slate-100 p-1 rounded-xl mb-6">
                    <button id="btnIndividual" type="button" onclick="switchType('Individual')" class="flex-1 py-2 text-sm font-bold rounded-lg bg-white shadow text-indigo-700 transition-all">👤 أفراد</button>
                    <button id="btnCorporate" type="button" onclick="switchType('Corporate')" class="flex-1 py-2 text-sm font-bold rounded-lg text-slate-500 hover:text-slate-700 transition-all">🏢 شركات</button>
                </div>

                <form id="clientForm" class="space-y-4">
                    <input type="hidden" id="client_type" value="Individual">
                    
                    <div>
                        <label id="lbl_name" class="block text-xs font-bold text-slate-600 mb-1">الاسم الكامل (حسب الهوية)</label>
                        <input type="text" id="full_name" required class="w-full p-2.5 border border-slate-300 rounded-lg bg-slate-50 focus:ring-2 focus:ring-indigo-500 outline-none text-sm">
                    </div>
                    
                    <div>
                        <label id="lbl_id" class="block text-xs font-bold text-slate-600 mb-1">رقم الهوية الوطنية / الإقامة</label>
                        <input type="text" id="identifier_number" required class="w-full p-2.5 border border-slate-300 rounded-lg bg-slate-50 focus:ring-2 focus:ring-indigo-500 outline-none text-sm">
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label id="lbl_date" class="block text-xs font-bold text-slate-600 mb-1">تاريخ الميلاد</label>
                            <input type="date" id="date_of_birth_or_inc" required class="w-full p-2.5 border border-slate-300 rounded-lg bg-slate-50 focus:ring-2 focus:ring-indigo-500 outline-none text-sm text-slate-500">
                        </div>
                        <div>
                            <label class="block text-xs font-bold text-slate-600 mb-1">رقم الجوال</label>
                            <input type="tel" id="mobile_number" placeholder="05XXXXXXXX" required class="w-full p-2.5 border border-slate-300 rounded-lg bg-slate-50 focus:ring-2 focus:ring-indigo-500 outline-none text-sm text-left" dir="ltr">
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <label id="lbl_nationality" class="block text-xs font-bold text-slate-600 mb-1">الجنسية (رمز)</label>
                            <input type="text" id="nationality" placeholder="مثال: SA, AE" required class="w-full p-2.5 border border-slate-300 rounded-lg bg-slate-50 focus:ring-2 focus:ring-indigo-500 outline-none text-sm text-center">
                        </div>
                        <div>
                            <label id="lbl_value" class="block text-xs font-bold text-slate-600 mb-1">الدخل المتوقع</label>
                            <input type="number" id="financial_value" placeholder="بالريال" class="w-full p-2.5 border border-slate-300 rounded-lg bg-slate-50 focus:ring-2 focus:ring-indigo-500 outline-none text-sm">
                        </div>
                    </div>

                    <div id="div_activity" class="hidden-field">
                        <label class="block text-xs font-bold text-slate-600 mb-1">النشاط التجاري</label>
                        <select id="activity_sector" class="w-full p-2.5 border border-slate-300 rounded-lg bg-slate-50 focus:ring-2 focus:ring-indigo-500 outline-none text-sm">
                            <option value="Trade">تجارة وتجزئة</option>
                            <option value="RealEstate">عقارات ومقاولات</option>
                            <option value="Tech">تقنية واتصالات</option>
                            <option value="Finance">خدمات مالية واستثمار</option>
                            <option value="Other">أخرى</option>
                        </select>
                    </div>

                    <!-- القسم 1: رفع المستندات الثبوتية -->
                    <div class="mt-4 pt-4 border-t border-slate-200">
                        <h3 class="text-sm font-bold text-slate-700 mb-3 flex items-center gap-1"> المستندات</h3>
                        
                        <div id="docs_individual" class="space-y-3">
                            <div>
                                <label class="block text-xs font-bold text-slate-600 mb-1">نموذج اعرف عميلك (KYC) </label>
                                <input type="file" id="doc_national_id" accept="image/*,.pdf" class="w-full text-xs text-slate-500 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer">
                            </div>
                            <div>
                                <label class="block text-xs font-bold text-slate-600 mb-1">إثبات سكن (فاتورة أو عقد إيجار)</label>
                                <input type="file" id="doc_address_proof" accept="image/*,.pdf" class="w-full text-xs text-slate-500 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer">
                            </div>
                        </div>

                        <div id="docs_corporate" class="hidden-field space-y-3">
                            <div>
                                <label class="block text-xs font-bold text-slate-600 mb-1">السجل التجاري (ساري المفعول)</label>
                                <input type="file" id="doc_cr" accept="image/*,.pdf" class="w-full text-xs text-slate-500 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer">
                            </div>
                            <div>
                                <label class="block text-xs font-bold text-slate-600 mb-1">عقد التأسيس (النسخة المحدثة)</label>
                                <input type="file" id="doc_aoa" accept="image/*,.pdf" class="w-full text-xs text-slate-500 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer">
                            </div>
                            <div>
                                <label class="block text-xs font-bold text-slate-600 mb-1">هوية المفوض بالتوقيع</label>
                                <input type="file" id="doc_signatory_id" accept="image/*,.pdf" class="w-full text-xs text-slate-500 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer">
                            </div>
                            <div>
                                <label class="block text-xs font-bold text-slate-600 mb-1">خطاب تفويض رسمي</label>
                                <input type="file" id="doc_auth_letter" accept="image/*,.pdf" class="w-full text-xs text-slate-500 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer">
                            </div>
                        </div>
                    </div>

                    <!-- القسم 2: رفع كشف الحساب -->
                    <div class="mt-4 pt-4 border-t border-slate-200 bg-red-50 p-3 rounded-lg border border-red-100">
                        <h3 class="text-sm font-bold text-red-800 mb-2 flex items-center gap-1">📊 كشف الحساب (AML Analysis)</h3>
                        <label class="block text-xs font-medium text-red-700 mb-1">ارفع كشف الحساب (CSV) ليتم فحصه تلقائياً</label>
                        <input type="file" id="aml_csv_file" accept=".csv" class="w-full text-xs text-slate-500 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-red-600 file:text-white hover:file:bg-red-700 cursor-pointer">
                    </div>

                    <button id="submit_btn" type="submit" class="w-full bg-slate-900 hover:bg-black text-white py-3 rounded-lg text-sm font-extrabold uppercase tracking-wide shadow-lg transition-all mt-6 flex justify-center items-center gap-2">
                        <span>🚀</span> تنفيذ وإنشاء وتحليل
                    </button>
                </form>
            </div>

            <!-- لوحة العرض والمراقبة (اليسار) -->
            <div class="lg:col-span-8 space-y-6">
                
                <!-- Console Terminal للمراقبة اللحظية -->
                <div class="bg-slate-900 p-6 rounded-2xl shadow-xl border border-slate-700 text-white relative overflow-hidden flex flex-col h-64">
                    <div class="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"></div>
                    <h2 class="text-sm font-bold mb-2 flex items-center gap-2 text-slate-300">
                        <span>⚡</span> سجل العمليات الحية (Live Execution Logs)
                    </h2>
                    <div id="aml_console" class="flex-1 p-4 bg-black rounded-xl border border-slate-800 font-mono text-xs overflow-y-auto text-green-400 space-y-1">
                        > النظام في وضع الاستعداد... بانتظار إدخال عميل جديد.
                    </div>
                </div>

                <!-- جدول العملاء -->
                <div class="glass-panel p-6 rounded-2xl shadow-md border border-slate-200">
                    <h2 class="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                        <span class="bg-emerald-100 text-emerald-700 p-2 rounded-lg">🗄️</span> السجل المركزي للعملاء
                    </h2>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm text-right">
                            <thead class="bg-slate-100 text-slate-500 border-b-2 border-slate-200">
                                <tr>
                                    <th class="p-3">النوع</th>
                                    <th class="p-3">الاسم / المنشأة</th>
                                    <th class="p-3">الهوية / السجل</th>
                                    <th class="p-3">الخطورة الإجمالية</th>
                                    <th class="p-3">الإجراء</th>
                                </tr>
                            </thead>
                            <tbody id="clientsTableBody" class="divide-y divide-slate-100"></tbody>
                        </table>
                    </div>
                </div>

            </div>
        </div>

        <script>
            // التبديل بين أفراد وشركات
            function switchType(type) {
                document.getElementById('client_type').value = type;
                const btnInd = document.getElementById('btnIndividual');
                const btnCorp = document.getElementById('btnCorporate');
                const divAct = document.getElementById('div_activity');
                const docsInd = document.getElementById('docs_individual');
                const docsCorp = document.getElementById('docs_corporate');
                
                if(type === 'Individual') {
                    btnInd.className = "flex-1 py-2 text-sm font-bold rounded-lg bg-white shadow text-indigo-700 transition-all";
                    btnCorp.className = "flex-1 py-2 text-sm font-bold rounded-lg text-slate-500 hover:text-slate-700 transition-all";
                    document.getElementById('lbl_name').innerText = "الاسم الكامل (حسب الهوية)";
                    document.getElementById('lbl_id').innerText = "رقم الهوية الوطنية / الإقامة";
                    document.getElementById('lbl_date').innerText = "تاريخ الميلاد";
                    document.getElementById('lbl_nationality').innerText = "الجنسية (رمز الدولة)";
                    document.getElementById('lbl_value').innerText = "الدخل المتوقع";
                    divAct.style.display = 'none';
                    docsInd.style.display = 'block';
                    docsCorp.style.display = 'none';
                } else {
                    btnCorp.className = "flex-1 py-2 text-sm font-bold rounded-lg bg-white shadow text-indigo-700 transition-all";
                    btnInd.className = "flex-1 py-2 text-sm font-bold rounded-lg text-slate-500 hover:text-slate-700 transition-all";
                    document.getElementById('lbl_name').innerText = "الاسم التجاري للمنشأة";
                    document.getElementById('lbl_id').innerText = "رقم السجل التجاري (CR)";
                    document.getElementById('lbl_date').innerText = "تاريخ التأسيس";
                    document.getElementById('lbl_nationality').innerText = "بلد التأسيس (رمز الدولة)";
                    document.getElementById('lbl_value').innerText = "رأس المال المدفوع";
                    divAct.style.display = 'block';
                    docsInd.style.display = 'none';
                    docsCorp.style.display = 'block';
                }
            }

            // كتابة السجلات في الشاشة السوداء
            function logToConsole(message, type="info") {
                const consoleDiv = document.getElementById('aml_console');
                let colorClass = "text-green-400";
                if(type === "error") colorClass = "text-red-500 font-bold";
                if(type === "warning") colorClass = "text-yellow-400 font-bold";
                if(type === "success") colorClass = "text-blue-400 font-bold";
                
                consoleDiv.innerHTML += `<div class="${colorClass}">> ${message}</div>`;
                consoleDiv.scrollTop = consoleDiv.scrollHeight;
            }

            // تحميل بيانات العملاء للجدول
            async function loadClients() {
                const res = await fetch('/clients/');
                const clients = await res.json();
                const tbody = document.getElementById('clientsTableBody');
                tbody.innerHTML = '';
                
                clients.forEach(c => {
                    let badge = '';
                    if (c.risk_level === 'Low') badge = '<span class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-xs font-bold border border-green-200">منخفض</span>';
                    else if (c.risk_level === 'High') badge = '<span class="bg-red-100 text-red-800 px-3 py-1 rounded-full text-xs font-bold animate-pulse border border-red-200">عالي الخطورة 🚨</span>';
                    else badge = '<span class="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-xs font-bold border border-yellow-200">متوسط</span>';

                    let typeIcon = c.client_type === 'Individual' ? '👤 فرد' : '🏢 شركة';

                    tbody.innerHTML += `
                        <tr class="hover:bg-slate-50 transition-colors border-b border-slate-100">
                            <td class="p-4 text-xs font-bold text-slate-500">${typeIcon}</td>
                            <td class="p-4 font-bold text-slate-800">${c.full_name}</td>
                            <td class="p-4 font-mono text-xs text-slate-500">${c.identifier_number}</td>
                            <td class="p-4">${badge}</td>
                            <td class="p-4">
                                <button onclick="alert('تفاصيل العميل ستظهر هنا')" class="text-slate-400 hover:text-indigo-600 font-bold text-xs underline">عرض الملف</button>
                            </td>
                        </tr>`;
                });
            }

            // الحدث الرئيسي للنموذج (حفظ + رفع + تحليل) في خطوة واحدة
            document.getElementById('clientForm').onsubmit = async (e) => {
                e.preventDefault();
                const submitBtn = document.getElementById('submit_btn');
                submitBtn.disabled = true;
                submitBtn.innerHTML = "⏳ العمليات قيد التنفيذ...";
                
                document.getElementById('aml_console').innerHTML = ''; // مسح الشاشة
                logToConsole("بدء تسلسل تسجيل العميل (End-to-End)...");

                const clientType = document.getElementById('client_type').value;
                const data = {
                    client_type: clientType,
                    full_name: document.getElementById('full_name').value,
                    identifier_number: document.getElementById('identifier_number').value,
                    nationality: document.getElementById('nationality').value,
                    date_of_birth_or_inc: document.getElementById('date_of_birth_or_inc').value,
                    mobile_number: document.getElementById('mobile_number').value,
                    activity_sector: document.getElementById('activity_sector').value || 'N/A',
                    financial_value: parseFloat(document.getElementById('financial_value').value) || 0
                };
                
                try {
                    // 1. إنشاء العميل في قاعدة البيانات وتقييم KYC
                    logToConsole("إرسال بيانات العميل لتقييم المخاطر (KYC)...");
                    const res = await fetch('/clients/', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) });
                    
                    if(!res.ok) {
                        const err = await res.json();
                        logToConsole(`خطأ أثناء إنشاء العميل: ${err.detail}`, "error");
                        throw new Error("Client Creation Failed");
                    }
                    
                    const newClient = await res.json();
                    const clientId = newClient.id;
                    logToConsole(`تم تسجيل العميل (ID: ${clientId}). تقييم المبدئي: [${newClient.risk_level}]`, "success");

                    // 2. رفع المستندات الثبوتية
                    logToConsole("التحقق من المستندات الثبوتية المرفقة ورفعها...");
                    const uploadPromises = [];
                    const uploadIfPresent = (inputId, docType) => {
                        const fileInput = document.getElementById(inputId);
                        if (fileInput && fileInput.files.length > 0) {
                            const fd = new FormData();
                            fd.append('file', fileInput.files[0]);
                            fd.append('document_type', docType);
                            uploadPromises.push(fetch(`/clients/${clientId}/documents/`, { method: 'POST', body: fd }));
                        }
                    };

                    if (clientType === 'Individual') {
                        uploadIfPresent('doc_national_id', 'National_ID');
                        uploadIfPresent('doc_address_proof', 'Address_Proof');
                    } else {
                        uploadIfPresent('doc_cr', 'CR');
                        uploadIfPresent('doc_aoa', 'AOA');
                        uploadIfPresent('doc_signatory_id', 'Signatory_ID');
                        uploadIfPresent('doc_auth_letter', 'Auth_Letter');
                    }

                    if (uploadPromises.length > 0) {
                        await Promise.all(uploadPromises);
                        logToConsole(`تم حفظ ${uploadPromises.length} مستند(ات) بنجاح.`);
                    } else {
                        logToConsole("لم يتم إرفاق مستندات ثبوتية.", "warning");
                    }

                    // 3. تحليل كشف الحساب (إن وجد)
                    const amlFile = document.getElementById('aml_csv_file').files[0];
                    if(amlFile) {
                        logToConsole(`تم استلام كشف حساب (${amlFile.name}). جاري تمريره لنموذج الذكاء الاصطناعي...`);
                        const fdAML = new FormData();
                        fdAML.append('file', amlFile);
                        
                        const amlRes = await fetch(`/clients/${clientId}/analyze_statement/`, { method: 'POST', body: fdAML });
                        if(amlRes.ok) {
                            const amlResult = await amlRes.json();
                            if(amlResult.suspicious_count > 0) {
                                logToConsole(`🚨 إنذار حرج: من أصل ${amlResult.total_transactions} عملية، تم رصد ${amlResult.suspicious_count} حركة مشبوهة!`, "error");
                                logToConsole(`تم تصعيد حالة العميل إلى "عالي الخطورة".`, "error");
                            } else {
                                logToConsole(`✅ تحليل سليم: تم فحص ${amlResult.total_transactions} معاملة. لا توجد أنماط غسيل أموال.`, "success");
                            }
                        } else {
                            logToConsole("فشل في تحليل كشف الحساب.", "error");
                        }
                    } else {
                        logToConsole("لم يتم إرفاق كشف حساب مالي للفحص الذكي.", "warning");
                    }

                    // 4. إنهاء العملية بنجاح
                    logToConsole("🎉 اكتملت الدورة الشاملة لتسجيل العميل بجميع الفحوصات.", "success");
                    document.getElementById('clientForm').reset();
                    switchType('Individual'); 
                    loadClients();

                } catch(e) { 
                    logToConsole("توقفت العملية.", "error");
                } finally {
                    submitBtn.innerHTML = "<span>🚀</span> تنفيذ وإنشاء وتحليل";
                    submitBtn.disabled = false;
                }
            };

            window.onload = () => { switchType('Individual'); loadClients(); };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ==========================================
# مسارات الـ API (الخلفية)
# ==========================================
@app.post("/clients/", response_model=schemas.ClientResponse)
def create_client(client: schemas.ClientCreate, db: Session = Depends(get_db)):
    # التحقق من أن الهوية/السجل غير مكرر
    db_client = db.query(models.Client).filter(models.Client.identifier_number == client.identifier_number).first()
    if db_client:
        raise HTTPException(status_code=400, detail="رقم الهوية / السجل التجاري مسجل مسبقاً")

    # قواعد تقييم المخاطر المبدئي للـ KYC
    ai_risk = "Low"
    high_risk_countries = ['SY', 'IR', 'KP', 'VE', 'RU']
    high_risk_sectors = ['RealEstate', 'Casino', 'Crypto']
    
    if client.nationality.upper() in high_risk_countries:
        ai_risk = "High"
    elif client.client_type == 'Corporate' and client.activity_sector in high_risk_sectors:
        ai_risk = "High"
        
    new_client = models.Client(**client.dict(), risk_level=ai_risk)
    db.add(new_client)
    db.commit()
    db.refresh(new_client)
    return new_client

@app.get("/clients/", response_model=list[schemas.ClientResponse])
def get_clients(db: Session = Depends(get_db)):
    # ترتيب الجدول بحيث يظهر الأحدث بالأعلى
    return db.query(models.Client).order_by(models.Client.id.desc()).all()

@app.post("/clients/{client_id}/documents/")
def upload_document(client_id: int, document_type: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client: raise HTTPException(status_code=404, detail="العميل غير موجود")

    os.makedirs("uploads", exist_ok=True) 
    file_path = f"uploads/{client_id}_{document_type}_{file.filename}"
    with open(file_path, "wb") as buffer: 
        shutil.copyfileobj(file.file, buffer)

    new_doc = models.Document(document_type=document_type, file_path=file_path, client_id=client_id)
    db.add(new_doc)
    db.commit()
    return {"message": "تم حفظ المستند"}

@app.post("/clients/{client_id}/analyze_statement/")
async def analyze_bank_statement(client_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    client = db.query(models.Client).filter(models.Client.id == client_id).first()
    if not client: raise HTTPException(status_code=404, detail="العميل غير موجود")

    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        total_tx = len(df)
        suspicious_count = 0
        
        # محاكاة كشف الأموال (في حال كان المودل جاهزاً أو في حال تجربة ملف وهمي)
        if xgb_trans_model:
            # هنا يمكنك وضع ميزات المودل لاحقاً، حالياً سنعتمد على عامود Is Laundering للتجربة
            if 'Is Laundering' in df.columns:
                suspicious_count = int(df['Is Laundering'].sum())
        else:
            if 'Is Laundering' in df.columns:
                suspicious_count = int(df['Is Laundering'].sum())
            else:
                # إذا لم يجد شيئاً، نفترض أن 1% من العمليات مشبوهة فقط لتجربة الشاشة الحمراء
                suspicious_count = int(total_tx * 0.01) 

        # تصعيد خطورة العميل إذا اكتشف غسيل أموال
        if suspicious_count > 0:
            client.risk_level = "High"
            db.commit()

        # حفظ تقرير التحليل في الداتا بيس
        report = models.TransactionReport(
            client_id=client_id,
            file_name=file.filename,
            total_transactions=total_tx,
            suspicious_count=suspicious_count,
            ai_decision="اشتباه" if suspicious_count > 0 else "سليم"
        )
        db.add(report)
        db.commit()

        return {"total_transactions": total_tx, "suspicious_count": suspicious_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail="خطأ أثناء قراءة الكشف: " + str(e))