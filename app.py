
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. قراءة البيانات
df = pd.read_csv('clients_with_fatf_ofac.csv')

# 2. إنشاء معيار الحقيقة (Ground Truth) لتدريب المودل عليه
def determine_true_risk(row):
    if row['fatf_country_flag'] == 1 or row['ofac_country_flag'] == 1 or row['sanctions_flag'] == 1:
        return 1 # عالي الخطورة
    elif row['pep_flag'] == 1 and row['sectoral_sanctions_flag'] == 1:
        return 1 # عالي الخطورة
    else:
        return 0 # طبيعي

df['True_Risk_Label'] = df.apply(determine_true_risk, axis=1)

# 3. معالجة البيانات النصية
features = ['client_type', 'sector_risk', 'pep_flag', 'sanctions_flag', 
            'fatf_country_flag', 'ofac_country_flag', 'sectoral_sanctions_flag', 
            'ownership_opacity_score']

X = df[features].copy()
y = df['True_Risk_Label']

le_client_type = LabelEncoder()
X['client_type'] = le_client_type.fit_transform(X['client_type'])

risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
X['sector_risk'] = X['sector_risk'].map(risk_mapping)
X.fillna(0, inplace=True)

# 4. تقسيم البيانات (80% تدريب - 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 5. بناء نموذج Random Forest وتدريبه
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
import joblib

# بعد سطر: rf_model.fit(X_train, y_train)
# قم بإضافة هذين السطرين:
joblib.dump(rf_model, 'aml_rf_model.pkl')
joblib.dump(le_client_type, 'client_type_encoder.pkl')
print("تم حفظ النموذج بنجاح!")

# 6. التنبؤ على الـ 20% المخصصة للاختبار
y_pred = rf_model.predict(X_test)

# 7. حساب جميع مقاييس الأداء
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("--- نتائج تقييم أداء نموذج Random Forest ---")
print(f"Accuracy  : {accuracy:.4f} ({accuracy*100}%)")
print(f"Precision : {precision:.4f} ({precision*100}%)")
print(f"Recall    : {recall:.4f} ({recall*100}%)")
print(f"F1 Score  : {f1:.4f} ({f1*100}%)")

# 8. استخراج أهمية المعايير (Feature Importance)
feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("\n--- وزن المعايير وتأثيرها على القرار ---")
print(feature_importances)

# 9. رسم مصفوفة الارتباك
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Normal (0)', 'High Risk (1)'], 
            yticklabels=['Normal (0)', 'High Risk (1)'])
plt.title('Confusion Matrix: Random Forest Classifier (20% Test Data)')
plt.ylabel('True Risk Label (Actual)')
plt.xlabel('AI Prediction')
plt.show()
