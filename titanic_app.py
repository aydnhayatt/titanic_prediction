import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import joblib  # Modeli kaydetmek için


# Veri setlerini yükle
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train.isnull().sum()) #eksik veri olup olmadığına bakılır.
#print(test.isnull().sum())

# Eksik yaş değerlerini medyan ile doldur
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

# 'Cabin' sütunundaki eksik veriyi işaretlemek için yeni bir özellik oluştur (Kabini var mı?) yoksa 0 varsa 1 .
train['Has_Cabin'] = train['Cabin'].notnull().astype(int)
test['Has_Cabin'] = test['Cabin'].notnull().astype(int)

# 'Cabin' sütununu veri setlerinden kaldır
train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

# 'Embarked' sütunundaki eksik verileri en sık kullanılan liman ile doldurur.
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Test setindeki eksik 'Fare' değerlerini medyan ile doldur
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Yolcu isimlerinden unvanları çıkarır unvanlar için "Title" sütunu ekler.
def get_title(name):
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# Train ve test setlerine unvan sütunu ekler.
train['Title'] = train['Name'].apply(get_title)
test['Title'] = test['Name'].apply(get_title)

# Nadir unvanları 'Rare' olarak gruplandır ve benzer unvanları birleştir
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
               'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

def replace_rare_titles(title):
    if title in rare_titles:
        return 'Rare'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    else:
        return title

train['Title'] = train['Title'].apply(replace_rare_titles)
test['Title'] = test['Title'].apply(replace_rare_titles)

# Cinsiyeti sayısallaştır: erkek=0, kadın=1
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# Kategorik değişkenleri (Embarked ve Title) one-hot encode ile sayısallaştır
train = pd.get_dummies(train, columns=['Embarked', 'Title'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked', 'Title'], drop_first=True)

# Yeni özellikler oluştur/birleştirir.
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1  # Aile büyüklüğü
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train['IsAlone'] = (train['FamilySize'] == 1).astype(int)  # Tek başına mı?
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)

# Train ve test setlerindeki sütunları hizala (aynı sütunlar olsun)
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Modelde kullanılacak özellikler
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch',
            'Embarked_Q', 'Embarked_S', 'Has_Cabin', 'FamilySize', 'IsAlone']

# Bağımsız değişkenler ve hedef değişken
X_train = train[features]
y_train = train['Survived']
X_test = test[features]

# Eğitim ve doğrulama setine ayır (overfitting önlemek için).
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Random Forest modeli oluştur ve eğit.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)

#Modeli eğittikten sonra kaydedilir.
joblib.dump(model, 'titanic_rf_model.pkl')

# Doğrulama setinde tahmin yap
val_preds = model.predict(X_val)

# Performans ölçülerini yazdırır.
print("🎯 Validation Accuracy:", accuracy_score(y_val, val_preds))
print("\n📋 Doğrulama Sınıflandırma Raporu:\n")
print(classification_report(y_val, val_preds))

# Confusion Matrix ile görselleştirir.
cm = confusion_matrix(y_val, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Hayatta Kalmadı (0)", "Hayatta Kaldı (1)"])
disp.plot(cmap="Blues")
plt.show()
