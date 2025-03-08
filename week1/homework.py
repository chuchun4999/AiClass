import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 CSV 파일 읽기
file_path = "./week1/iris.csv" 
df = pd.read_csv(file_path)

# 데이터프레임 확인
print(df)

from sklearn.model_selection import train_test_split
X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']] # 특성 (sepal length, sepal width, petal length, petal width)
y = df['Name']

# 문자열 라벨을 숫자형으로 변환 (Label Encoding)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# (1) DF
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# (2) RF
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# (3) SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

# (4) LR
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# 결과 출력
print("Decision Tree accuracy:", dt_acc)
print("Random Forest accuracy:", rf_acc)
print("SVM accuracy:", svm_acc)
print("Logistic Regression accuracy:", lr_acc)