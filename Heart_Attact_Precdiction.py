from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)

# CSV dosyasını oku
file_path = "Heart_Attact_Precdiction_ML_Project\heart.csv"
data = pd.read_csv(file_path)

# Özellikler ve hedefi belirle
X = data.drop("output", axis=1)
y = data["output"]

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli seçilen algoritmaya göre eğitelim
selected_algorithm = "Random Forest"  # Varsayılan algoritma, istediğiniz bir algoritma ile değiştirilebilir

if selected_algorithm == "Decision Tree":
    model = DecisionTreeClassifier()
elif selected_algorithm == "Random Forest":
    model = RandomForestClassifier()
elif selected_algorithm == "k-NN":
    model = KNeighborsClassifier()

# Modeli eğitelim
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate')
def calculate():
    return render_template('calculate.html')

@app.route('/form', methods=['POST'])
def form():
    selected_algorithm = request.form.get('algorithm')
    age = float(request.form.get('age'))
    sex = float(request.form.get('sex'))
    cp = float(request.form.get('cp'))
    trestbps = float(request.form.get('trestbps'))
    chol = float(request.form.get('chol'))
    fbs = float(request.form.get('fbs'))
    restecg = float(request.form.get('restecg'))
    thalach = float(request.form.get('thalach'))
    exang = float(request.form.get('exang'))
    oldpeak = float(request.form.get('oldpeak'))
    slope = float(request.form.get('slope'))
    ca = float(request.form.get('ca'))
    thal = float(request.form.get('thal'))

    # Formdan gelen veriyi kullanarak tahmin yap
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]  # Probability of getting a heart attack

    return render_template('calculate.html', context={
        'selected_algorithm': selected_algorithm,
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'prediction': int(prediction[0]),
        'probability': round(probability[0], 4) * 100  # Convert probability to percentage
    })

# ... (rest of your code remains the same)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    selected_algorithm = request.form.get('algorithm')

    # Seçilen algoritmayı uygulayın
    if selected_algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    elif selected_algorithm == "Random Forest":
        model = RandomForestClassifier()
    elif selected_algorithm == "k-NN":
        model = KNeighborsClassifier()

    # Özellikler ve hedefi belirleyin
    X = data.drop("output", axis=1)
    y = data["output"]

    # Veriyi eğitim ve test setlerine bölin
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # random_state=42 ekledik

    # Modeli eğitin
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapın
    y_pred = model.predict(X_test)

    # Değerlendirme metriklerini hesapla
    accuracy = metrics.accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Confusion matrix'i string olarak dönüştür
    confusion_mat_str = str(confusion_mat)

    # Konsola bilgileri yazdır
    print(f"Seçilen Algoritma: {selected_algorithm}")
    print(f"Doğruluk: {accuracy:.2f}")

    return render_template('index.html',
                            result_text=f"Sonuç (Doğruluk): {accuracy:.2f}",
                           selected_algorithm=selected_algorithm,
                           classification_report=classification_rep,
                           confusion_matrix=confusion_mat_str,
                           accuracy_text=f"Doğruluk: {accuracy:.2f}")

@app.route('/view_dataset')
def view_dataset():
    # Veri setini gösteren bir sayfa döndür
    return render_template('view_dataset.html', data=data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

