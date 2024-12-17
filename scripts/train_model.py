import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import read_files_from_folder, read_txt, read_docx, read_pdf


# Функция для чтения и разметки данных
def load_data_and_labels(folder_path):
    texts = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') or filename.endswith('.docx') or filename.endswith('.pdf'):
            label = filename.split('_')[0]  # Предполагаем, что имя файла начинается с кода УДК
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.txt'):
                text = read_txt(file_path)
            elif filename.endswith('.docx'):
                text = read_docx(file_path)
            elif filename.endswith('.pdf'):
                text = read_pdf(file_path)
            texts.append(text)
            labels.append(label)
    return texts, labels

# Путь к папке с обучающими данными
train_folder_path = '../data/train_articles'  # Папка с размеченными данными

# Загрузка данных
texts, labels = load_data_and_labels(train_folder_path)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Создание и обучение модели
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Сохранение модели
joblib.dump(model, '../models/udc_classifier.pkl')
