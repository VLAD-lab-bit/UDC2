import os
import json
import joblib
from utils import read_files_from_folder
from extract_keywords import extract_keywords_and_phrases

# Загрузка предобученной модели классификации
model = joblib.load('../models/udc_classifier.pkl')

# Определяем функцию для классификации текстов
def classify_texts(texts):
    results = {}
    for filename, text in texts.items():
        keywords = extract_keywords_and_phrases(text)
        udc_code = model.predict([text])[0]
        results[filename] = {'keywords': keywords, 'udc_code': udc_code}
    return results

# Основная функция
def main():
    folder_path = '../data/articles'  # Укажите путь к папке с текстами статей
    output_folder = '../output'  # Папка для сохранения выходного файла
    output_file = os.path.join(output_folder, 'results.json')  # Имя файла для записи результатов

    # Создаем папку для выходных данных, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Читаем файлы из папки
    texts = read_files_from_folder(folder_path)

    # Классифицируем тексты
    results = classify_texts(texts)

    # Записываем результаты в файл
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()



"""import os
import json
from utils import read_files_from_folder
from extract_keywords import extract_keywords

# Словарь ключевых слов для УДК
udc_keywords = {
    '0': ['наука', 'информация', 'технологии'],
    '1': ['философия', 'психология'],
    '2': ['религия', 'теология'],
    '30': ['общественные науки', 'методы', 'теория'],
    '31': ['демография', 'социология', 'статистика'],
    '32': ['политика'],
    '33': ['экономика', 'народное хозяйство'],
    '34': ['право', 'юридические науки'],
    '35': ['государственное управление', 'военное искусство', 'военные науки'],
    '36': ['социальное обеспечение', 'материальные потребности'],
    '37': ['образование', 'воспитание', 'обучение', 'досуг'],
    '39': ['этнография', 'нравы', 'обычаи', 'фольклор'],
    '50': ['естественные науки', 'математика'],
    '51': ['математика'],
    '52': ['астрономия', 'геодезия'],
    '53': ['физика'],
    '54': ['химия', 'кристаллография', 'минералогия'],
    '55': ['геология', 'геофизика'],
    '56': ['палеонтология'],
    '57': ['биология'],
    '58': ['ботаника'],
    '59': ['зоология'],
    '60': ['прикладные науки'],
    '61': ['медицина', 'охрана здоровья'],
    '62': ['инженерное дело', 'техника'],
    '63': ['сельское хозяйство', 'лесное хозяйство'],
    '64': ['домоводство', 'коммунальное хозяйство'],
    '65': ['управление', 'производство', 'торговля', 'транспорт'],
    '66': ['химическая технология', 'пищевая промышленность', 'металлургия'],
    '67': ['промышленность', 'ремесла', 'механическая технология'],
    '68': ['точная механика'],
    '69': ['строительство', 'строительные материалы'],
    '37': ['искусство', 'декоративно-прикладное искусство', 'фотография', 'музыка', 'игры', 'спорт'],
    '8': ['языкознание', 'филология', 'литература'],
    '9': ['география', 'биография', 'история']
}

# Функция для классификации текста на основе ключевых слов
def classify_text(text):
    keywords = extract_keywords(text)  # Извлекаем ключевые слова из текста
    keyword_set = set(keywords)  # Преобразуем ключевые слова в множество для удобства

    best_match = None  # Переменная для хранения лучшего совпадения
    best_match_count = 0  # Переменная для хранения количества совпадений

    # Проходим по каждому коду УДК и его ключевым словам
    for udc_code, kw_list in udc_keywords.items():
        match_count = len(keyword_set.intersection(kw_list))  # Считаем количество совпадающих ключевых слов
        if match_count > best_match_count:
            best_match = udc_code  # Обновляем лучший код УДК
            best_match_count = match_count  # Обновляем количество совпадений

    return best_match  # Возвращаем лучший код УДК

# Основная функция
def main():
    folder_path = '../data/articles'  # Укажите путь к папке с текстами статей
    output_folder = '../output'  # Папка для сохранения выходного файла
    output_json_file = os.path.join(output_folder, 'results.json')  # Имя файла для записи результатов в формате JSON
    output_txt_file = os.path.join(output_folder, 'results.txt')  # Имя файла для записи результатов в формате TXT

    # Создаем папку для выходных данных, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Читаем файлы из папки
    texts = read_files_from_folder(folder_path)

    results = {}
    for filename, text in texts.items():
        udc_code = classify_text(text)  # Классифицируем текст
        keywords = extract_keywords(text)  # Извлекаем ключевые слова
        results[filename] = {'keywords': keywords, 'udc_code': udc_code}  # Сохраняем результаты

    # Записываем результаты в файл JSON
    with open(output_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    # Записываем результаты в файл TXT
    with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
        for filename, data in results.items():
            txt_file.write(f'--- {filename} ---\n')
            txt_file.write(f'Текст статьи:\n{texts[filename]}\n\n')  # Добавляем текст статьи
            txt_file.write(f'Ключевые слова: {", ".join(data["keywords"])}\n')
            txt_file.write(f'Индекс УДК: {data["udc_code"]}\n\n')

if __name__ == "__main__":
    main()
    
    
    
    import os
import json
import joblib
from utils import read_files_from_folder, clean_text
from extract_keywords import extract_keywords

# Загрузка предобученной модели классификации
model = joblib.load('../models/udc_classifier.pkl')

# Словарь ключевых слов для УДК (используется для дополнительной классификации)
udc_keywords = {
    '0': ['наука', 'информация', 'технология'],
    '1': ['философия', 'психология'],
    '2': ['религия', 'теология'],
    '30': ['общественный', 'метод', 'теория'],
    '31': ['демография', 'социология', 'статистика'],
    '32': ['политика'],
    '33': ['экономика', 'народное хозяйство'],
    '34': ['право', 'юридический'],
    '35': ['государственное управление', 'военное искусство', 'военный'],
    '36': ['социальное обеспечение', 'материальный'],
    '37': ['образование', 'воспитание', 'обучение', 'досуг'],
    '39': ['этнография', 'нравы', 'обычай', 'фольклор'],
    '50': ['естественный наука', 'математика'],
    '51': ['математика'],
    '52': ['астрономия', 'геодезия'],
    '53': ['физика'],
    '54': ['химия', 'кристаллография', 'минералогия'],
    '55': ['геология', 'геофизика'],
    '56': ['палеонтология'],
    '57': ['биология'],
    '58': ['ботаника'],
    '59': ['зоология'],
    '60': ['прикладной наука'],
    '61': ['медицина', 'охрана здоровье'],
    '62': ['инженерное дело', 'техника'],
    '63': ['сельское хозяйство', 'лесное хозяйство'],
    '64': ['домоводство', 'коммунальное хозяйство'],
    '65': ['управление', 'производство', 'торговля', 'транспорт'],
    '66': ['химическая технология', 'пищевая промышленность', 'металлургия'],
    '67': ['промышленность', 'ремесло', 'механическая технология'],
    '68': ['точная механика'],
    '69': ['строительство', 'строительные материалы'],
    '7': ['искусство', 'декоративно-прикладное искусство', 'фотография', 'музыка', 'игры', 'спорт'],
    '8': ['языкознание', 'филология', 'литература'],
    '9': ['география', 'биография', 'история']
}

# Функция для классификации текста на основе ключевых слов и обученной модели
def classify_text(text, top_n=2):
    # Извлекаем ключевые слова из текста
    keywords = extract_keywords(text)
    keyword_set = set(keywords)

    # Классификация с использованием обученной модели
    model_prediction = model.predict([text])[0]

    # Сравнение ключевых слов с словарем УДК
    matches = []
    for udc_code, kw_list in udc_keywords.items():
        match_count = len(keyword_set.intersection(kw_list))
        if match_count > 0:
            matches.append((udc_code, match_count))

    matches = sorted(matches, key=lambda x: x[1], reverse=True)[:top_n]
    best_matches = [udc_code for udc_code, _ in matches]

    # Возвращаем результат модели и совпадающие ключевые слова
    if model_prediction not in best_matches:
        best_matches.insert(0, model_prediction)

    return best_matches

# Основная функция
def main():
    folder_path = '../data/articles'  # Укажите путь к папке с текстами статей
    output_folder = '../output'  # Папка для сохранения выходного файла
    output_json_file = os.path.join(output_folder, 'results.json')  # Имя файла для записи результатов в формате JSON
    output_txt_file = os.path.join(output_folder, 'results.txt')  # Имя файла для записи результатов в формате TXT

    # Создаем папку для выходных данных, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Читаем файлы из папки
    texts = read_files_from_folder(folder_path)

    results = {}
    for filename, text in texts.items():
        udc_codes = classify_text(text)  # Классифицируем текст
        keywords = extract_keywords(text)  # Извлекаем ключевые слова
        results[filename] = {'keywords': keywords, 'udc_codes': udc_codes}  # Сохраняем результаты

    # Записываем результаты в файл JSON
    with open(output_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

    # Записываем результаты в файл TXT
    with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
        for filename, data in results.items():
            txt_file.write(f'--- {filename} ---\n')
            txt_file.write(f'Текст статьи:\n{texts[filename]}\n\n')  # Добавляем текст статьи
            txt_file.write(f'Ключевые слова: {", ".join(data["keywords"])}\n')
            txt_file.write(f'Индексы УДК: {", ".join(data["udc_codes"])}\n\n')

if __name__ == "__main__":
    main()

"""