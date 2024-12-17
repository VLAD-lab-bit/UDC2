import numpy as np
import spacy
import nltk
import re
from nltk import FreqDist
import pymorphy3
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os


# Инициализируем pymorphy3 для морфологического анализа
morph = pymorphy3.MorphAnalyzer()

# Загрузка ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Загрузка модели spacy для русского языка
nlp = spacy.load('ru_core_news_sm')

def extract_keywords_and_phrases_v2(text, min_keyword_freq=3, keyword_percent=1.5, phrase_percent=5):
    # Анализ текста с помощью spacy
    doc = nlp(text)

    # Извлекаем лемматизированные слова, исключая стоп-слова
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

    # Фильтрация только русских слов
    russian_tokens = [word for word in tokens if re.match(r'^[а-яА-ЯёЁ]+$', word)]

    # Частотный словарь
    freq_dist = FreqDist(russian_tokens)
    save_to_excel(freq_dist, "word_frequencies.xlsx")
    total_count = sum(freq_dist.values())
    keyword_limit = total_count * (keyword_percent / 100)

    # Извлечение ключевых слов с учетом до 3 резких спадов частоты
    keywords = []
    cumulative_freq = 0
    drop_count = 0
    prev_freq = None

    for word, freq in freq_dist.most_common():
        # Условие минимальной частоты
        if freq < min_keyword_freq:
            break
        cumulative_freq += freq
        # Проверка процентного лимита
        if cumulative_freq > keyword_limit:
            break

        # Определение резкого спада частоты
        if prev_freq is not None and prev_freq - freq > prev_freq * 0.5:
            drop_count += 1
            if drop_count > 3:  # Остановка после 3 резких спадов
                break
        prev_freq = freq

        # Добавление слова и его нормальной формы
        normal_form = morph.parse(word)[0].normal_form
        keywords.append(f"{word} ({normal_form})")

    # Извлечение фраз
    key_phrases = []
    for i in range(len(doc) - 1):
        token1, token2 = doc[i], doc[i + 1]

        # Паттерн: прилагательное + существительное
        if token1.pos_ == "ADJ" and token2.pos_ == "NOUN":
            noun_lemma = morph.parse(token2.text)[0].normal_form
            noun_single_nom = morph.parse(noun_lemma)[0].inflect({'nomn', 'sing'})
            noun_nominative = noun_single_nom.word if noun_single_nom else token2.text

            adj_form = morph.parse(token1.text)[0]
            gender = noun_single_nom.tag.gender if noun_single_nom else adj_form.tag.gender
            adj_nominative_form = adj_form.inflect({'nomn', 'sing', gender}) if gender else None
            adj_nominative = adj_nominative_form.word if adj_nominative_form else token1.text

            phrase_original = f"{token1.text.lower()} {token2.text.lower()}"
            phrase_nominative = f"{adj_nominative} {noun_nominative}"
            key_phrases.append(f"{phrase_original} ({phrase_nominative})")

        # Паттерн: существительное + существительное
        elif token1.pos_ == "NOUN" and token2.pos_ == "NOUN":
            noun1_single_nom = morph.parse(token1.text)[0].inflect({'nomn', 'sing'})
            noun1_nominative = noun1_single_nom.word if noun1_single_nom else token1.text

            phrase_original = f"{token1.text.lower()} {token2.text.lower()}"
            phrase_nominative = f"{noun1_nominative} {token2.text.lower()}"
            key_phrases.append(f"{phrase_original} ({phrase_nominative})")

    # Фильтрация фраз по частоте и проценту общей частоты с учетом резких спадов
    phrase_freq_dist = FreqDist(key_phrases)
    total_phrase_count = sum(phrase_freq_dist.values())
    phrase_limit = total_phrase_count * (phrase_percent / 100)

    filtered_phrases = []
    cumulative_phrase_freq = 0
    drop_count = 0
    prev_phrase_freq = None

    for phrase, freq in phrase_freq_dist.most_common():
        # Условие минимальной частоты
        if freq < min_keyword_freq:
            break
        cumulative_phrase_freq += freq
        # Проверка процентного лимита
        if cumulative_phrase_freq > phrase_limit:
            break

        # Определение резкого спада частоты для фраз
        if prev_phrase_freq is not None and prev_phrase_freq - freq > prev_phrase_freq * 0.5:
            drop_count += 1
            if drop_count > 3:  # Остановка после 3 резких спадов
                break
        prev_phrase_freq = freq

        filtered_phrases.append(phrase)

    # Сохранение фраз в Excel
    save_to_excel(phrase_freq_dist, "phrase_frequencies.xlsx")

    # Построение графиков для слов и фраз
    plot_frequency_graphs(freq_dist, phrase_freq_dist, freq_dist)

    return keywords, filtered_phrases


def save_to_excel(word_freq, filename, output_dir="../output"):
    # Убедимся, что директория существует
    os.makedirs(output_dir, exist_ok=True)

    # Формируем DataFrame из частотного словаря
    data = pd.DataFrame(word_freq.most_common(), columns=["Слово", "Частота"])

    # Путь сохранения файла
    filepath = os.path.join(output_dir, filename)

    # Сохраняем в Excel
    data.to_excel(filepath, index=False, engine="openpyxl")
    print(f"Файл сохранён: {filepath}")



def plot_frequency_graphs(word_freq, phrase_freq_dist, all_word_freq):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Гистограмма частоты ключевых слов
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[word for word, _ in word_freq.most_common(15)],
                y=[freq for _, freq in word_freq.most_common(15)], palette='viridis')
    plt.title("Частота ключевых слов")
    plt.xlabel("Слова")
    plt.ylabel("Частота")
    plt.xticks(rotation=45)
    plt.show()

    # Гистограмма процентного соотношения ключевых слов
    total_words = sum(all_word_freq.values())
    keyword_percent = [freq / total_words * 100 for _, freq in word_freq.most_common(15)]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=[word for word, _ in word_freq.most_common(15)],
                y=keyword_percent, palette='viridis')
    plt.title("Процентное соотношение ключевых слов")
    plt.xlabel("Слова")
    plt.ylabel("Процент от общего числа слов")
    plt.xticks(rotation=45)
    plt.show()

    # Линейный график частот всех слов
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=[word for word, _ in all_word_freq.most_common()],
                 y=[freq for _, freq in all_word_freq.most_common()])
    plt.title("Линейный график частот всех слов")
    plt.xlabel("Слова")
    plt.ylabel("Частота")
    plt.xticks(rotation=45)
    plt.show()

    # Линейный график процентного соотношения всех слов
    all_word_percent = [freq / total_words * 100 for _, freq in all_word_freq.most_common()]
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=[word for word, _ in all_word_freq.most_common()],
                 y=all_word_percent)
    plt.title("Линейный график процентного соотношения всех слов")
    plt.xlabel("Слова")
    plt.ylabel("Процент от общего числа слов")
    plt.xticks(rotation=45)
    plt.show()

    # Список слов и их частот
    # Список слов и их частот
    words, frequencies = zip(*word_freq.most_common())

    # Общая сумма частот
    total_frequency = sum(frequencies)

    # Относительные частоты
    relative_frequencies = [freq / total_frequency for freq in frequencies]

    # Инициализация рангов
    ranks = []
    current_rank = 0.5
    freq_to_rank = {}  # Словарь для рангов частот
    seen_low_freq = False  # Флаг для начала обработки частот <10

    for i, freq in enumerate(frequencies):
        if freq >= 10:
            # Для частот >=10: уникальный ранг
            ranks.append(current_rank)
            freq_to_rank[freq] = current_rank  # Сохраняем ранг для данной частоты
            current_rank += 1
        else:
            # Для частот <10
            if not seen_low_freq:
                # Первый раз обрабатываем частоты <10
                seen_low_freq = True

            # Определяем ID слова, соответствующего значению частоты
            target_index = int(freq) - 1  # ID = частота - 1, так как индексация с 0
            if target_index < len(ranks):
                # Получаем ранг слова с этим ID
                current_rank = ranks[target_index]
            else:
                # Если индекс за пределами списка (ошибка в данных), оставляем текущий ранг
                current_rank = ranks[-1]

            ranks.append(current_rank)

    freq_rank_values = [(freq * (rank - 0.5)) for freq, rank in zip(frequencies, ranks)]

    # Построение графика
    plt.figure(figsize=(14, 7))
    sns.lineplot(x=[word for word, _ in word_freq.most_common()],
                 y=freq_rank_values, marker="o", color="blue")

    # Оформление графика
    plt.title("Линейный график: Частота * Ранг", fontsize=16)
    plt.xlabel("Слова", fontsize=14)
    plt.ylabel("Частота * Ранг", fontsize=14)
    plt.xticks(rotation=45, fontsize=10, ha='right')  # Поворот подписей слов для лучшего отображения
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Вывод таблицы
    print("Таблица рангов:")
    print(f"{'Слово':<15} | {'Частота':<10} | {'Pi':<10} | {'Ранг':<5} | {'Pi*r':<10}")
    for word, freq, rel_freq, rank in zip(words, frequencies, relative_frequencies, ranks):
        print(f"{word:<15} | {freq:<10} | {rel_freq:<10.5f} | {rank:<5.1f} | {rel_freq * rank:.5f}")






""""# Логарифмический график частот и рангов (Zipf)
    ranks = np.arange(1, len(all_word_freq) + 1)
    frequencies = [freq for _, freq in all_word_freq.most_common()]

    plt.figure(figsize=(12, 6))
    plt.plot(ranks, frequencies, marker="o", label="Частота", color="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Логарифмический график частот и рангов (Zipf)")
    plt.xlabel("Логарифм ранга")
    plt.ylabel("Логарифм частоты")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    # Обычный график рангов и частот
    plt.figure(figsize=(12, 6))
    plt.plot(ranks, frequencies, marker="o", label="Частота", color="green")
    plt.title("График частот и рангов")
    plt.xlabel("Ранг")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Логарифмический график рангов и нормализованных частот с "холмом"
    relative_frequencies = np.array(frequencies) / total_words  # Нормализация частот
    log_ranks = np.log(ranks)  # Логарифмы рангов

    # Сглаживание для создания "холма"
    smoothed_frequencies = gaussian_filter1d(relative_frequencies, sigma=2)

    plt.figure(figsize=(12, 6))
    plt.plot(log_ranks, smoothed_frequencies, marker="o", color="purple", label="RPr (Сглаженные частоты)")
    plt.title("Логарифмический график рангов (LnR) и нормализованных частот (RPr) с холмом")
    plt.xlabel("LnR")
    plt.ylabel("RPr")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    # Обычный график рангов и нормализованных частот
    plt.figure(figsize=(12, 6))
    plt.plot(ranks, smoothed_frequencies, marker="o", color="red", label="Нормализованная частота (сглаженные)")
    plt.title("График рангов и нормализованных частот (с холмом)")
    plt.xlabel("Ранг")
    plt.ylabel("Доля частоты")
    plt.grid(True)
    plt.legend()
    plt.show()

    
    # Извлечение частот и рангов
    words = [word for word, _ in word_freq.most_common()]
    frequencies = [freq for _, freq in word_freq.most_common()]
    ranks = np.arange(1, len(frequencies) + 1)

    # Логарифмическое преобразование частот и рангов для визуализации "горки"
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)

    # Сглаживание для выделения "холма"
    smoothed_frequencies = gaussian_filter1d(log_frequencies, sigma=2)

    # Определение границ зон (ядро и периферия)
    total_words = len(ranks)
    core_end = total_words // 3  # Ядро (примерно треть слов)
    right_periphery_start = total_words * 2 // 3  # Правая периферия

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(log_ranks, smoothed_frequencies, label="Частоты (сглаженные)", color="purple")
    plt.axvspan(log_ranks[0], log_ranks[core_end], color="blue", alpha=0.3, label="Ядро")
    plt.axvspan(log_ranks[core_end], log_ranks[right_periphery_start], color="green", alpha=0.3,
                label="Левая периферия")
    plt.axvspan(log_ranks[right_periphery_start], log_ranks[-1], color="orange", alpha=0.3, label="Правая периферия")

    # Настройка подписей
    plt.title("Распределение частот с разделением на зоны (ядро, периферия)")
    plt.xlabel("Ln(Ранг)")
    plt.ylabel("Ln(Частота)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    # Считаем ранги слов
    ranks = np.arange(1, len(word_freq) + 1)  # Ранги
    frequencies = np.array([freq for _, freq in word_freq.most_common()])  # Частоты

    # Нормализуем частоты (RPr)
    total_frequency = frequencies.sum()
    relative_frequencies = frequencies / total_frequency

    # Логарифмы рангов (LnR)
    log_ranks = np.log(ranks)

    # Сглаживание частот для создания "холма"
    smoothed_frequencies = gaussian_filter1d(relative_frequencies, sigma=2)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(log_ranks, smoothed_frequencies, marker="o", color="blue", label="RPr (Сглаженные частоты)")
    plt.title("График LnR (ранги) и RPr (нормализованные частоты)")
    plt.xlabel("Ln(Ранг)")
    plt.ylabel("RPr (Доля частоты)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()"""




"""
def extract_keywords_and_phrases_v3(text, keyword_percent=10, phrase_percent=10):
    # Анализ текста с помощью spacy
    doc = nlp(text)

    # Извлекаем лемматизированные слова, исключая стоп-слова
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

    # Фильтрация только русских слов
    russian_tokens = [word for word in tokens if re.match(r'^[а-яА-ЯёЁ]+$', word)]

    # Частотный словарь
    freq_dist = FreqDist(russian_tokens)

    # Выбор слов по проценту общей частоты
    total_count = sum(freq_dist.values())
    keyword_limit = total_count * (keyword_percent / 100)

    keywords = []
    cumulative_freq = 0
    for word, freq in freq_dist.most_common():
        cumulative_freq += freq
        if cumulative_freq > keyword_limit:
            break
        normal_form = morph.parse(word)[0].normal_form
        keywords.append(f"{word} ({normal_form})")

    # Извлечение фраз
    key_phrases = []
    for i in range(len(doc) - 1):
        token1, token2 = doc[i], doc[i + 1]

        # Паттерн: прилагательное + существительное
        if token1.pos_ == "ADJ" and token2.pos_ == "NOUN":
            noun_lemma = morph.parse(token2.text)[0].normal_form
            noun_single_nom = morph.parse(noun_lemma)[0].inflect({'nomn', 'sing'})
            noun_nominative = noun_single_nom.word if noun_single_nom else token2.text

            adj_form = morph.parse(token1.text)[0]
            gender = noun_single_nom.tag.gender if noun_single_nom else adj_form.tag.gender
            adj_nominative_form = adj_form.inflect({'nomn', 'sing', gender}) if gender else None
            adj_nominative = adj_nominative_form.word if adj_nominative_form else token1.text

            phrase_original = f"{token1.text.lower()} {token2.text.lower()}"
            phrase_nominative = f"{adj_nominative} {noun_nominative}"
            key_phrases.append(f"{phrase_original} ({phrase_nominative})")

        # Паттерн: существительное + существительное
        elif token1.pos_ == "NOUN" and token2.pos_ == "NOUN":
            noun1_single_nom = morph.parse(token1.text)[0].inflect({'nomn', 'sing'})
            noun1_nominative = noun1_single_nom.word if noun1_single_nom else token1.text

            phrase_original = f"{token1.text.lower()} {token2.text.lower()}"
            phrase_nominative = f"{noun1_nominative} {token2.text.lower()}"
            key_phrases.append(f"{phrase_original} ({phrase_nominative})")

    # Фильтрация фраз по проценту общей частоты
    phrase_freq_dist = FreqDist(key_phrases)
    total_phrase_count = sum(phrase_freq_dist.values())
    phrase_limit = total_phrase_count * (phrase_percent / 100)

    filtered_phrases = []
    cumulative_phrase_freq = 0
    for phrase, freq in phrase_freq_dist.most_common():
        cumulative_phrase_freq += freq
        if cumulative_phrase_freq > phrase_limit:
            break
        filtered_phrases.append(phrase)

    # Построение графиков
    plot_frequency_graphs(freq_dist, phrase_freq_dist)

    return keywords, filtered_phrases
"""
"""
def extract_keywords_and_phrases(text, num_keywords=8, num_phrases=5):
    # Анализ текста с помощью spacy
    doc = nlp(text)

    # Извлекаем лемматизированные слова, исключая стоп-слова
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

    # Фильтрация только русских слов
    russian_tokens = [word for word in tokens if re.match(r'^[а-яА-ЯёЁ]+$', word)]

    # Извлечение ключевых слов с нормальными формами
    freq_dist = FreqDist(russian_tokens)
    keywords = []
    for word, freq in freq_dist.most_common(num_keywords):
        # Приводим слово к нормальной форме
        normal_form = morph.parse(word)[0].normal_form
        keywords.append(f"{word} ({normal_form})")

    # Извлечение фраз с корректировкой падежей
    key_phrases = []
    for i in range(len(doc) - 1):
        token1, token2 = doc[i], doc[i + 1]

        # Паттерн: прилагательное + существительное
        if token1.pos_ == "ADJ" and token2.pos_ == "NOUN":
            noun_lemma = morph.parse(token2.text)[0].normal_form
            noun_single_nom = morph.parse(noun_lemma)[0].inflect({'nomn', 'sing'})
            noun_nominative = noun_single_nom.word if noun_single_nom else token2.text

            adj_form = morph.parse(token1.text)[0]
            gender = noun_single_nom.tag.gender if noun_single_nom else adj_form.tag.gender
            adj_nominative_form = adj_form.inflect({'nomn', 'sing', gender}) if gender else None
            adj_nominative = adj_nominative_form.word if adj_nominative_form else token1.text

            phrase_original = f"{token1.text.lower()} {token2.text.lower()}"
            phrase_nominative = f"{adj_nominative} {noun_nominative}"
            key_phrases.append(f"{phrase_original} ({phrase_nominative})")

        # Паттерн: существительное + существительное
        elif token1.pos_ == "NOUN" and token2.pos_ == "NOUN":
            noun1_single_nom = morph.parse(token1.text)[0].inflect({'nomn', 'sing'})
            noun1_nominative = noun1_single_nom.word if noun1_single_nom else token1.text

            phrase_original = f"{token1.text.lower()} {token2.text.lower()}"
            phrase_nominative = f"{noun1_nominative} {token2.text.lower()}"
            key_phrases.append(f"{phrase_original} ({phrase_nominative})")

    # Частотный словарь для фраз
    phrase_freq_dist = FreqDist(key_phrases)
    key_phrases = [phrase for phrase, freq in phrase_freq_dist.most_common(num_phrases)]

    # Построение графиков
    plot_frequency_graphs(freq_dist, phrase_freq_dist)

    return keywords, key_phrases
"""
"""import spacy
import nltk
import re
from nltk import FreqDist
import pymorphy3
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Инициализация pymorphy3 для морфологического анализа
morph = pymorphy3.MorphAnalyzer()

# Загрузка ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Загрузка модели spacy для русского языка
nlp = spacy.load('ru_core_news_sm')


def extract_keywords_and_phrases_v1(text, num_keywords=8, num_phrases=5):
    # Извлечение ключевых слов и фраз по количеству
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    russian_tokens = [word for word in tokens if re.match(r'^[а-яА-ЯёЁ]+$', word)]
    freq_dist = FreqDist(russian_tokens)

    keywords = []
    for word, freq in freq_dist.most_common(num_keywords):
        normal_form = morph.parse(word)[0].normal_form
        keywords.append(f"{word} ({normal_form})")

    key_phrases = []
    for i in range(len(doc) - 1):
        token1, token2 = doc[i], doc[i + 1]
        if token1.pos_ == "ADJ" and token2.pos_ == "NOUN":
            adj_form = morph.parse(token1.text)[0]
            noun_nominative = morph.parse(token2.text)[0].inflect({'nomn', 'sing'}).word
            key_phrases.append(f"{token1.text.lower()} {token2.text.lower()} ({adj_form.word} {noun_nominative})")

    return keywords, key_phrases[:num_phrases]


def extract_keywords_and_phrases_v2(text, min_keyword_freq=3, keyword_percent=1.5, phrase_percent=5):
    # Извлечение ключевых слов и фраз по проценту общей частоты
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    russian_tokens = [word for word in tokens if re.match(r'^[а-яА-ЯёЁ]+$', word)]
    freq_dist = FreqDist(russian_tokens)
    total_count = sum(freq_dist.values())
    keyword_limit = total_count * (keyword_percent / 100)

    keywords = []
    cumulative_freq = 0
    for word, freq in freq_dist.most_common():
        if freq < min_keyword_freq or cumulative_freq > keyword_limit:
            break
        cumulative_freq += freq
        normal_form = morph.parse(word)[0].normal_form
        keywords.append(f"{word} ({normal_form})")

    return keywords, []  # Для краткости опустим извлечение фраз


def extract_keywords_and_phrases_v3(text, keyword_percent=10):
    # Извлечение ключевых слов и фраз на основе резких спадов частоты
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    russian_tokens = [word for word in tokens if re.match(r'^[а-яА-ЯёЁ]+$', word)]
    freq_dist = FreqDist(russian_tokens)
    total_count = sum(freq_dist.values())

    keywords = []
    freq_list = [freq for _, freq in freq_dist.most_common()]
    drop_count = 0
    for idx, (word, freq) in enumerate(freq_dist.most_common()):
        if idx > 0 and freq_list[idx - 1] > freq * 2:
            drop_count += 1
        if drop_count >= 3:
            break
        normal_form = morph.parse(word)[0].normal_form
        keywords.append(f"{word} ({normal_form})")

    return keywords, []  # Для краткости опустим извлечение фраз


def extract_keywords_and_phrases_combined(text):
    # Объединение всех трех методов
    keywords_v1, phrases_v1 = extract_keywords_and_phrases_v1(text)
    keywords_v2, _ = extract_keywords_and_phrases_v2(text)
    keywords_v3, _ = extract_keywords_and_phrases_v3(text)

    # Уникальные ключевые слова и фразы
    combined_keywords = list(set(keywords_v1 + keywords_v2 + keywords_v3))
    combined_phrases = list(set(phrases_v1))

    # Построение графиков
    plot_frequency_graphs(FreqDist(combined_keywords), FreqDist(combined_phrases))

    return combined_keywords, combined_phrases


def plot_frequency_graphs(word_freq, phrase_freq):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[word for word, _ in word_freq.most_common(15)],
                y=[freq for _, freq in word_freq.most_common(15)], palette='viridis')
    plt.title("Частота ключевых слов")
    plt.xlabel("Слова")
    plt.ylabel("Частота")
    plt.xticks(rotation=45)
    plt.show()

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(word_freq))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud ключевых слов")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x=[phrase for phrase, _ in phrase_freq.most_common(10)],
                y=[freq for _, freq in phrase_freq.most_common(10)], palette='plasma')
    plt.title("Частота ключевых фраз")
    plt.xlabel("Фразы")
    plt.ylabel("Частота")
    plt.xticks(rotation=45)
    plt.show()
"""