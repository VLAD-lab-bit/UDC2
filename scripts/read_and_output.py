import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import read_files_from_folder
from extract_keywords import extract_keywords_and_phrases_v2   # Импортируем новый вариант функции"
from udc_classifier import classify_text_by_udc

def main():
    folder_path = '../data/articles'       # Папка с файлами
    output_folder = '../output'            # Папка для результатов
    full_output_file = os.path.join(output_folder, 'full_output.txt')  # Файл с полными данными
    summary_output_file = os.path.join(output_folder, 'summary_output.txt')  # Файл с кратким отчетом

    # Создаем папку для выходных данных, если ее нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Загружаем модель и токенайзер
    print("Загрузка модели...")
    tokenizer = AutoTokenizer.from_pretrained("symanto/xlm-roberta-base-snli-mnli-anli-xnli")
    model = AutoModelForSequenceClassification.from_pretrained("symanto/xlm-roberta-base-snli-mnli-anli-xnli")

    # Читаем все файлы из указанной папки
    texts = read_files_from_folder(folder_path)

    # Открываем оба файла для записи результатов
    with open(full_output_file, 'w', encoding='utf-8') as full_file, open(summary_output_file, 'w', encoding='utf-8') as summary_file:
        for filename, text in texts.items():
            # Извлекаем ключевые слова и фразы из текста, используя автоматические критерии
            keywords, key_phrases = extract_keywords_and_phrases_v2(
                text,
                min_keyword_freq=2,     # Минимальная частота для ключевых слов и фраз
                keyword_percent=7.5,     # Процент верхних ключевых слов по общей частоте
                phrase_percent=7.5       # Процент верхних фраз по общей частоте
            )

            # Классифицируем текст по УДК на основе ключевых слов и модели
            codes_output, descriptions_output = classify_text_by_udc(text, keywords, key_phrases, tokenizer, model)

            # --- Запись в полный файл ---
            full_file.write(f'=== Файл: {filename} ===\n\n')
            full_file.write(text)
            full_file.write('\n\n')
            full_file.write(f'Ключевые слова: {", ".join(keywords)}\n')
            full_file.write(f'Ключевые фразы: {", ".join(key_phrases)}\n')
            full_file.write(f'Коды УДК: {codes_output}\n')
            full_file.write(f'Описание: {descriptions_output}\n')
            full_file.write('\n' + '=' * 50 + '\n\n')

            # --- Запись в файл краткого отчета ---
            summary_file.write(f'Файл: {filename}\n')
            summary_file.write(f'Ключевые слова: {", ".join(keywords)}\n')
            summary_file.write(f'Ключевые фразы: {", ".join(key_phrases)}\n')
            summary_file.write(f'Коды УДК: {codes_output}\n')
            summary_file.write('Подробности по УДК:\n')
            summary_file.write(f'{descriptions_output}\n')
            summary_file.write('\n' + '-' * 30 + '\n\n')

if __name__ == "__main__":
    main()
