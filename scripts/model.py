from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Загрузка модели и токенайзера
model = AutoModelForSequenceClassification.from_pretrained("symanto/xlm-roberta-base-snli-mnli-anli-xnli")
tokenizer = AutoTokenizer.from_pretrained("symanto/xlm-roberta-base-snli-mnli-anli-xnli")

# Входное предложение для классификации
input_sentence = "Область науки и философии стремится понять основные истины о нашем существовании и знаниях."

# Коды УДК первого уровня с их описаниями
udc_first_level = {
    "0": "Общие вопросы. Наука и знание. Организация. Информация. Документация. Библиотечное дело. Учреждения. Публикации.",
    "1": "Философия вцелом",
    "2": "Религия. Теология",
    "3": "Науки",
    "4": "Не используется",
    "5": "Математика и естественные науки",
    "6": "Прикладные науки. Медицина. Техника",
    "7": "Искусство. Развлечения. Спорт",
    "8": "Язык. Лингвистика. Литература",
    "9": "География. Биография. История",
    "10": "Психология о важном"
}

# Создание пар (входное предложение, описание УДК-кода)
input_pairs = [(input_sentence, description) for description in udc_first_level.values()]

# Токенизация и подготовка данных
inputs = tokenizer(input_pairs, truncation="only_first", return_tensors="pt", padding=True)
logits = model(**inputs).logits

# Расчет вероятностей для каждой категории
probs = torch.softmax(logits, dim=1)
positive_probs = probs[..., [0]].tolist()  # Вероятности первой колонки (positive class)

# Сопоставление вероятностей с кодами УДК
udc_probabilities = {code: prob[0] for code, prob in zip(udc_first_level.keys(), positive_probs)}

# Вывод результата: сортировка по вероятности
sorted_udc = sorted(udc_probabilities.items(), key=lambda item: item[1], reverse=True)

# Печать результатов классификации
print("Классификация по УДК (первый уровень):")
for code, prob in sorted_udc:
    print(f"Код УДК: {code}, вероятность: {prob:.2f}, описание: {udc_first_level[code]}")

# Вывод наибольших вероятных кодов в формате код:код:...
most_probable_codes = ":".join([code for code, _ in sorted_udc[:3]])  # Например, топ-3
print(f"\nКод УДК для текста: {most_probable_codes}")
