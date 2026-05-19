"""Эвристический анализ стиля общения по текстам сообщений.
Не использует LLM — только регулярные выражения и статистику."""

import re
from collections import Counter

# Мат-словарь (20+ русских матерных слов и производных)
MAT_VOCAB = [
    "хуй",
    "хуя",
    "хуе",
    "хуё",
    "хуи",
    "хуйня",
    "хуйнуть",
    "похуй",
    "нахуй",
    "схуя",
    "пизд",
    "пизда",
    "пиздец",
    "пиздить",
    "распиздяй",
    "запиздярить",
    "пиздатый",
    "ебал",
    "ебать",
    "ебан",
    "ёбан",
    "заеба",
    "наеба",
    "отъеба",
    "выеба",
    "уебищ",
    "блядь",
    "бля",
    "блять",
    "забля",
    "нах",
    "нафиг",
    "аху",
    "оху",
    "охуеть",
    "охуен",
    "ахуеть",
    "мудак",
    "мудил",
    "мудень",
    "гандон",
    "гондон",
    "долбоёб",
    "долбоеб",
    "срать",
    "сру",
    "засра",
    "насра",
    "пидор",
    "пидорас",
    "петух",
    "шлюха",
    "проститутк",
    "хер",
    "херня",
    "похер",
    "жопа",
    "жоп",
    "залуп",
    "залупа",
    "говно",
    "говн",
    "говенный",
    "ссать",
    "ссык",
    "ссыку",
]

# Собираем один большой regex с word boundaries
_MAT_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in MAT_VOCAB) + r")\w*",
    re.IGNORECASE,
)

# Regex для emoji (диапазон Unicode)
_EMOJI_PATTERN = re.compile(
    r"[\U0001F300-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001F600-\U0001F64F"
    r"\u2600-\u27BF"
    r"\uFE00-\uFE0F"
    r"]+",
    re.UNICODE,
)

# CAPS: сообщение где >50% букв заглавные и длина >5 символов
_CAPS_THRESHOLD = 0.5
_CAPS_MIN_LEN = 5

# Пунктуация завершения: заканчивается на . ! ? ) ...
_ENDING_PUNCT_RE = re.compile(r"[.!?)]$|\.\.\.$")

# Сленговый словарь (15+ слов)
SLANG_VOCAB = {
    "крч",
    "кста",
    "кстати",
    "прив",
    "спс",
    "плз",
    "пожалуйста",
    "норм",
    "ок",
    "ok",
    "го",
    "хз",
    "имхо",
    "лол",
    "lol",
    "кек",
    "жиза",
    "рофл",
    "краш",
    "сорян",
    "сори",
    "окей",
    "ага",
    "неа",
    "давай",
    "пон",
    "понял",
    "ясно",
}

# Ласкательные формы
AFFECTIONATE_VOCAB = [
    "солнышко",
    "котёнок",
    "котенок",
    "зайка",
    "зай",
    "дорогой",
    "дорогая",
    "родной",
    "родная",
    "милый",
    "милая",
    "любимый",
    "любимая",
    "малыш",
    "сладкий",
    "сладкая",
    "золотце",
    "зайчик",
    "котеночек",
    "лапка",
    "лапочка",
    "ангелочек",
    "рыбка",
    "киса",
    "пупсик",
    "малышка",
]

# Регистрируем word-boundary regex для сленга и ласкательных
_SLANG_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in SLANG_VOCAB) + r")\b",
    re.IGNORECASE,
)
_AFFECTIONATE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in AFFECTIONATE_VOCAB) + r")\b",
    re.IGNORECASE,
)


def analyze_messages_heuristic(messages_texts: list[str]) -> dict:
    """
    Анализирует список текстов сообщений и возвращает эвристический профиль стиля.

    Возвращает словарь с полями:
        mat_frequency, mat_words, emoji_frequency, top_emoji,
        caps_frequency, avg_msg_length, punctuation_ratio,
        slang_words, affectionate_forms
    """
    total = len(messages_texts)
    if total == 0:
        return {
            "mat_frequency": 0.0,
            "mat_words": [],
            "emoji_frequency": 0.0,
            "top_emoji": [],
            "caps_frequency": 0.0,
            "avg_msg_length": 0.0,
            "punctuation_ratio": 0.0,
            "slang_words": [],
            "affectionate_forms": [],
        }

    mat_count = 0
    mat_words_set: set[str] = set()
    emoji_count = 0
    emoji_counter: Counter[str] = Counter()
    caps_count = 0
    total_length = 0
    punct_count = 0
    slang_set: set[str] = set()
    affection_set: set[str] = set()

    for msg in messages_texts:
        if not msg:
            continue

        # Мат
        mat_found = _MAT_PATTERN.findall(msg)
        if mat_found:
            mat_count += 1
            mat_words_set.update(m.lower() for m in mat_found)

        # Эмодзи
        emoji_found = _EMOJI_PATTERN.findall(msg)
        if emoji_found:
            emoji_count += 1
            for e in emoji_found:
                # Считаем каждый отдельный символ эмодзи
                for ch in e:
                    if ord(ch) > 0x1F000 or ord(ch) in range(0x2600, 0x27C0):
                        emoji_counter[ch] += 1

        # CAPS
        letters = [ch for ch in msg if ch.isalpha()]
        if len(letters) > _CAPS_MIN_LEN:
            upper_count = sum(1 for ch in letters if ch.isupper())
            if upper_count / len(letters) > _CAPS_THRESHOLD:
                caps_count += 1

        # Длина
        total_length += len(msg)

        # Пунктуация
        msg_stripped = msg.strip()
        if msg_stripped and _ENDING_PUNCT_RE.search(msg_stripped):
            punct_count += 1

        # Сленг
        slang_found = _SLANG_PATTERN.findall(msg)
        if slang_found:
            slang_set.update(s.lower() for s in slang_found)

        # Ласкательные
        affection_found = _AFFECTIONATE_PATTERN.findall(msg)
        if affection_found:
            affection_set.update(a.lower() for a in affection_found)

    return {
        "mat_frequency": round(mat_count / total, 4) if total else 0.0,
        "mat_words": sorted(mat_words_set),
        "emoji_frequency": round(emoji_count / total, 4) if total else 0.0,
        "top_emoji": [e for e, _ in emoji_counter.most_common(5)],
        "caps_frequency": round(caps_count / total, 4) if total else 0.0,
        "avg_msg_length": round(total_length / total, 1) if total else 0.0,
        "punctuation_ratio": round(punct_count / total, 4) if total else 0.0,
        "slang_words": sorted(slang_set),
        "affectionate_forms": sorted(affection_set),
    }
