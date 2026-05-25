"""Тесты для модуля humanizer: analyze_ai_score, humanize_text, humanize_response."""

from __future__ import annotations

from src.core.humanizer import analyze_ai_score, humanize_text, humanize_response
from src.core.humanizer.humanizer import _preservation_check


class TestAnalyzeAiScore:
    """Тесты функции analyze_ai_score."""

    def test_analyze_ai_score_clean_text(self):
        """Естественный текст без AI-маркеров — score < 0.2."""
        score, breakdown = analyze_ai_score("привет, как дела? чем занимаешься?")
        assert score < 0.2, f"Ожидался низкий score, получен {score}"
        assert breakdown["markers"] == []
        assert breakdown["patterns"] == []

    def test_analyze_ai_score_ai_text(self):
        """Текст с AI-шаблонами — score > 0.5."""
        text = (
            "Конечно, я понимаю вашу ситуацию! "
            "Безусловно, я всегда рядом и здесь чтобы помочь. "
            "Во-первых, давайте разберёмся. Во-вторых, это совершенно нормально."
        )
        score, breakdown = analyze_ai_score(text)
        assert score > 0.5, f"Ожидался высокий score, получен {score}"
        assert len(breakdown["markers"]) >= 3, (
            f"Ожидалось >=3 маркеров, получено {len(breakdown['markers'])}"
        )

    def test_analyze_ai_score_empty(self):
        """None и пустая строка возвращают (0.02, ...)."""
        score_none, bd_none = analyze_ai_score(None)  # type: ignore[arg-type]
        assert score_none == 0.02
        assert bd_none["markers"] == []

        score_empty, bd_empty = analyze_ai_score("")
        assert score_empty == 0.02
        assert bd_empty["markers"] == []

    def test_analyze_ai_score_whitespace(self):
        """Строка из пробелов тоже обрабатывается."""
        score, breakdown = analyze_ai_score("   \t\n  ")
        # whitespace-only — текст очень короткий, получит небольшой штраф
        assert 0 <= score <= 1.0


class TestHumanizeText:
    """Тесты функции humanize_text."""

    def test_humanize_text_removes_markers(self):
        """Удаляет AI-маркеры из текста."""
        text = "Я понимаю ваши чувства, это совершенно нормально. Во-первых, давайте поговорим."
        result = humanize_text(text)
        # Маркеры должны быть удалены или заменены
        assert "Я понимаю" not in result.lower() or "понимаю твою" in result.lower()
        assert "во-первых" not in result.lower()
        assert "давайте" not in result.lower()
        assert "я понимаю вашу" not in result.lower()
        assert "это совершенно нормально" not in result.lower()

    def test_humanize_text_preserves_content(self):
        """Обычные слова не меняются."""
        text = "сегодня отличная погода, пойдём гулять в парк"
        result = humanize_text(text)
        assert "отличная погода" in result
        assert "гулять" in result
        assert "парк" in result

    def test_humanize_text_case_insensitive(self):
        """Замена работает без учёта регистра."""
        text = "КОНЕЧНО, я понимаю. Разумеется, это так."
        result = humanize_text(text)
        assert "конечно" not in result.lower()
        assert "разумеется" not in result.lower()

    def test_humanize_text_empty(self):
        """Пустая строка возвращается как есть."""
        assert humanize_text("") == ""
        assert humanize_text(None) == ""  # type: ignore[arg-type]

    def test_humanize_text_collapses_spaces(self):
        """Множественные пробелы схлопываются."""
        text = "привет.   удалены.   маркеры."
        result = humanize_text(text)
        assert "   " not in result


class TestHumanizeResponse:
    """Тесты функции humanize_response."""

    def test_humanize_response_removes_cliches(self):
        """Удаляет шаблонные концовки из ответа."""
        text = "Вот что я нашёл. Рад помочь! Если что — обращайся!"
        result = humanize_response(text)
        assert "рад помочь" not in result.lower()
        assert "если что" not in result.lower()
        assert "обращайся" not in result.lower()
        assert "нашёл" in result.lower()

    def test_humanize_response_removes_cliche_obrashaysya(self):
        """Удаляет 'обращайся' из концовки."""
        text = "Готово, если будут вопросы — обращайся"
        result = humanize_response(text)
        assert "обращайся" not in result.lower()

    def test_humanize_response_context_hint(self):
        """Добавляет контекстную фразу для recipe."""
        text = "Вот рецепт борща: свёкла, капуста, мясо."
        result = humanize_response(text, context_hint="recipe")
        assert "приятного аппетита" in result.lower()

    def test_humanize_response_short_clean(self):
        """Короткий естественный текст возвращается без изменений."""
        text = "ок, спасибо"
        result = humanize_response(text)
        assert result == text

    def test_humanize_response_empty(self):
        """Пустой текст — пустой результат."""
        assert humanize_response("") == ""
        assert humanize_response(None) == ""  # type: ignore[arg-type]

    def test_humanize_response_news_context(self):
        """Контекстная фраза для news."""
        # Нужен текст длиннее 30 символов или с AI-паттернами
        result = humanize_response(
            "Вот сводка новостей за сегодня: всё спокойно, ничего важного",
            context_hint="news",
        )
        assert "буду держать в курсе" in result.lower()

    def test_humanize_response_memory_context(self):
        """Контекстная фраза для memory."""
        result = humanize_response(
            "Я запомнил этот важный факт на будущее",
            context_hint="memory",
        )
        assert "запомнил" in result.lower()

    def test_humanize_response_unknown_context(self):
        """Неизвестный context_hint не добавляет фразу."""
        result = humanize_response("какой-то текст", context_hint="unknown_hint")
        assert result == "какой-то текст"

    def test_humanize_response_no_emoji_style(self):
        """Стиль 'без эмодзи' убирает эмодзи из контекстной фразы."""
        result = humanize_response(
            "Вот рецепт борща со свёклой и капустой",
            context_hint="recipe",
            style_profile="без эмодзи",
        )
        assert "🍲" not in result
        assert "приятного аппетита" in result.lower()


class TestHumanizeResponseNoTouch:
    """Regression: контент, который humanize_response НЕ должен трогать."""

    def test_preserves_json(self):
        """JSON-строка с context_hint не получает tail."""
        text = '{"key": "value"}'
        result = humanize_response(text, context_hint="recipe")
        assert result == text

    def test_preserves_code_block(self):
        """Кодовый блок с context_hint не получает tail."""
        text = "```\nprint('hello')\n```"
        result = humanize_response(text, context_hint="recipe")
        assert result == text

    def test_no_tail_on_short_answer(self):
        """Короткий ответ (<30 символов) без контекста возвращается как есть."""
        text = "Привет!"
        result = humanize_response(text)
        assert result == text

    def test_no_tail_on_send(self):
        """Короткий ответ с context_hint='send' не получает tail."""
        text = "Отправляю"
        result = humanize_response(text, context_hint="send")
        assert result == text


class TestHumanizeResponseTails:
    """Regression: контекстные tail-фразы добавляются для длинных ответов."""

    def test_tail_on_news(self):
        """news → хвост 'буду держать в курсе 📰'."""
        result = humanize_response(
            "Вот сводка новостей за сегодня: всё спокойно, "
            "ничего важного не произошло, все события штатно развиваются",
            context_hint="news",
        )
        assert "буду держать в курсе" in result.lower()

    def test_tail_on_recipe(self):
        """recipe → хвост 'приятного аппетита! 🍲'."""
        result = humanize_response(
            "Вот рецепт борща: свёкла, капуста, мясо, картошка, "
            "морковка, лук, томатная паста и сметана",
            context_hint="recipe",
        )
        assert "приятного аппетита" in result.lower()

    def test_tail_on_memory(self):
        """memory → хвост 'запомнил, не забуду 🧠'."""
        result = humanize_response(
            "Я запомнил этот важный факт на будущее, он пригодится в работе",
            context_hint="memory",
        )
        assert "запомнил, не забуду" in result.lower()

    def test_tail_on_search(self):
        """search → хвост 'если нужно копнуть глубже — скажи 🔍'."""
        result = humanize_response(
            "Вот результаты поиска по вашему запросу: "
            "найдено несколько подходящих вариантов",
            context_hint="search",
        )
        assert "глубже" in result.lower()


class TestHumanizeResponseClicheRemoval:
    """Regression: удаление шаблонных концовок (_CLICHÉ_ENDINGS)."""

    def test_removes_single_cliche(self):
        """Одиночное клише в конце удаляется."""
        text = "Ответ готов. Если что пиши."
        result = humanize_response(text)
        assert "если что пиши" not in result.lower()
        assert "Ответ готов" in result

    def test_removes_chain_cliches(self):
        """Цепочка из нескольких клише в конце удаляется целиком.

        Все три клише ('рад помочь', 'если будут вопросы', 'обращайся')
        последовательно удаляются за 3 прохода цикла.
        """
        text = "Рад помочь. Если будут вопросы — обращайся!"
        result = humanize_response(text)
        assert "обращайся" not in result.lower()
        assert "если будут вопросы" not in result.lower()

    def test_preserves_mid_sentence_cliche(self):
        """Клише в середине предложения НЕ удаляется.

        'рад помочь' в середине — это часть обычного текста,
        паттерн ищет только концовки на конце строки.
        """
        text = "Я рад помочь тебе с проектом."
        result = humanize_response(text)
        assert "рад помочь" in result.lower()


class TestPreservationCheck:
    """Regression: _preservation_check не даёт humanize_deep потерять данные."""

    def test_preserves_url(self):
        """URL (https://...) не теряется."""
        original = "Подробнее на сайте https://example.com/page"
        result = _preservation_check(original, original)
        assert result == original

    def test_preserves_date(self):
        """Дата в формате '15 мая' не теряется."""
        original = "Встреча 15 мая в 14:00"
        result = _preservation_check(original, original)
        assert result == original

    def test_preserves_mention(self):
        """@mention не теряется."""
        original = "Спроси у @username про задачу"
        result = _preservation_check(original, original)
        assert result == original
