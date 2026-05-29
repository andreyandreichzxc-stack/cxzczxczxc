"""NL-парсер отложенных сообщений."""

import re
from datetime import datetime, timedelta, timezone


def parse_schedule_message(text: str, tz_name: str | None = None) -> dict | None:
    """Парсит 'напомни Маше про встречу завтра в 10:00' → {contact, text, send_at}.

    Args:
        text: сырой текст сообщения
        tz_name: IANA-имя часового пояса (напр. 'Europe/Moscow').
                 Если передан — send_at рассчитывается в локальном времени пользователя.
    """
    # Pre-check: быстрый выход без тяжёлого regex
    if not any(kw in text.lower() for kw in ["напомни", "отправь", "напиши"]):
        return None

    patterns = [
        # Pattern 2: "напомни \"Маша Иванова\" про встречу завтра в 10:00"
        #   groups: (контакт в кавычках)(текст)(день)(HH:MM?)
        #   Самый специфичный — пробуем первым, чтобы не захватило открывающую кавычку как часть слова.
        (
            2,
            r'(?:напомни|отправь|напиши)\s+"([^"]+)"\s+(?:про\s+)?(.+?)\s+(завтра|послезавтра|сегодня|через\s+\d+\s+(?:час|минут|день|дня|дней))\s*(?:в\s+(\d{1,2}:\d{2}))?',
        ),
        # Pattern 0: "напомни Маше про встречу завтра в 10:00"
        #   groups: (контакт)(текст)(день)(HH:MM?)
        (
            0,
            r"(?:напомни|отправь|напиши)\s+(\S+?)\s+(?:про\s+)?(.+?)\s+(завтра|послезавтра|через\s+\d+\s+(?:час|минут|день|дня|дней))\s*(?:в\s+(\d{1,2}:\d{2}))?",
        ),
        # Pattern 1: "напомни Маше про встречу в 10:00 завтра"
        #   groups: (контакт)(текст)(HH:MM)(день)
        (
            1,
            r"(?:напомни|отправь|напиши)\s+(\S+?)\s+(.+?)\s+в\s+(\d{1,2}:\d{2})\s+(завтра|послезавтра|сегодня)",
        ),
    ]

    for pattern_idx, pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            contact = m.group(1)
            msg_text = m.group(2).strip()

            # Группы переставлены в зависимости от паттерна
            if pattern_idx in (0, 2):
                # Pattern 0/2: group(3)=день, group(4)=HH:MM (опционально)
                time_str = m.group(3)
                hour_min = m.group(4) if m.lastindex and m.lastindex >= 4 else None
            else:
                # Pattern 1: group(3)=HH:MM, group(4)=день
                hour_min = m.group(3)
                time_str = m.group(4) if m.lastindex and m.lastindex >= 4 else None

            # Резолвим время — учитываем часовой пояс пользователя
            if tz_name:
                import zoneinfo  # Python 3.9+

                try:
                    user_tz = zoneinfo.ZoneInfo(tz_name)
                    now = datetime.now(user_tz)
                except (zoneinfo.ZoneInfoNotFoundError, KeyError, ValueError):
                    now = datetime.now(timezone.utc)
            else:
                now = datetime.now(timezone.utc)
            send_at = now + timedelta(hours=1)  # default: через час

            # Парсим день
            if time_str and "завтра" in time_str.lower():
                send_at = now.replace(hour=9, minute=0) + timedelta(days=1)
            elif time_str and "послезавтра" in time_str.lower():
                send_at = now.replace(hour=9, minute=0) + timedelta(days=2)
            elif time_str and "сегодня" in time_str.lower():
                send_at = now.replace(hour=9, minute=0)
            elif time_str and "через" in time_str.lower():
                nums = re.findall(r"\d+", time_str)
                if nums:
                    n = int(nums[0])
                    if "минут" in time_str:
                        send_at = now + timedelta(minutes=n)
                    elif "час" in time_str:
                        send_at = now + timedelta(hours=n)
                    elif "день" in time_str or "дня" in time_str or "дней" in time_str:
                        send_at = now + timedelta(days=n)

            # Применяем час:минуты если указаны
            if hour_min:
                h, m = map(int, hour_min.split(":"))
                send_at = send_at.replace(hour=h, minute=m, second=0, microsecond=0)

            return {"contact": contact, "text": msg_text, "send_at": send_at}

    return None
