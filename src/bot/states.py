from aiogram.fsm.state import State, StatesGroup


class LoginStates(StatesGroup):
    api_id = State()
    api_hash = State()
    phone = State()
    code = State()
    password_2fa = State()


class SettingsStates(StatesGroup):
    waiting_openai_key = State()
    waiting_gemini_key = State()
    waiting_mistral_key = State()
    waiting_digest_time = State()
    waiting_news_time = State()
    waiting_lead_hours = State()
    waiting_timezone = State()
    waiting_auto_reply_text = State()
    waiting_sync_interval = State()


class NewsTopicStates(StatesGroup):
    waiting_topic = State()


class DraftStates(StatesGroup):
    waiting_edit = State()
