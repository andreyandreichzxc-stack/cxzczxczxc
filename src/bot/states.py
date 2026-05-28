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
    waiting_cloudflare_key = State()
    waiting_digest_time = State()
    waiting_news_time = State()
    waiting_lead_hours = State()
    waiting_timezone = State()
    waiting_auto_reply_text = State()
    waiting_sync_interval = State()
    waiting_quiet_hours_start = State()
    waiting_quiet_hours_end = State()
    waiting_import_keys = State()
    waiting_custom_instructions = State()
    waiting_alias = State()
    waiting_deepseek_key = State()
    waiting_custom_model_name = State()


class NewsTopicStates(StatesGroup):
    waiting_topic = State()


class DraftStates(StatesGroup):
    waiting_edit = State()


class OnboardingStates(StatesGroup):
    waiting_start = State()
    waiting_login = State()
    waiting_llm_key = State()
    waiting_timezone = State()
    waiting_sync_choice = State()
