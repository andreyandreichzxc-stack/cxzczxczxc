# 🔌 Подключение любых API-ключей / Custom Endpoints

Бот поддерживает **любые OpenAI-совместимые API** через поле `endpoint` в слоте ключа.

---

## Формат добавления ключа

### 1. Авто-определение (простой ключ)
```
sk-abc123xyz...
```
Бот определит провайдера по префиксу: `sk-` → OpenAI, `AIzaSy` → Gemini, `Nb` → Mistral.

### 2. Явный провайдер + ключ
```
openai:sk-abc123xyz
gemini:AIzaSyXxx...
mistral:NbXxx...
cloudflare:Xxx... 
openrouter:sk-or-xxx...
```
Две части через двоеточие: `провайдер:ключ`.

### 3. Кастомный endpoint (3 части — НОВОЕ)
```
провайдер:ключ:https://api.твой-сервер.com/v1
```

| Сервис | Команда |
|--------|---------|
| **DeepSeek** | `openai:sk-abc123:https://api.deepseek.com/v1` |
| **OpenGateway** | `openai:ogw_live_xxx:https://opengateway.gitlawb.com/v1` |
| **Ollama (локальный)** | `openai:ollama:http://localhost:11434/v1` |
| **vLLM / TGI** | `openai:token:http://192.168.1.100:8000/v1` |
| **Groq** | `openai:gsk_xxx:https://api.groq.com/openai/v1` |
| **Together AI** | `openai:together_xxx:https://api.together.xyz/v1` |
| **Fireworks** | `openai:fw_xxx:https://api.fireworks.ai/inference/v1` |
| **AnyScale** | `openai:esecret_xxx:https://api.endpoints.anyscale.com/v1` |
| **Любой OpenAI-совместимый** | `openai:key:https://твой-url/v1` |

---

## Команды в боте

| Команда | Что делает |
|---------|-----------|
| `/keys add` | Добавить ключ пошагово (мастер) |
| `/keys import` | Импорт ключей списком (можно несколько строк) |
| `/keys` | Показать все ключи |
| `/keys remove <номер>` | Удалить ключ |

### Пример сессии `/keys import`:
```
/keys import
openai:sk-proj-abc123:https://api.deepseek.com/v1
openai:ogw_live_xyz:https://opengateway.gitlawb.com/v1
openrouter:sk-or-xxx
```

---

## Какие провайдеры доступны

| Провайдер | Префикс ключа | Совместимость | Endpoint по умолчанию |
|-----------|---------------|---------------|----------------------|
| `openai` | `sk-`, `sk-proj-` | OpenAI SDK | `api.openai.com/v1` |
| `gemini` | `AIzaSy` | Google genai SDK | (не сменный) |
| `mistral` | `Nb` | OpenAI SDK | `api.mistral.ai/v1` |
| `cloudflare` | любая строка | OpenAI SDK | `api.cloudflare.com/client/v4` |
| `openrouter` | `sk-or-` | OpenAI SDK | `openrouter.ai/api/v1` |

**Важно:** Gemini использует свой SDK, endpoint для него не сменный. Все остальные — OpenAI-совместимые через `AsyncOpenAI`, поэтому **любой** из них можно перенаправить на кастомный URL.

---

## Как это работает внутри

1. Ключ сохраняется в `LlmKeySlot` с полями: `provider`, `key_enc`, `endpoint`
2. При создании провайдера: если `endpoint` не пустой → пробрасывается как `base_url=endpoint`
3. Все запросы к API идут на указанный URL вместо дефолтного
4. Обратная совместимость: ключи без endpoint работают как раньше

---

## Если API НЕ OpenAI-совместимый

Если твой API использует **другой формат** (не `/v1/chat/completions`):

1. Создай класс-провайдер в `src/llm/` (скопируй `openai_provider.py`)
2. Реализуй методы: `validate_key()`, `chat()`, `embed()`, `embed_batch()`
3. Зарегистрируй в `src/llm/router.py` в `build_provider()`
4. Добавь название в `_known_providers`

**Пример — DeepSeek уже работает через openai-провайдер, ничего писать не надо.** Большинство современных API (Groq, Together, Fireworks, Anyscale, OpenRouter) уже OpenAI-совместимы.

---

## Частые вопросы

**Q: Почему Gemini без кастомного endpoint?**
A: Google использует свой SDK (`google-genai`), а не OpenAI SDK. Если нужен Gemini через прокси — используй OpenRouter (`sk-or-` ключ + endpoint OpenRouter).

**Q: Что если ключ не проходит валидацию?**
A: Бот при сохранении проверяет ключ через `validate_key()` — запрос к `/v1/models`. Если endpoint отвечает нестандартно, валидация может не пройти. Можно добавить ключ через БД вручную.

**Q: Как добавить модель по умолчанию для endpoint?**
A: В `src/config.py` есть поля `openai_chat_light_model`, `openai_chat_heavy_model` — они используются для всех OpenAI-совместимых провайдеров.
