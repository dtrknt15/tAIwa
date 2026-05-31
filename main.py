import gradio as nn_gradio  # Rename to avoid name shadowing
import openai
import json
import time
import os

# --- Detect HF Spaces Environment or local env keys ---
IS_HF_SPACE = (os.getenv("SYSTEM") == "spaces") or (os.getenv("SPACE_ID") is not None)
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN", "")
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

if IS_HF_SPACE:
    # If in HF Space, default to HF Serverless Inference with Gemma model
    DEFAULT_API_KEY = DEFAULT_HF_TOKEN
    DEFAULT_BASE_URL = "https://api-inference.huggingface.co/v1"
    DEFAULT_MODEL = "google/gemma-4-E2B-it"
else:
    # Default local configuration
    DEFAULT_API_KEY = DEFAULT_OPENAI_KEY
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "手動入力(gpt-4o-miniなど)"

# --- Premium CSS Styling ---
THEME_CSS = """
/* Typography & Reset */
body, html {
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 24px !important;
    transition: background-color 0.3s ease, color 0.3s ease;
}

/* --- Light Mode Colors & Styles --- */
:root, .gradio-container {
    --body-background-fill: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: rgba(0, 0, 0, 0.08) !important;
    --block-border-width: 1px !important;
    --block-title-text-color: #4f46e5 !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: rgba(0, 0, 0, 0.1) !important;
    --input-text-color: #0f172a !important;
    --body-text-color: #0f172a !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --border-color-primary: rgba(0, 0, 0, 0.08) !important;
    --button-secondary-background-fill: #f1f5f9 !important;
    --button-secondary-text-color: #0f172a !important;
    --button-secondary-border-color: rgba(0, 0, 0, 0.1) !important;
    --slider-color: #4f46e5 !important;
}

body, html {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

.gradio-container {
    background-color: #f8fafc !important;
}

/* Glassmorphism panels in Light Mode */
.glass-panel {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(148, 163, 184, 0.1) !important;
}

/* Chat bubble styling for Light Mode */
.chat-bubble {
    padding: 14px 18px;
    border-radius: 12px;
    margin: 8px 0;
    line-height: 1.6;
    font-size: 0.98rem;
    border-left: 5px solid;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(148, 163, 184, 0.08);
}
.chat-bubble:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(148, 163, 184, 0.15);
}

.bubble-persona-0 {
    background: rgba(99, 102, 241, 0.08) !important;
    border-color: #4f46e5 !important;
    color: #1e1b4b !important;
}
.bubble-persona-1 {
    background: rgba(236, 72, 153, 0.08) !important;
    border-color: #db2777 !important;
    color: #500724 !important;
}
.bubble-persona-2 {
    background: rgba(16, 185, 129, 0.08) !important;
    border-color: #059669 !important;
    color: #064e3b !important;
}
.bubble-persona-3 {
    background: rgba(245, 158, 11, 0.08) !important;
    border-color: #d97706 !important;
    color: #451a03 !important;
}

.persona-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: bold;
    text-transform: uppercase;
    margin-left: 8px;
    background: rgba(0, 0, 0, 0.06);
}

/* --- Dark Mode Colors & Styles --- */
.dark, .dark .gradio-container {
    --body-background-fill: #0f111a !important;
    --block-background-fill: #161c2d !important;
    --block-border-color: rgba(255, 255, 255, 0.08) !important;
    --block-border-width: 1px !important;
    --block-title-text-color: #a5b4fc !important;
    --input-background-fill: #0f111a !important;
    --input-border-color: rgba(255, 255, 255, 0.1) !important;
    --input-text-color: #e2e8f0 !important;
    --body-text-color: #e2e8f0 !important;
    --background-fill-primary: #161c2d !important;
    --background-fill-secondary: #0f111a !important;
    --border-color-primary: rgba(255, 255, 255, 0.08) !important;
    --button-secondary-background-fill: #1e293b !important;
    --button-secondary-text-color: #e2e8f0 !important;
    --button-secondary-border-color: rgba(255, 255, 255, 0.1) !important;
    --slider-color: #6366f1 !important;
}

.dark body, .dark html {
    background-color: #0f111a !important;
    color: #e2e8f0 !important;
}

.dark .gradio-container {
    background-color: #0f111a !important;
}

/* Glassmorphism panels in Dark Mode */
.dark .glass-panel {
    background: rgba(22, 28, 45, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

/* Chat bubble styling for Dark Mode */
.dark .chat-bubble {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.dark .chat-bubble:hover {
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
}

.dark .bubble-persona-0 {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: #6366f1 !important;
    color: #e0e7ff !important;
}
.dark .bubble-persona-1 {
    background: rgba(236, 72, 153, 0.15) !important;
    border-color: #ec4899 !important;
    color: #fce7f3 !important;
}
.dark .bubble-persona-2 {
    background: rgba(16, 185, 129, 0.15) !important;
    border-color: #10b981 !important;
    color: #d1fae5 !important;
}
.dark .bubble-persona-3 {
    background: rgba(245, 158, 11, 0.15) !important;
    border-color: #f59e0b !important;
    color: #fef3c7 !important;
}

.dark .persona-badge {
    background: rgba(255, 255, 255, 0.15);
}

/* Buttons and accents */
.start-btn {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.start-btn:hover {
    filter: brightness(1.1) !important;
}

/* Align items to bottom in a Row */
.align-end {
    align-items: flex-end !important;
}

/* Refresh Button styling */
.refresh-btn {
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    font-size: 1.25rem !important;
    background: #f1f5f9 !important;
    border: 1px solid rgba(0, 0, 0, 0.1) !important;
    color: #4f46e5 !important;
    transition: all 0.2s ease !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    margin-bottom: 0px !important;
}

.refresh-btn:hover {
    background: #e2e8f0 !important;
    color: #4f46e5 !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
    transform: translateY(-1px);
}

.dark .refresh-btn {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #a5b4fc !important;
}

.dark .refresh-btn:hover {
    background: #334155 !important;
    color: #a5b4fc !important;
    border-color: rgba(165, 180, 252, 0.3) !important;
}
"""

HEAD_HTML = '<meta name="google" content="notranslate">'

# --- Localization & Translation Setup ---
i18n = nn_gradio.I18n(
    en={
        "app_title": "# 🌐 tAIwa - Multi-Persona Chatroom",
        "app_desc": "A tool where multiple AI agents (and you) can converse freely.",
        "api_config": "### ⚙️ 1. OpenAI-compatible API Configuration",
        "api_key": "API Key",
        "base_url": "Base URL",
        "model_name": "Model Name",
        "persona_config": "### 👥 2. Persona Settings",
        "num_personas": "Number of Personas",
        "persona_title_1": "#### Ichinohe",
        "persona_title_2": "#### Ninomiya",
        "persona_title_3": "#### Miura",
        "persona_title_4": "#### Shinomiya",
        "name_label": "Name",
        "human_label": "You (Human)",
        "system_prompt_label": "System Prompt / Description",
        "system_prompt_default": "You are an expert with critical thinking. Speak in a concise and conversational tone.",
        "persona_default_name_1": "Ichinohe",
        "persona_default_name_2": "Ninomiya",
        "persona_default_name_3": "Miura",
        "persona_default_name_4": "Shinomiya",
        "chatroom_panel": "### 💬 Chatroom Panel",
        "agenda_label": "Agenda / Topic",
        "agenda_placeholder": "Enter the discussion topic...",
        "agenda_default": "What is the biggest challenge in deploying local AI models?",
        "btn_start": "🚀 Start",
        "btn_next": "➡️ Next Turn",
        "btn_reset": "🔄 Reset",
        "human_resp_label": "Your Response",
        "human_resp_placeholder": "Enter your response...",
        "btn_send": "Send"
    },
    ja={
        "app_title": "# 🌐 tAIwa - マルチペルソナ チャットルーム",
        "app_desc": "複数のAIエージェント（とあなた）が自由に対話できるツールです。",
        "api_config": "### ⚙️ 1. OpenAI互換API設定",
        "api_key": "APIキー",
        "base_url": "ベースURL",
        "model_name": "モデル名",
        "persona_config": "### 👥 2. ペルソナ設定",
        "num_personas": "ペルソナの数",
        "persona_title_1": "#### 一戸",
        "persona_title_2": "#### 二宮",
        "persona_title_3": "#### 三浦",
        "persona_title_4": "#### 四宮",
        "name_label": "名前",
        "human_label": "あなた (Human)",
        "system_prompt_label": "システムプロンプト / 説明",
        "system_prompt_default": "あなたは批判的思考を持つ専門家です。簡潔かつ会話的なトーンで話してください。",
        "persona_default_name_1": "一戸",
        "persona_default_name_2": "二宮",
        "persona_default_name_3": "三浦",
        "persona_default_name_4": "四宮",
        "chatroom_panel": "### 💬 チャットルーム パネル",
        "agenda_label": "アジェンダ / トピック",
        "agenda_placeholder": "ディスカッショントピックを入力してください...",
        "agenda_default": "ローカルAIモデルをデプロイする上での最大の課題は何ですか？",
        "btn_start": "🚀 開始する",
        "btn_next": "➡️ 次のターン",
        "btn_reset": "🔄 リセット",
        "human_resp_label": "あなたの返答",
        "human_resp_placeholder": "返答を入力してください...",
        "btn_send": "送信"
    }
)

def get_language(request: nn_gradio.Request = None) -> str:
    """Helper to detect language from request headers."""
    if not request:
        return "en"
    accept_lang = request.headers.get("accept-language", "en")
    if "ja" in accept_lang.lower():
        return "ja"
    return "en"

LOCALIZED_STRINGS = {
    "en": {
        "setup_failed_key": "⚠️ **Setup failed**: API Key is required.",
        "agenda_prefix": "📌 **Agenda/Topic**: {agenda}",
        "setup_instruction": "Configure settings on the left, then click **🚀 Start**.",
        "empty_chat_log": "<div style='text-align: center; color: #64748b; padding: 40px 0;'>Chat log is empty. Set agenda and start to begin!</div>",
        "turn_human": "👉 Turn: **{name}** (You). Please enter your message below.",
        "turn_ai": "⚡ Turn: **{name}** (AI). Please wait for AI to generate a response...",
        "turn_start": "Configure personas to start.",
        "api_key_missing": "⚠️ System error: API Key is missing. Please set it in the config.",
        "api_error": "⚠️ API Error: {error}",
        "roleplay_system_prompt": (
            "{persona_prompt}\n\n"
            "You are roleplaying as {persona_name} in a collaborative group conversation.\n"
            "The topic of conversation is: '{agenda}'\n"
            "The participants in this discussion are: {participant_names}."
        ),
        "roleplay_full_prompt": (
            "Here is the dialogue history so far:\n"
            "--- BEGIN DIALOGUE ---\n"
            "{history_text}\n"
            "--- END DIALOGUE ---\n\n"
            "It is your turn to speak. Write your next response as {persona_name}.\n"
            "Directly contribute to the discussion by reacting to what others have said or posing new viewpoints.\n"
            "Keep your response natural, short (1-3 paragraphs max), and do not prefix it with your name."
        )
    },
    "ja": {
        "setup_failed_key": "⚠️ **セットアップ失敗**: APIキーが必要です。",
        "agenda_prefix": "📌 **アジェンダ/トピック**: {agenda}",
        "setup_instruction": "左側の設定を行ってから、**🚀 開始する** をクリックしてください。",
        "empty_chat_log": "<div style='text-align: center; color: #64748b; padding: 40px 0;'>チャットログは空です。アジェンダを設定して開始してください！</div>",
        "turn_human": "👉 ターン: **{name}** (あなた)。以下にメッセージを入力してください。",
        "turn_ai": "⚡ ターン: **{name}** (AI)。AIの応答生成をお待ちください...",
        "turn_start": "ペルソナを設定して開始してください。",
        "api_key_missing": "⚠️ システムエラー: APIキーがありません。設定に入力してください。",
        "api_error": "⚠️ APIエラー: {error}",
        "roleplay_system_prompt": (
            "{persona_prompt}\n\n"
            "あなたは共同グループ会話で {persona_name} としてロールプレイをしています。\n"
            "会話のトピックは「{agenda}」です。\n"
            "このディスカッションの参加者は {participant_names} です。\n"
            "必ず日本語で自然に会話してください。"
        ),
        "roleplay_full_prompt": (
            "これまでの対話履歴は以下の通りです：\n"
            "--- 対話開始 ---\n"
            "{history_text}\n"
            "--- 対話終了 ---\n\n"
            "あなたの発言順です。{persona_name} として次の応答を書いてください。\n"
            "他の参加者の発言に反応したり、新しい視点を提示したりして、ディスカッションに直接貢献してください。\n"
            "応答は自然で短く（最大1〜3段落）保ち、冒頭に自分の名前を付けないでください。\n"
            "必ず日本語で自然に会話してください。"
        )
    }
}

def format_local_base_url(base_url):
    """Ensures local base URLs (like localhost:1234) have the /v1 suffix appended automatically."""
    if not base_url:
        return base_url
    base_url_stripped = base_url.rstrip("/")
    if any(loc in base_url_stripped.lower() for loc in ["localhost", "127.0.0.1", "::1"]):
        if not base_url_stripped.endswith("/v1") and not base_url_stripped.endswith("/v1/"):
            return base_url_stripped + "/v1"
    return base_url_stripped

def generate_ai_response(api_key, base_url, model, system_prompt, full_prompt):
    """Calls OpenAI-compatible API to generate a response."""
    base_url = format_local_base_url(base_url)
    print(f"\n[AI Request] Connecting to API...")
    print(f"  Base URL: {base_url}")
    print(f"  Model: {model}")
    try:
        if not api_key:
            is_local = any(loc in (base_url or "").lower() for loc in ["localhost", "127.0.0.1", "::1"])
            if is_local:
                api_key = "lm-studio"
            else:
                return "⚠️ System error: API Key is missing. Please set it in the config."
        
        client = openai.OpenAI(
            api_key=api_key, 
            base_url=base_url if base_url else None,
            timeout=60.0
        )
        
        print(f"  Sending chat completion request...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=100000
        )
        print(f"  Response received successfully.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [API Error] {e}")
        return f"⚠️ API Error: {str(e)}"

def format_history_for_prompt(chat_log):
    """Formats the accumulated chat log for context in the LLM prompt."""
    formatted_lines = []
    for speaker, text in chat_log:
        formatted_lines.append(f"{speaker}: {text}")
    return "\n".join(formatted_lines)

def update_ui_state(lang, current_idx, participants_list):
    """Determines helper messages and handles human vs AI turn UI updates."""
    if not participants_list:
        return LOCALIZED_STRINGS[lang]["turn_start"], False, ""
    
    current_persona = participants_list[current_idx]
    name = current_persona["name"]
    is_human = current_persona["is_human"]
    
    if is_human:
        status_msg = LOCALIZED_STRINGS[lang]["turn_human"].format(name=name)
        return status_msg, True, name
    else:
        status_msg = LOCALIZED_STRINGS[lang]["turn_ai"].format(name=name)
        return status_msg, False, ""

def update_models_dropdown(api_key, base_url, current_model):
    """Fetches the list of models from the OpenAI-compatible API and updates the dropdown choices."""
    base_url = format_local_base_url(base_url)
    if not api_key:
        is_local = any(loc in (base_url or "").lower() for loc in ["localhost", "127.0.0.1", "::1"])
        if is_local:
            api_key = "lm-studio"
        else:
            choices = [current_model] if current_model else [DEFAULT_MODEL]
            return nn_gradio.Dropdown(choices=choices, value=current_model)
    
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
            timeout=5.0
        )
        models_data = client.models.list()
        model_ids = sorted([model.id for model in models_data.data])
        if not model_ids:
            choices = [current_model] if current_model else [DEFAULT_MODEL]
        else:
            choices = model_ids
            if current_model and current_model not in choices:
                choices = [current_model] + choices
        return nn_gradio.Dropdown(choices=choices, value=current_model)
    except Exception as e:
        print(f"Error fetching models: {e}")
        choices = [current_model] if current_model else [DEFAULT_MODEL]
        return nn_gradio.Dropdown(choices=choices, value=current_model)

# Gradio interface build function
def build_app():
    with nn_gradio.Blocks(title="tAIwa - Multi-Persona Chatroom") as demo:
        
        # State stores
        state_history = nn_gradio.State([])          # List of (speaker, content, index_for_color)
        state_participants = nn_gradio.State([])      # List of persona dicts
        state_current_idx = nn_gradio.State(0)       # Index in the participant list
        state_agenda = nn_gradio.State("")           # Topic/Agenda
        
        nn_gradio.Markdown(i18n("app_title"))
        nn_gradio.Markdown(i18n("app_desc"))
        
        with nn_gradio.Row():
            # --- LEFT COLUMN: SETTINGS ---
            with nn_gradio.Column(scale=4, elem_classes=["glass-panel"]):
                nn_gradio.Markdown(i18n("api_config"))
                api_key_input = nn_gradio.Textbox(label=i18n("api_key"), type="password", placeholder="sk-...", value=DEFAULT_API_KEY)
                base_url_input = nn_gradio.Textbox(label=i18n("base_url"), placeholder="https://api.openai.com/v1", value=DEFAULT_BASE_URL)
                with nn_gradio.Row(elem_classes=["align-end"]):
                    model_input = nn_gradio.Dropdown(
                        label=i18n("model_name"),
                        choices=[DEFAULT_MODEL],
                        value=DEFAULT_MODEL,
                        allow_custom_value=True,
                        scale=4
                    )
                    btn_fetch_models = nn_gradio.Button("🔄", scale=0, min_width=50, elem_classes=["refresh-btn"])
                
                nn_gradio.Markdown(i18n("persona_config"))
                persona_count = nn_gradio.Slider(minimum=2, maximum=4, step=1, value=3, label=i18n("num_personas"))
                
                # Persona Configuration Blocks (Max 4)
                persona_configs = []
                for i in range(4):
                    with nn_gradio.Group(visible=(i < 3)) as group:
                        nn_gradio.Markdown(i18n(f"persona_title_{i+1}"))
                        with nn_gradio.Row():
                            name = nn_gradio.Textbox(label=i18n("name_label"), value=i18n(f"persona_default_name_{i+1}"), scale=3)
                            is_human = nn_gradio.Checkbox(label=i18n("human_label"), value=False, scale=1)
                        prompt = nn_gradio.Textbox(label=i18n("system_prompt_label"), value=i18n("system_prompt_default"), lines=2)
                        persona_configs.append({
                            "group": group,
                            "name": name,
                            "is_human": is_human,
                            "prompt": prompt
                        })
                
                # Handle dynamic slider visibility changes
                def adjust_persona_visibility(count):
                    updates = []
                    for i in range(4):
                        updates.append(nn_gradio.Group(visible=(i < count)))
                    return updates
                
                persona_count.change(
                    fn=adjust_persona_visibility,
                    inputs=[persona_count],
                    outputs=[cfg["group"] for cfg in persona_configs]
                )
                
            # --- RIGHT COLUMN: CHATROOM ---
            with nn_gradio.Column(scale=6, elem_classes=["glass-panel"]):
                nn_gradio.Markdown(i18n("chatroom_panel"))
                agenda_input = nn_gradio.Textbox(
                    label=i18n("agenda_label"), 
                    placeholder=i18n("agenda_placeholder"), 
                    value=i18n("agenda_default")
                )
                
                with nn_gradio.Row():
                    btn_initialize = nn_gradio.Button(i18n("btn_start"), elem_classes=["start-btn"])
                    btn_step = nn_gradio.Button(i18n("btn_next"), variant="secondary", interactive=False)
                    btn_reset = nn_gradio.Button(i18n("btn_reset"), variant="stop")
                
                status_box = nn_gradio.Markdown(i18n("setup_instruction"))
                
                # Visual Chat Log (Custom styled HTML display for color-coding)
                chat_display = nn_gradio.HTML(
                    value=i18n("empty_chat_log"),
                    label="Chat Log"
                )
                
                with nn_gradio.Row():
                    btn_step_bottom = nn_gradio.Button(i18n("btn_next"), variant="secondary", interactive=False)
                
                # Human Response input row (initially hidden/disabled)
                with nn_gradio.Row(visible=False) as human_input_row:
                    human_textbox = nn_gradio.Textbox(
                        label=i18n("human_resp_label"), 
                        placeholder=i18n("human_resp_placeholder"),
                        scale=4
                    )
                    btn_submit_human = nn_gradio.Button(i18n("btn_send"), variant="primary", scale=1)

        # --- Functions and Event Logic ---
        
        def render_chat_html(history_list):
            """Renders conversation history as custom-styled HTML blocks."""
            if not history_list:
                return "<div style='text-align: center; color: #64748b; padding: 40px 0;'>Chat log is empty. Set agenda to begin!</div>"
            
            html_content = ["<div style='display: flex; flex-direction: column; gap: 8px;'>"]
            for speaker, text, color_idx in history_list:
                clean_text = text.replace("\n", "<br>")
                # Escape markdown block quotes or HTML formatting inside text for safety
                html_content.append(f"""
                <div class="chat-bubble bubble-persona-{color_idx}">
                    <strong>{speaker}</strong>:
                    <div style="margin-top: 4px;">{clean_text}</div>
                </div>
                """)
            html_content.append("</div>")
            return "".join(html_content)

        def initialize_conversation(request: nn_gradio.Request, count, agenda, api_key, base_url, model, *args):
            """Collects configurations and starts the conversation flow."""
            lang = get_language(request)
            
            # Setup list of active personas
            personas = []
            for i in range(count):
                name_val = args[i]
                is_human_val = args[4 + i]
                prompt_val = args[8 + i]
                
                personas.append({
                    "name": name_val,
                    "is_human": is_human_val,
                    "prompt": prompt_val,
                    "color_idx": i
                })
            
            # Initial agenda message
            initial_msg = LOCALIZED_STRINGS[lang]["agenda_prefix"].format(agenda=agenda)
            system_log_item = ("System" if lang == "en" else "システム", initial_msg, 0)
            initial_history = [system_log_item]
            
            # Reset pointer to turn 0
            current_idx = 0
            
            # If the first persona is AI, execute their response immediately!
            first_persona = personas[0]
            if not first_persona["is_human"]:
                history_text = format_history_for_prompt([]) # Empty dialogue history
                participant_names = ", ".join([p["name"] for p in personas])
                
                system_prompt = LOCALIZED_STRINGS[lang]["roleplay_system_prompt"].format(
                    persona_prompt=first_persona['prompt'],
                    persona_name=first_persona['name'],
                    agenda=agenda,
                    participant_names=participant_names
                )
                
                full_prompt = LOCALIZED_STRINGS[lang]["roleplay_full_prompt"].format(
                    persona_name=first_persona['name'],
                    history_text=history_text
                )
                
                # Generate first response
                response_text = generate_ai_response(api_key, base_url, model, system_prompt, full_prompt)
                
                # Localize backend system or API errors
                if "⚠️ System error" in response_text:
                    response_text = LOCALIZED_STRINGS[lang]["api_key_missing"]
                elif "⚠️ API Error" in response_text:
                    err_msg = response_text.replace("⚠️ API Error: ", "")
                    response_text = LOCALIZED_STRINGS[lang]["api_error"].format(error=err_msg)

                # Append message to log
                new_item = (first_persona["name"], response_text, first_persona["color_idx"])
                initial_history.append(new_item)
                
                # Move to next turn
                current_idx = 1
            
            # Format UI for the next turn
            status_msg, show_human_input, human_name = update_ui_state(lang, current_idx, personas)
            
            # Return updated states and elements
            return (
                status_msg,
                initial_history,
                current_idx,
                agenda,
                personas,
                render_chat_html(initial_history),
                nn_gradio.Button(interactive=not show_human_input), # btn_step
                nn_gradio.Button(interactive=not show_human_input), # btn_step_bottom
                nn_gradio.Row(visible=show_human_input),
                nn_gradio.Button(interactive=False) # Disable setup button after starting
            )

        # Collect arguments for initialization helper
        init_inputs = [
            persona_count, agenda_input, api_key_input, base_url_input, model_input
        ] + [cfg["name"] for cfg in persona_configs] + [cfg["is_human"] for cfg in persona_configs] + [cfg["prompt"] for cfg in persona_configs]

        btn_fetch_models.click(
            fn=update_models_dropdown,
            inputs=[api_key_input, base_url_input, model_input],
            outputs=[model_input]
        )

        btn_initialize.click(
            fn=initialize_conversation,
            inputs=init_inputs,
            outputs=[
                status_box, state_history, state_current_idx, state_agenda, 
                state_participants, chat_display, btn_step, btn_step_bottom, human_input_row, btn_initialize
            ]
        )

        def proceed_next_turn(request: nn_gradio.Request, history, current_idx, agenda, personas, api_key, base_url, model):
            """Executes the AI turn or proceeds to wait for human turn."""
            lang = get_language(request)
            if not personas:
                return LOCALIZED_STRINGS[lang]["turn_start"], history, current_idx, "", nn_gradio.Button(interactive=False), nn_gradio.Button(interactive=False), nn_gradio.Row(visible=False)
            
            current_persona = personas[current_idx]
            
            # Security check: if current is human, we should not execute AI generation
            if current_persona["is_human"]:
                # Should wait for human input, do nothing here
                status_msg, show_human_input, _ = update_ui_state(lang, current_idx, personas)
                return status_msg, history, current_idx, render_chat_html(history), nn_gradio.Button(interactive=False), nn_gradio.Button(interactive=False), nn_gradio.Row(visible=True)
            
            # Build prompts - limit context history to last 12 messages to keep token usage low
            recent_history = [(h[0], h[1]) for h in history if h[0] not in ["System", "システム"]]
            recent_history = recent_history[-12:]
            history_text = format_history_for_prompt(recent_history)
            participant_names = ", ".join([p["name"] for p in personas])
            
            system_prompt = LOCALIZED_STRINGS[lang]["roleplay_system_prompt"].format(
                persona_prompt=current_persona['prompt'],
                persona_name=current_persona['name'],
                agenda=agenda,
                participant_names=participant_names
            )
            
            full_prompt = LOCALIZED_STRINGS[lang]["roleplay_full_prompt"].format(
                persona_name=current_persona['name'],
                history_text=history_text
            )
            
            # Generate response
            response_text = generate_ai_response(api_key, base_url, model, system_prompt, full_prompt)
            
            # Localize backend system or API errors
            if "⚠️ System error" in response_text:
                response_text = LOCALIZED_STRINGS[lang]["api_key_missing"]
            elif "⚠️ API Error" in response_text:
                err_msg = response_text.replace("⚠️ API Error: ", "")
                response_text = LOCALIZED_STRINGS[lang]["api_error"].format(error=err_msg)

            # Append message to log
            new_item = (current_persona["name"], response_text, current_persona["color_idx"])
            updated_history = history + [new_item]
            
            # Move to next index
            next_idx = (current_idx + 1) % len(personas)
            
            # Determine state for the next turn
            status_msg, show_human_input, _ = update_ui_state(lang, next_idx, personas)
            
            return (
                status_msg,
                updated_history,
                next_idx,
                render_chat_html(updated_history),
                nn_gradio.Button(interactive=not show_human_input), # btn_step
                nn_gradio.Button(interactive=not show_human_input), # btn_step_bottom
                nn_gradio.Row(visible=show_human_input)
            )

        btn_step.click(
            fn=proceed_next_turn,
            inputs=[
                state_history, state_current_idx, state_agenda, state_participants,
                api_key_input, base_url_input, model_input
            ],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row
            ]
        )

        btn_step_bottom.click(
            fn=proceed_next_turn,
            inputs=[
                state_history, state_current_idx, state_agenda, state_participants,
                api_key_input, base_url_input, model_input
            ],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row
            ]
        )

        def handle_human_response(request: nn_gradio.Request, human_text, history, current_idx, personas):
            """Processes human response text input and advances the turn."""
            lang = get_language(request)
            if not human_text.strip():
                # Don't allow empty response
                status_msg, show_human_input, _ = update_ui_state(lang, current_idx, personas)
                return (
                    status_msg, history, current_idx, render_chat_html(history), 
                    nn_gradio.Button(interactive=False), nn_gradio.Button(interactive=False), nn_gradio.Row(visible=True), ""
                )
            
            current_persona = personas[current_idx]
            
            # Append human reply
            new_item = (current_persona["name"], human_text.strip(), current_persona["color_idx"])
            updated_history = history + [new_item]
            
            # Move to next turn
            next_idx = (current_idx + 1) % len(personas)
            
            # Determine state for the next turn
            status_msg, show_human_input, _ = update_ui_state(lang, next_idx, personas)
            
            return (
                status_msg,
                updated_history,
                next_idx,
                render_chat_html(updated_history),
                nn_gradio.Button(interactive=not show_human_input), # btn_step
                nn_gradio.Button(interactive=not show_human_input), # btn_step_bottom
                nn_gradio.Row(visible=show_human_input),
                "" # Clear human text field
            )

        btn_submit_human.click(
            fn=handle_human_response,
            inputs=[human_textbox, state_history, state_current_idx, state_participants],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, human_textbox
            ]
        )
        
        # Also support pressing Enter in the human textbox to submit
        human_textbox.submit(
            fn=handle_human_response,
            inputs=[human_textbox, state_history, state_current_idx, state_participants],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, human_textbox
            ]
        )

        def reset_room(request: nn_gradio.Request):
            """Resets the chatroom to pristine state."""
            lang = get_language(request)
            return (
                LOCALIZED_STRINGS[lang]["setup_instruction"],
                [],
                0,
                "",
                [],
                LOCALIZED_STRINGS[lang]["empty_chat_log"],
                nn_gradio.Button(interactive=False), # btn_step
                nn_gradio.Button(interactive=False), # btn_step_bottom
                nn_gradio.Row(visible=False),
                nn_gradio.Button(interactive=True)
            )

        btn_reset.click(
            fn=reset_room,
            inputs=[],
            outputs=[
                status_box, state_history, state_current_idx, state_agenda, 
                state_participants, chat_display, btn_step, btn_step_bottom, human_input_row, btn_initialize
            ]
        )

    return demo

if __name__ == "__main__":
    demo_app = build_app()
    demo_app.launch(
        server_name="0.0.0.0" if IS_HF_SPACE else "127.0.0.1",
        server_port=7860,
        share=False,
        css=THEME_CSS,
        head=HEAD_HTML,
        i18n=i18n
    )