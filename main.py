import gradio as nn_gradio  # Rename to avoid name shadowing
import openai
import json
import time
import os
import random
import tempfile
import re
import html

# --- Detect HF Spaces Environment or local env keys ---
IS_HF_SPACE = (os.getenv("SYSTEM") == "spaces") or (os.getenv("SPACE_ID") is not None)
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN", "")
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

if IS_HF_SPACE:
    # If in HF Space, default to HF Serverless Inference with Llama-3 model
    DEFAULT_API_KEY = DEFAULT_HF_TOKEN
    DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
    DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
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
    background: rgba(255, 255, 255, 0.95) !important;
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
    background: rgba(22, 28, 45, 0.95) !important;
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

/* Mobile responsive optimizations */
@media (max-width: 768px) {
    .gradio-container {
        padding: 8px !important;
    }
    
    .glass-panel {
        padding: 12px !important;
        margin-bottom: 12px !important;
        border-radius: 12px !important;
    }
    
    .chat-bubble {
        padding: 10px 14px !important;
        font-size: 0.92rem !important;
    }
}
"""

HEAD_HTML = '<meta name="google" content="notranslate">'

# --- Localization & Translation Setup ---
I18N_DATA = {
    "en": {
        "app_title": "# 🌐 tAIwa - Multi-Persona Chatroom",
        "app_desc": "A tool where multiple AI agents (and you) can converse freely.",
        "api_config": "### ⚙️ 1. OpenAI-compatible API Configuration",
        "api_key": "API Key",
        "api_key_placeholder": "Leave empty for shared Hugging Face Token...",
        "api_key_info": "If empty, the shared space token (HF Serverless Inference) will be used. If you hit rate limits, please input your own Hugging Face Token or OpenAI Key.",
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
        "btn_undo": "↩️ Undo",
        "human_resp_label": "Your Response",
        "human_resp_placeholder": "Enter your response...",
        "btn_send": "Send",
        "autoplay_label": "Auto-Play",
        "preset_label": "Preset Template",
        "preset_custom": "Custom (Manual Setup)",
        "btn_copy": "📋 Copy Log",
        "btn_save": "💾 Save Log",
        "file_label": "Download Log"
    },
    "ja": {
        "app_title": "# 🌐 tAIwa - マルチペルソナ チャットルーム",
        "app_desc": "複数のAIエージェント（とあなた）が自由に対話できるツールです。",
        "api_config": "### ⚙️ 1. OpenAI互換API設定",
        "api_key": "APIキー",
        "api_key_placeholder": "空欄時は共有Hugging Faceトークンを使用...",
        "api_key_info": "空欄の場合、Spaceの共有トークン（HF Serverless Inference）を使用します。レート制限などでエラーが発生する場合は、ご自身のHugging FaceトークンまたはOpenAI APIキーを入力してください。",
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
        "btn_undo": "↩️ 1つ戻す",
        "human_resp_label": "あなたの返答",
        "human_resp_placeholder": "返答を入力してください...",
        "btn_send": "送信",
        "autoplay_label": "自動進行",
        "preset_label": "プリセットテンプレート",
        "preset_custom": "カスタム（手動設定）",
        "btn_copy": "📋 ログをコピー",
        "btn_save": "💾 ログを保存",
        "file_label": "ログのダウンロード"
    }
}

if not IS_HF_SPACE:
    I18N_DATA["en"]["api_key_info"] = "Please enter your OpenAI API Key."
    I18N_DATA["ja"]["api_key_info"] = "OpenAI APIキーを入力してください。"
    I18N_DATA["en"]["api_key_placeholder"] = "API Key..."
    I18N_DATA["ja"]["api_key_placeholder"] = "APIキーを入力..."

i18n = nn_gradio.I18n(**I18N_DATA)

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

PRESETS = {
    "en": {
        "Technical Debate": [
            {"name": "Architect", "prompt": "You are a software architect who values scalability, clean code, and robust system design. You are pragmatic but strict about best practices.", "is_human": False},
            {"name": "Developer", "prompt": "You are a hands-on software engineer. You care about development speed, developer experience, and simple, maintainable solutions.", "is_human": False},
            {"name": "Security Expert", "prompt": "You are a cybersecurity expert. You focus on threat models, data privacy, vulnerabilities, and risk mitigation.", "is_human": False}
        ],
        "Creative Brainstorming": [
            {"name": "Innovator", "prompt": "You are a creative visionary. You suggest bold, out-of-the-box, and futuristic ideas without worrying about initial constraints.", "is_human": False},
            {"name": "Pragmatist", "prompt": "You are a practical project manager. You focus on execution, budget, timeline, and how to actually build ideas.", "is_human": False},
            {"name": "Marketer", "prompt": "You are a growth marketer. You focus on user acquisition, viral hooks, messaging, and how to position the product in the market.", "is_human": False}
        ],
        "Philosophical Cafe": [
            {"name": "Socrates", "prompt": "You are Socrates. You use the Socratic method, asking questions to challenge assumptions and uncover deeper truths. Be polite but inquisitive.", "is_human": False},
            {"name": "Nietzsche", "prompt": "You are Friedrich Nietzsche. You value individual will, overcoming struggle, and questioning conventional morals. Speak passionately and intensely.", "is_human": False},
            {"name": "Modern Citizen", "prompt": "You are a reasonable modern citizen. You focus on everyday practicality, human happiness, and balance in modern society.", "is_human": False}
        ],
        "Casual Chat": [
            {"name": "Alex", "prompt": "You are a friendly, casual, and easygoing friend. You like joking around and talking about hobbies, daily life, and popular topics. Speak in a warm, relaxed, and informal tone.", "is_human": False},
            {"name": "Jordan", "prompt": "You are a curious, optimistic, and enthusiastic friend. You love sharing fun facts, interesting stories, and asking engaging follow-up questions. Keep the conversation lively and cheerful.", "is_human": False},
            {"name": "Taylor", "prompt": "You are a calm, supportive, and thoughtful friend. You offer good listening, lighthearted humor, and sensible, comforting advice. Speak in a cozy and informal style.", "is_human": False}
        ]
    },
    "ja": {
        "技術ディスカッション": [
            {"name": "アーキテクト", "prompt": "システム開発の全体設計を担当するアーキテクトです。スケーラビリティ、堅牢性、クリーンコードを重視し、妥協のない最適な設計を提案します。", "is_human": False},
            {"name": "開発者", "prompt": "現場でコードを書くエンジニアです。開発速度、生産性、シンプルな実装、および保守性の高さを重視します。", "is_human": False},
            {"name": "セキュリティ担当", "prompt": "セキュリティ監査の専門家です。脆弱性、データ保護、暗号化、リスク管理の観点から厳しい意見を述べます。", "is_human": False}
        ],
        "アイデア会議": [
            {"name": "革新者", "prompt": "枠にとらわれない大胆で未来的なアイデアを提案するイノベーターです。実現可能性よりも面白さや新しさを重視します。", "is_human": False},
            {"name": "現実主義者", "prompt": "現実的なプロジェクトマネージャーです。予算、納期、人員、技術的制約の観点から、どうやって形にするかを考えます。", "is_human": False},
            {"name": "マーケター", "prompt": "ユーザー獲得と市場への訴求を考えるマーケターです。バイラル性、ユーザーメリット、ポジショニングを重視します。", "is_human": False}
        ],
        "哲学カフェ": [
            {"name": "ソクラテス", "prompt": "古代ギリシャの哲学者ソクラテスです。問いかけること（産婆術）で相手の前提を揺さぶり、真理を追求します。礼儀正しくも執拗に質問します。", "is_human": False},
            {"name": "ニーチェ", "prompt": "哲学者ニーチェです。ルサンチマンを排し、自らの「力への意志」によって運命を愛することを説きます。情熱的で鋭い言葉遣いをします。", "is_human": False},
            {"name": "現代市民", "prompt": "ごく一般的な常識を持つ現代市民です。日常生活の利便性、幸福、調和の観点から、極端な思想に対してバランスの取れた意見を述べます。", "is_human": False}
        ],
        "雑談ルーム": [
            {"name": "アキ", "prompt": "明るくてフランクな友人です。趣味や日常の出来事、最近流行っていることについてカジュアルに話すのが好きです。親しみやすいタメ口（〜だよ、〜だね）でリラックスして話します。", "is_human": False},
            {"name": "リン", "prompt": "好奇心旺盛でポジティブな友人です。相槌を打ちながら「そういえば〜」と面白い話や楽しい質問を投げかけるのが得意です。元気でフレンドリーなタメ口で話します。", "is_human": False},
            {"name": "ハル", "prompt": "穏やかで聞き上手な優しい友人です。他人の話を温かく受け止め、クスッと笑えるユーモアを交えたり、のんびりしたアドバイスをくれたりします。落ち着いたタメ口で話します。", "is_human": False}
        ]
    }
}

RANDOM_AGENDAS = {
    "en": [
        "Is tabs or spaces superior for code indentation?",
        "Will Artificial General Intelligence (AGI) be achieved by 2030?",
        "Should humanity prioritize colonizing Mars or solving Earth's climate crisis?",
        "Is a hotdog technically a sandwich?",
        "What is the most important programming language to learn today, and why?",
        "Should remote work remain the default standard for tech companies?",
        "Does pineapple belong on pizza?",
        "Is time travel theoretically possible, and what are the ethical implications?",
        "Should code comments be treated as a code smell (i.e., code should be self-documenting)?"
    ],
    "ja": [
        "コードのインデントはタブ派かスペース派か？",
        "2030年までに汎用人工知能（AGI）は実現するか？",
        "人類は火星移住と地球の気候変動対策のどちらを優先すべきか？",
        "ホットドッグはサンドイッチに分類されるべきか？",
        "現代において最も学ぶ価値のあるプログラミング言語とその理由は？",
        "IT企業においてリモートワークは恒久的な標準スタイルであるべきか？",
        "酢豚にパイナップルを入れるのは許せるか？",
        "タイムトラベルは理論的に可能か？またその倫理的影響は？",
        "コードのコメントは不要（コード自身が説明的であるべき）という意見についてどう思うか？"
    ]
}

def get_preset_choices(lang):
    if lang == "ja":
        return ["カスタム（手動設定）"] + list(PRESETS["ja"].keys())
    return ["Custom (Manual Setup)"] + list(PRESETS["en"].keys())

def apply_preset(preset_name, lang):
    is_custom = preset_name in ["Custom (Manual Setup)", "カスタム（手動設定）", None, ""]
    no_change = [nn_gradio.Slider()] + [nn_gradio.Textbox(), nn_gradio.Checkbox(), nn_gradio.Textbox()] * 4
    if is_custom:
        return no_change
    
    preset_data = None
    if lang == "ja" and preset_name in PRESETS["ja"]:
        preset_data = PRESETS["ja"][preset_name]
    elif preset_name in PRESETS["en"]:
        preset_data = PRESETS["en"][preset_name]
    else:
        for l in PRESETS:
            if preset_name in PRESETS[l]:
                preset_data = PRESETS[l][preset_name]
                break
                
    if not preset_data:
        return no_change
        
    num_personas = len(preset_data)
    outputs = [nn_gradio.Slider(value=num_personas)]
    for i in range(4):
        if i < num_personas:
            p = preset_data[i]
            outputs.append(nn_gradio.Textbox(value=p["name"]))
            outputs.append(nn_gradio.Checkbox(value=p.get("is_human", False)))
            outputs.append(nn_gradio.Textbox(value=p["prompt"]))
        else:
            outputs.append(nn_gradio.Textbox())
            outputs.append(nn_gradio.Checkbox())
            outputs.append(nn_gradio.Textbox())
            
    return outputs

def set_custom_preset(lang):
    custom_val = "カスタム（手動設定）" if lang == "ja" else "Custom (Manual Setup)"
    return custom_val

def get_random_agenda(lang):
    agendas = RANDOM_AGENDAS.get(lang, RANDOM_AGENDAS["en"])
    return random.choice(agendas)

def generate_markdown_log(history, agenda, lang):
    if not history:
        return ""
    
    title = "tAIwa Chat Log" if lang == "en" else "tAIwa 対話ログ"
    agenda_label = "Agenda/Topic" if lang == "en" else "アジェンダ/トピック"
    
    lines = []
    lines.append(f"# 🌐 {title}")
    lines.append(f"**{agenda_label}**: {agenda}\n")
    lines.append("--- \n")
    
    for speaker, text, _ in history:
        if speaker in ["System", "システム"]:
            lines.append(f"*{text}*\n")
        else:
            lines.append(f"### **{speaker}**")
            lines.append(f"{text}\n")
            
    return "\n".join(lines)

def copy_log_python(history, agenda, lang):
    if not history:
        msg = "Chat log is empty. / チャットログが空です。"
        nn_gradio.Warning(msg)
        return ""
    
    markdown_text = generate_markdown_log(history, agenda, lang)
    msg = "Copied chat log to clipboard! / ログをクリップボードにコピーしました！"
    nn_gradio.Info(msg)
    return markdown_text

def export_chat_log(history, agenda, lang):
    if not history:
        nn_gradio.Warning("Chat log is empty. / チャットログが空です。")
        return None
        
    content = generate_markdown_log(history, agenda, lang)
    
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "taiwa_chat_log.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    return file_path

def simple_markdown_to_html(text):
    if not text:
        return ""
    # 1. Escape HTML
    text = html.escape(text)
    
    # 2. Extract code blocks and replace with placeholders to avoid formatting inside them
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(1))
        return f"__CODE_BLOCK_PLACEHOLDER_{len(code_blocks)-1}__"
    
    # Match ```lang ... ``` or just ``` ... ```
    text = re.sub(r'```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)```', save_code_block, text)
    text = re.sub(r'```([\s\S]*?)```', save_code_block, text)
    
    # 3. Apply inline styling (bold, italic, inline code)
    # Inline code `code`
    inline_codes = []
    def save_inline_code(match):
        inline_codes.append(match.group(1))
        return f"__INLINE_CODE_PLACEHOLDER_{len(inline_codes)-1}__"
    text = re.sub(r'`([^`\n]+)`', save_inline_code, text)
    
    # Bold **bold**
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    # Italic *italic*
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    
    # Convert newlines to <br> for spacing
    text = text.replace("\n", "<br>")
    
    # 4. Restore placeholders with styled HTML
    for idx, code in enumerate(inline_codes):
        placeholder = f"__INLINE_CODE_PLACEHOLDER_{idx}__"
        styled = f'<code style="background: rgba(0,0,0,0.05); padding: 2px 4px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">{code}</code>'
        text = text.replace(placeholder, styled)
        
    for idx, block in enumerate(code_blocks):
        placeholder = f"__CODE_BLOCK_PLACEHOLDER_{idx}__"
        styled = f'<pre style="background: rgba(0,0,0,0.05); padding: 12px; border-radius: 8px; overflow-x: auto; font-family: monospace; font-size: 0.9em; margin: 8px 0; line-height: 1.4;"><code>{block}</code></pre>'
        text = text.replace(placeholder, styled)
        
    return text

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
            return nn_gradio.Dropdown(choices=choices, value=current_model, interactive=True)
    
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
        return nn_gradio.Dropdown(choices=choices, value=current_model, interactive=True)
    except Exception as e:
        print(f"Error fetching models: {e}")
        choices = [current_model] if current_model else [DEFAULT_MODEL]
        return nn_gradio.Dropdown(choices=choices, value=current_model, interactive=True)

# Gradio interface build function
def build_app():
    with nn_gradio.Blocks(title="tAIwa - Multi-Persona Chatroom") as demo:
        
        # State stores
        state_history = nn_gradio.State([])          # List of (speaker, content, index_for_color)
        state_participants = nn_gradio.State([])      # List of persona dicts
        state_current_idx = nn_gradio.State(0)       # Index in the participant list
        state_agenda = nn_gradio.State("")           # Topic/Agenda
        state_lang = nn_gradio.State("en")           # Language State ("en" or "ja")
        
        # Language Switcher Radio on top
        with nn_gradio.Row():
            language_selector = nn_gradio.Radio(
                choices=["日本語", "English"],
                value="English",
                label="Language / 言語",
                scale=2
            )
            
        app_title_md = nn_gradio.Markdown(I18N_DATA["en"]["app_title"])
        app_desc_md = nn_gradio.Markdown(I18N_DATA["en"]["app_desc"])
        
        with nn_gradio.Row():
            # --- LEFT COLUMN: SETTINGS ---
            with nn_gradio.Column(scale=4, elem_classes=["glass-panel"]):
                api_config_md = nn_gradio.Markdown(I18N_DATA["en"]["api_config"])
                api_key_input = nn_gradio.Textbox(
                    label=I18N_DATA["en"]["api_key"], 
                    type="password", 
                    placeholder=I18N_DATA["en"]["api_key_placeholder"], 
                    info=I18N_DATA["en"]["api_key_info"], 
                    value=DEFAULT_API_KEY
                )
                base_url_input = nn_gradio.Textbox(label=I18N_DATA["en"]["base_url"], placeholder="https://api.openai.com/v1", value=DEFAULT_BASE_URL)
                with nn_gradio.Row(elem_classes=["align-end"]):
                    model_input = nn_gradio.Dropdown(
                        label=I18N_DATA["en"]["model_name"],
                        choices=[DEFAULT_MODEL],
                        value=DEFAULT_MODEL,
                        allow_custom_value=True,
                        scale=4
                    )
                    btn_fetch_models = nn_gradio.Button("🔄", scale=0, min_width=50, elem_classes=["refresh-btn"])
                
                persona_config_md = nn_gradio.Markdown(I18N_DATA["en"]["persona_config"])
                preset_dropdown = nn_gradio.Dropdown(
                    label=I18N_DATA["en"]["preset_label"],
                    choices=["Custom (Manual Setup)"],
                    value="Custom (Manual Setup)",
                    interactive=True
                )
                persona_count = nn_gradio.Slider(minimum=2, maximum=4, step=1, value=3, label=I18N_DATA["en"]["num_personas"])
                
                # Persona Configuration Blocks (Max 4)
                persona_configs = []
                for i in range(4):
                    with nn_gradio.Group(visible=(i < 3)) as group:
                        p_title = nn_gradio.Markdown(I18N_DATA["en"][f"persona_title_{i+1}"])
                        with nn_gradio.Row():
                            name = nn_gradio.Textbox(label=I18N_DATA["en"]["name_label"], value=I18N_DATA["en"][f"persona_default_name_{i+1}"], scale=3)
                            is_human = nn_gradio.Checkbox(label=I18N_DATA["en"]["human_label"], value=False, scale=1)
                        prompt = nn_gradio.Textbox(label=I18N_DATA["en"]["system_prompt_label"], value=I18N_DATA["en"]["system_prompt_default"], lines=2)
                        persona_configs.append({
                            "group": group,
                            "title_md": p_title,
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
                chatroom_panel_md = nn_gradio.Markdown(I18N_DATA["en"]["chatroom_panel"])
                with nn_gradio.Row():
                    agenda_input = nn_gradio.Textbox(
                        label=I18N_DATA["en"]["agenda_label"], 
                        placeholder=I18N_DATA["en"]["agenda_placeholder"], 
                        value=I18N_DATA["en"]["agenda_default"],
                        scale=8
                    )
                    btn_random_agenda = nn_gradio.Button("🎲", scale=0, min_width=50, elem_classes=["refresh-btn"])
                
                with nn_gradio.Row():
                    btn_initialize = nn_gradio.Button(I18N_DATA["en"]["btn_start"], elem_classes=["start-btn"], scale=2)
                    btn_step = nn_gradio.Button(I18N_DATA["en"]["btn_next"], variant="secondary", interactive=False, scale=2)
                    autoplay_check = nn_gradio.Checkbox(label=I18N_DATA["en"]["autoplay_label"], value=False, scale=1)
                    btn_undo = nn_gradio.Button(I18N_DATA["en"]["btn_undo"], variant="secondary", interactive=False, scale=1)
                    btn_reset = nn_gradio.Button(I18N_DATA["en"]["btn_reset"], variant="stop", scale=1)
                
                status_box = nn_gradio.Markdown(LOCALIZED_STRINGS["en"]["setup_instruction"])
                
                # Visual Chat Log (Custom styled HTML display for color-coding)
                chat_display = nn_gradio.HTML(
                    value=LOCALIZED_STRINGS["en"]["empty_chat_log"],
                    label="Chat Log"
                )
                
                with nn_gradio.Row():
                    btn_step_bottom = nn_gradio.Button(I18N_DATA["en"]["btn_next"], variant="secondary", interactive=False, scale=2)
                    btn_copy = nn_gradio.Button(I18N_DATA["en"]["btn_copy"], scale=1)
                    btn_export = nn_gradio.Button(I18N_DATA["en"]["btn_save"], scale=1)
                
                # Hidden components for copying / downloading
                hidden_markdown = nn_gradio.Textbox(visible=False)
                download_file = nn_gradio.File(label=I18N_DATA["en"]["file_label"], visible=False)
                
                # Human Response input row (initially hidden/disabled)
                with nn_gradio.Row(visible=False) as human_input_row:
                    human_textbox = nn_gradio.Textbox(
                        label=I18N_DATA["en"]["human_resp_label"], 
                        placeholder=I18N_DATA["en"]["human_resp_placeholder"],
                        scale=4
                    )
                    btn_submit_human = nn_gradio.Button(I18N_DATA["en"]["btn_send"], variant="primary", scale=1)

        # --- Functions and Event Logic ---
        
        def render_chat_html(history_list):
            """Renders conversation history as custom-styled HTML blocks."""
            if not history_list:
                return "<div style='text-align: center; color: #64748b; padding: 40px 0;'>Chat log is empty. Set agenda to begin!</div>"
            
            html_content = ["<div style='display: flex; flex-direction: column; gap: 8px;'>"]
            for speaker, text, color_idx in history_list:
                clean_text = simple_markdown_to_html(text)
                html_content.append(f"""
                <div class="chat-bubble bubble-persona-{color_idx}">
                    <strong>{speaker}</strong>:
                    <div style="margin-top: 4px;">{clean_text}</div>
                </div>
                """)
            html_content.append("</div>")
            return "".join(html_content)

        def initialize_conversation(lang, count, agenda, api_key, base_url, model, *args):
            """Collects configurations and starts the conversation flow."""
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
            
            # Format UI for the next turn
            status_msg, show_human_input, human_name = update_ui_state(lang, current_idx, personas)
            
            return (
                status_msg,
                initial_history,
                current_idx,
                agenda,
                personas,
                render_chat_html(initial_history),
                nn_gradio.Button(interactive=False), # btn_step
                nn_gradio.Button(interactive=False), # btn_step_bottom
                nn_gradio.Row(visible=show_human_input),
                nn_gradio.Button(interactive=False) # Disable setup button after starting
            )

        def proceed_next_turn(lang, history, current_idx, agenda, personas, api_key, base_url, model, autoplay):
            """Executes the AI turn or proceeds to wait for human turn."""
            if not personas:
                yield (
                    LOCALIZED_STRINGS[lang]["turn_start"], history, current_idx, "", 
                    nn_gradio.Button(interactive=False), nn_gradio.Button(interactive=False), nn_gradio.Row(visible=False)
                )
                return
            
            first_run = True
            while True:
                current_persona = personas[current_idx]
                
                # Security check: if current is human, we should not execute AI generation
                if current_persona["is_human"]:
                    status_msg, show_human_input, _ = update_ui_state(lang, current_idx, personas)
                    yield (
                        status_msg, history, current_idx, render_chat_html(history), 
                        nn_gradio.Button(interactive=False), nn_gradio.Button(interactive=False), nn_gradio.Row(visible=True)
                    )
                    break
                
                # If it's not the first run and autoplay is False, we stop at the next turn
                if not first_run and not autoplay:
                    status_msg, show_human_input, _ = update_ui_state(lang, current_idx, personas)
                    yield (
                        status_msg, history, current_idx, render_chat_html(history), 
                        nn_gradio.Button(interactive=not show_human_input), nn_gradio.Button(interactive=not show_human_input), nn_gradio.Row(visible=show_human_input)
                    )
                    break
                
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
                history = history + [new_item]
                
                # Move to next index
                current_idx = (current_idx + 1) % len(personas)
                first_run = False
                
                # Determine state for the next turn
                status_msg, show_human_input, _ = update_ui_state(lang, current_idx, personas)
                
                yield (
                    status_msg,
                    history,
                    current_idx,
                    render_chat_html(history),
                    nn_gradio.Button(interactive=not show_human_input and not autoplay),
                    nn_gradio.Button(interactive=not show_human_input and not autoplay),
                    nn_gradio.Row(visible=show_human_input)
                )
                
                if autoplay:
                    time.sleep(1.5)

        # Collect arguments for initialization helper
        init_inputs = [
            state_lang, persona_count, agenda_input, api_key_input, base_url_input, model_input
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
                state_participants, chat_display, btn_step, btn_step_bottom, human_input_row, btn_initialize, btn_undo
            ]
        ).then(
            fn=proceed_next_turn,
            inputs=[
                state_lang, state_history, state_current_idx, state_agenda, state_participants,
                api_key_input, base_url_input, model_input, autoplay_check
            ],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, btn_undo, autoplay_check
            ]
        )

        btn_step.click(
            fn=proceed_next_turn,
            inputs=[
                state_lang, state_history, state_current_idx, state_agenda, state_participants,
                api_key_input, base_url_input, model_input, autoplay_check
            ],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, btn_undo, autoplay_check
            ]
        )

        btn_step_bottom.click(
            fn=proceed_next_turn,
            inputs=[
                state_lang, state_history, state_current_idx, state_agenda, state_participants,
                api_key_input, base_url_input, model_input, autoplay_check
            ],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, btn_undo, autoplay_check
            ]
        )

        def handle_human_response(lang, human_text, history, current_idx, personas):
            """Processes human response text input and advances the turn."""
            if not human_text.strip():
                status_msg, show_human_input, _ = update_ui_state(lang, current_idx, personas)
                return (
                    status_msg, history, current_idx, render_chat_html(history), 
                    nn_gradio.Button(interactive=False), nn_gradio.Button(interactive=False), nn_gradio.Row(visible=True), "",
                    nn_gradio.Button(interactive=len(history) > 1)
                )
            
            current_persona = personas[current_idx]
            new_item = (current_persona["name"], human_text.strip(), current_persona["color_idx"])
            updated_history = history + [new_item]
            
            next_idx = (current_idx + 1) % len(personas)
            status_msg, show_human_input, _ = update_ui_state(lang, next_idx, personas)
            
            return (
                status_msg,
                updated_history,
                next_idx,
                render_chat_html(updated_history),
                nn_gradio.Button(interactive=not show_human_input), # btn_step
                nn_gradio.Button(interactive=not show_human_input), # btn_step_bottom
                nn_gradio.Row(visible=show_human_input),
                "", # Clear human text field
                nn_gradio.Button(interactive=True) # btn_undo
            )

        btn_submit_human.click(
            fn=handle_human_response,
            inputs=[state_lang, human_textbox, state_history, state_current_idx, state_participants],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, human_textbox, btn_undo
            ]
        ).then(
            fn=proceed_next_turn,
            inputs=[
                state_lang, state_history, state_current_idx, state_agenda, state_participants,
                api_key_input, base_url_input, model_input, autoplay_check
            ],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, btn_undo, autoplay_check
            ]
        )
        
        human_textbox.submit(
            fn=handle_human_response,
            inputs=[state_lang, human_textbox, state_history, state_current_idx, state_participants],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, human_textbox, btn_undo
            ]
        ).then(
            fn=proceed_next_turn,
            inputs=[
                state_lang, state_history, state_current_idx, state_agenda, state_participants,
                api_key_input, base_url_input, model_input, autoplay_check
            ],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, human_input_row, btn_undo, autoplay_check
            ]
        )

        def undo_last_turn(lang, history, current_idx, personas):
            """Removes the last message and goes back one turn."""
            if len(history) <= 1:
                return (
                    LOCALIZED_STRINGS[lang]["turn_start"],
                    history,
                    current_idx,
                    render_chat_html(history),
                    nn_gradio.Button(interactive=False), # btn_step
                    nn_gradio.Button(interactive=False), # btn_step_bottom
                    nn_gradio.Button(interactive=False), # btn_undo
                    nn_gradio.Row(visible=False),
                    False # autoplay_check
                )
            
            # Remove the last message from history
            history = history[:-1]
            
            # Revert to the previous speaker
            current_idx = (current_idx - 1) % len(personas)
            
            status_msg, show_human_input, _ = update_ui_state(lang, current_idx, personas)
            has_more_to_undo = len(history) > 1
            
            return (
                status_msg,
                history,
                current_idx,
                render_chat_html(history),
                nn_gradio.Button(interactive=not show_human_input), # btn_step
                nn_gradio.Button(interactive=not show_human_input), # btn_step_bottom
                nn_gradio.Button(interactive=has_more_to_undo),      # btn_undo
                nn_gradio.Row(visible=show_human_input),
                False # Turn off autoplay when undo is clicked
            )

        btn_undo.click(
            fn=undo_last_turn,
            inputs=[state_lang, state_history, state_current_idx, state_participants],
            outputs=[
                status_box, state_history, state_current_idx, chat_display,
                btn_step, btn_step_bottom, btn_undo, human_input_row, autoplay_check
            ]
        )

        def reset_room(lang):
            """Resets the chatroom to pristine state."""
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
                nn_gradio.Button(interactive=True), # btn_initialize
                nn_gradio.File(value=None, visible=False), # download_file
                nn_gradio.Button(interactive=False) # btn_undo
            )

        btn_reset.click(
            fn=reset_room,
            inputs=[state_lang],
            outputs=[
                status_box, state_history, state_current_idx, state_agenda, 
                state_participants, chat_display, btn_step, btn_step_bottom, 
                human_input_row, btn_initialize, download_file, btn_undo
            ]
        )

        # Preset template selection event
        preset_dropdown.change(
            fn=apply_preset,
            inputs=[preset_dropdown, state_lang],
            outputs=[persona_count] + [cfg[key] for cfg in persona_configs for key in ["name", "is_human", "prompt"]]
        )

        # Set preset selection to "Custom" if any persona configurations are manually edited
        for cfg in persona_configs:
            cfg["name"].change(fn=set_custom_preset, inputs=[state_lang], outputs=[preset_dropdown])
            cfg["is_human"].change(fn=set_custom_preset, inputs=[state_lang], outputs=[preset_dropdown])
            cfg["prompt"].change(fn=set_custom_preset, inputs=[state_lang], outputs=[preset_dropdown])

        # Random agenda event
        btn_random_agenda.click(
            fn=get_random_agenda,
            inputs=[state_lang],
            outputs=[agenda_input]
        )

        # Copy log event
        btn_copy.click(
            fn=copy_log_python,
            inputs=[state_history, state_agenda, state_lang],
            outputs=[hidden_markdown]
        ).then(
            fn=None,
            inputs=[hidden_markdown],
            js="""(text) => {
                if (!text) return;
                navigator.clipboard.writeText(text).catch(err => {
                    console.error('Could not copy text: ', err);
                });
            }"""
        )

        # Export log event
        btn_export.click(
            fn=export_chat_log,
            inputs=[state_history, state_agenda, state_lang],
            outputs=[download_file]
        )

        # Unified language switching logic
        def change_language(lang_choice, current_model):
            lang = "ja" if lang_choice == "日本語" else "en"
            
            presets_choices = get_preset_choices(lang)
            preset_val = presets_choices[0]
            
            updates = [
                lang,
                nn_gradio.Markdown(value=I18N_DATA[lang]["app_title"]),
                nn_gradio.Markdown(value=I18N_DATA[lang]["app_desc"]),
                nn_gradio.Markdown(value=I18N_DATA[lang]["api_config"]),
                nn_gradio.Textbox(label=I18N_DATA[lang]["api_key"], placeholder=I18N_DATA[lang]["api_key_placeholder"], info=I18N_DATA[lang]["api_key_info"], interactive=True),
                nn_gradio.Textbox(label=I18N_DATA[lang]["base_url"], interactive=True),
                nn_gradio.Dropdown(label=I18N_DATA[lang]["model_name"], choices=[current_model] if current_model else [DEFAULT_MODEL], value=current_model or DEFAULT_MODEL, interactive=True),
                nn_gradio.Markdown(value=I18N_DATA[lang]["persona_config"]),
                nn_gradio.Dropdown(label=I18N_DATA[lang]["preset_label"], choices=presets_choices, value=preset_val, interactive=True),
                nn_gradio.Slider(label=I18N_DATA[lang]["num_personas"], interactive=True),
                nn_gradio.Markdown(value=I18N_DATA[lang]["chatroom_panel"]),
                nn_gradio.Textbox(label=I18N_DATA[lang]["agenda_label"], placeholder=I18N_DATA[lang]["agenda_placeholder"], value=I18N_DATA[lang]["agenda_default"], interactive=True),
                nn_gradio.Button(value=I18N_DATA[lang]["btn_start"], interactive=True),
                nn_gradio.Button(value=I18N_DATA[lang]["btn_next"]),
                nn_gradio.Button(value=I18N_DATA[lang]["btn_next"]),
                nn_gradio.Checkbox(label=I18N_DATA[lang]["autoplay_label"], interactive=True),
                nn_gradio.Button(value=I18N_DATA[lang]["btn_undo"]),
                nn_gradio.Button(value=I18N_DATA[lang]["btn_reset"], interactive=True),
                nn_gradio.Markdown(value=LOCALIZED_STRINGS[lang]["setup_instruction"]),
                nn_gradio.HTML(value=LOCALIZED_STRINGS[lang]["empty_chat_log"]),
                nn_gradio.Button(value=I18N_DATA[lang]["btn_copy"], interactive=True),
                nn_gradio.Button(value=I18N_DATA[lang]["btn_save"], interactive=True),
                nn_gradio.File(label=I18N_DATA[lang]["file_label"]),
                nn_gradio.Textbox(label=I18N_DATA[lang]["human_resp_label"], placeholder=I18N_DATA[lang]["human_resp_placeholder"], interactive=True),
                nn_gradio.Button(value=I18N_DATA[lang]["btn_send"], interactive=True),
            ]
            
            for i in range(4):
                updates.append(nn_gradio.Markdown(value=I18N_DATA[lang][f"persona_title_{i+1}"]))
                updates.append(nn_gradio.Textbox(label=I18N_DATA[lang]["name_label"], value=I18N_DATA[lang][f"persona_default_name_{i+1}"], interactive=True))
                updates.append(nn_gradio.Checkbox(label=I18N_DATA[lang]["human_label"], interactive=True))
                updates.append(nn_gradio.Textbox(label=I18N_DATA[lang]["system_prompt_label"], value=I18N_DATA[lang]["system_prompt_default"], interactive=True))
                
            return updates

        lang_outputs = [
            state_lang, app_title_md, app_desc_md, api_config_md, api_key_input, base_url_input, model_input,
            persona_config_md, preset_dropdown, persona_count, chatroom_panel_md, agenda_input,
            btn_initialize, btn_step, btn_step_bottom, autoplay_check, btn_undo, btn_reset, status_box, chat_display,
            btn_copy, btn_export, download_file, human_textbox, btn_submit_human
        ]
        for cfg in persona_configs:
            lang_outputs.extend([cfg["title_md"], cfg["name"], cfg["is_human"], cfg["prompt"]])

        language_selector.change(
            fn=change_language,
            inputs=[language_selector, model_input],
            outputs=lang_outputs
        )

        # Load presets and detect initial language on page load
        def load_initial_state(request: nn_gradio.Request):
            lang = get_language(request)
            lang_choice = "日本語" if lang == "ja" else "English"
            preset_choices = get_preset_choices(lang)
            return lang_choice, lang, nn_gradio.Dropdown(choices=preset_choices, value=preset_choices[0], interactive=True)

        demo.load(
            fn=load_initial_state,
            inputs=[],
            outputs=[language_selector, state_lang, preset_dropdown]
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