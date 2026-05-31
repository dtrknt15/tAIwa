# 🌐 tAIwa - Multi-Persona Chatroom

**tAIwa** is a Gradio-based dialogue system that allows multiple AI agents (personas) and a human (you) to gather in a single chatroom to converse or debate freely on a set theme (agenda).

**Live Demo on Hugging Face Spaces**
<https://huggingface.co/spaces/dtrknt/tAIwa>

---

## ✨ Features

* 👥 **Multi-Persona Dialogue**: Configure up to 4 personas for group discussions among AI agents or including a human.
* 🤖 **Auto-Play Mode**: Let the AI agents carry out the conversation automatically without clicking buttons (it automatically pauses when it's your turn).
* 📋 **Persona Presets**: Load preconfigured settings such as "Tech Discussion", "Idea Brainstorming", or "Philosophy Cafe" with a single click.
* 🎲 **Random Agenda Generator**: Get interesting topic suggestions with one click when you're not sure what to discuss.
* 💾 **Copy & Save Dialogue Logs**: Copy the chat log to your clipboard formatted in clean Markdown, or download it as a file (`.md`).
* ⚙️ **OpenAI-Compatible API Support**:
  * Connects to the official OpenAI API, local API servers (LM Studio, Ollama, vLLM, etc.), Hugging Face Serverless Inference, and other OpenAI-compatible endpoints.
  * Auto-fetches the list of available models from the connected API endpoint.
* 🌐 **Multilingual Support (Japanese/English)**: Automatically detects the browser's language settings and displays the UI in Japanese or English.
* 🎨 **Premium UI Design**:
  * Beautiful glassmorphism design optimized for both light and dark modes.
  * Color-coded chat bubbles for each speaker (persona) for high readability.
  * Responsive layout for a comfortable user experience.

---

## 🚀 Setup & Execution

### 1. Install Required Libraries

Python 3.8 or higher is required. Install the necessary dependencies with the following command:

```bash
pip install gradio openai
```

### 2. Configure Environment Variables (Optional)

Setting up your API key in advance saves you the trouble of entering it at startup.

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

**macOS / Linux:**

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. Launch the Application

Run the following command in the root directory of the project:

```bash
python main.py
```

After startup, open the URL displayed in the console (default is `http://127.0.0.1:7860`) in your browser.

---

## 💡 How to Use

1. **API Settings (Top Left)**:
    * Set your API key, base URL, and model name. If you are using a local server, specify the local address (e.g., `http://localhost:1234/v1`) for the base URL.
    * Click the 🔄 button to fetch and select from the list of available models on the connected API.
2. **Persona Settings (Bottom Left)**:
    * Select your preferred preset template (Tech Discussion, Idea Brainstorming, Philosophy Cafe, etc.) to set the number of people, names, and system prompts all at once.
    * You can also adjust the "Number of Personas" slider to customize each persona's name, whether it is an AI or Human (Checkbox), and system prompt individually.
3. **Agenda & Discussion Start (Right Side)**:
    * Enter the discussion theme (agenda). Click the 🎲 button to generate and suggest a random topic.
    * Click the **🚀 Start** button to begin the chat.
4. **Conversation Flow**:
    * Check the **Auto-Play** checkbox to allow conversations between AI agents to proceed automatically. If unchecked (manual mode), click the **➡️ Next Turn** button to generate the next response.
    * When it is your turn, a message input field will appear at the bottom. Type your message and click **Send** (or press Enter).
    * Click the **🔄 Reset** button if you want to start over from the beginning.
5. **Saving Logs**:
    * Click the **📋 Copy Log** button below the chat panel to copy the entire log to your clipboard in Markdown format.
    * Click the **💾 Save Log** button to download the log file in Markdown format (`.md`).

# 📝 License

This project is licensed under the terms of the [Apache 2.0 License](LICENSE).
