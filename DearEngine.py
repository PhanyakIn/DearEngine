from dotenv import load_dotenv
load_dotenv()
import os
import time
import colorsys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Initialize Rich Console
console = Console()

# 0. THEME COLORS-----------------

def shade(hex_color: str, delta: float) -> str:
    """ปรับความสว่างของสี hex ขึ้น (delta > 0) หรือลง (delta < 0)"""
    try:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except Exception:
        r, g, b = 0x63 / 255, 0xA0 / 255, 0xC9 / 255  # fallback = #63a0c9
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    light = min(max(light + delta, 0.0), 1.0)
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))

THEME_BASE = os.getenv("DEAR_THEME_COLOR", "#63a0c9")
THEME = {
    "lightest": shade(THEME_BASE, +0.22),
    "lighter":  shade(THEME_BASE, +0.13),
    "light":    shade(THEME_BASE, +0.06),
    "base":     shade(THEME_BASE, 0),
    "dark":     shade(THEME_BASE, -0.07),
    "darker":   shade(THEME_BASE, -0.14),
    "darkest":  shade(THEME_BASE, -0.22),
}


# 1. TOOLS -----------------
@tool
def read_file(file_path: str) -> str:
    """Read the content of a specified file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write or update content in a specified file."""
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved file '{file_path}' !"
    except Exception as e:
        return f"Error writing file: {e}"

@tool
def list_files(directory: str = ".") -> str:
    """List all files in the given directory (recursive, max 50 files)."""
    try:
        files = []
        for root, dirs, filenames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'venv']]
            for f in filenames:
                files.append(os.path.relpath(os.path.join(root, f), directory))
        return "\n".join(files[:50])
    except Exception as e:
        return f"Error: {e}"

TOOLS = [read_file, write_file, list_files]

# 2. MODEL POOL MANAGER ---------------------
# Gemini -> Groq -> OpenRouter (:free only) -> Ollama

class AIModelPool:
    def __init__(self):
        self.models_config = [
            {
                "name": "Gemini",
                "type": "gemini",
                "model_name": os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
                "api_key": os.getenv("GEMINI_API_KEY_1") or os.getenv("GOOGLE_API_KEY") or "",
                "cooldown_until": 0,
            },
            {
                "name": "Groq",
                "type": "groq",
                "model_name": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                "api_key": os.getenv("GROQ_API_KEY", ""),
                "cooldown_until": 0,
            },
            {
                "name": "OpenRouter (free)",
                "type": "openrouter",
                "model_name": os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
                "api_key": os.getenv("OPENROUTER_API_KEY", ""),
                "cooldown_until": 0,
            },
            {
                "name": "Ollama (Local)",
                "type": "ollama",
                "model_name": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                "api_key": "local",  # no key
                "cooldown_until": 0,
            },
        ]

        # make sure openrouter is only using free model
        self.models_config = [
            c for c in self.models_config
            if not (c["type"] == "openrouter" and not c["model_name"].endswith(":free"))
        ]
        self.current_index = 0

    def get_active_model(self):
        now = time.time()
        total = len(self.models_config)
        for _ in range(total):
            cfg = self.models_config[self.current_index]
            if cfg["api_key"].strip() and now > cfg["cooldown_until"]:
                return cfg, self.instantiate_llm(cfg)
            self.current_index = (self.current_index + 1) % total

        # all model on cooldown, select the one that will finish its cooldown the soonest.
        cfg = min(self.models_config, key=lambda c: c["cooldown_until"])
        self.current_index = self.models_config.index(cfg)
        return cfg, self.instantiate_llm(cfg)

    def mark_limit_reached(self, cfg, cooldown_seconds=60):
        cfg["cooldown_until"] = time.time() + cooldown_seconds
        self.current_index = (self.current_index + 1) % len(self.models_config)

    def instantiate_llm(self, cfg):
        t = cfg["type"]
        if t == "gemini":
            return ChatGoogleGenerativeAI(
                model=cfg["model_name"], google_api_key=cfg["api_key"], temperature=0.1
            )
        elif t == "groq":
            return ChatGroq(
                model_name=cfg["model_name"], groq_api_key=cfg["api_key"], temperature=0.1
            )
        elif t == "openrouter":
            return ChatOpenAI(
                model_name=cfg["model_name"],
                openai_api_key=cfg["api_key"],
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
            )
        elif t == "ollama":
            return ChatOllama(model=cfg["model_name"], temperature=0.1)
        raise ValueError(f"Unknown model type: {t}")

# 2.5 AGENT LOOP ---------------------------

SYSTEM_PROMPT = "คุณคือ Vibe Coding AI Agent ชื่อ DearEngine ซึ่งผู้สร้างคือ Phanyakorn ช่วยสร้างและแก้โค้ดในเครื่องของผู้ใช้โดยใช้ Tools อย่างระมัดระวัง"

def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)

def run_agent(llm, user_input: str, max_steps: int = 10) -> str:
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]

    for _ in range(max_steps):
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            return content_to_text(ai.content) or "(no response)"

        for call in ai.tool_calls:
            tool_fn = tool_map.get(call["name"])
            if tool_fn is None:
                result = f"Unknown tool: {call['name']}"
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stop! เรียก tool ครบจำนวนสูงสุดแล้ว ลองสั่งงานให้เล็กลง"

# 3. BANNER (ANSI Shadow) -------------------

GLYPHS = {
    "D": ["██████╗ ", "██╔══██╗", "██║  ██║", "██║  ██║", "██████╔╝", "╚═════╝ "],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
    "N": ["███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"],
    "G": [" ██████╗ ", "██╔════╝ ", "██║  ███╗", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
}

BANNER_COLORS = [THEME["lightest"], THEME["lighter"], THEME["light"],
                 THEME["base"], THEME["dark"], THEME["darker"]]

def render_word(word: str) -> list:
    return ["".join(GLYPHS[ch][row] for ch in word) for row in range(6)]

def print_banner():
    dear, engine = render_word("DEAR"), render_word("ENGINE")
    one_line = [f"{d}    {e}" for d, e in zip(dear, engine)]
    width = len(one_line[0])

    if console.width >= width + 2:
        blocks = [one_line]
    else:
        blocks = [dear, engine]

    console.print()
    for block in blocks:
        for row, color in zip(block, BANNER_COLORS):
            console.print(Text(row, style=f"bold {color}", no_wrap=True, overflow="crop"))
    console.print()

# 4. CLI RUNNER ----------------------------

def main():
    pool = AIModelPool()

    print_banner()
    console.print(Panel.fit(
        f"[bold {THEME['light']}]DearEngine Infinite Vibe Coder Ready![/bold {THEME['light']}]\n"
        f"[{THEME['dark']}]Type your prompt into the input box below. | Type 'exit' to close DearEngine[/{THEME['dark']}]",
        border_style=THEME["base"]
    ))

    while True:
        try:
            user_input = prompt("VibeCoder > ", style=Style.from_dict({'': f"{THEME['light']} bold"}))

            if not user_input.strip():
                continue
            if user_input.strip().lower() in ['exit', 'quit']:
                console.print(f"[{THEME['lighter']}]Bye! Happy Coding![/{THEME['lighter']}]")
                break

            console.print(f"\n[bold {THEME['light']}]User:[/bold {THEME['light']}] {user_input}")

            success = False
            for _ in range(len(pool.models_config) * 2):
                cfg, llm = pool.get_active_model()
                console.print(f"[{THEME['dark']}]>>> Using Model: {cfg['name']} ({cfg['model_name']})...[/{THEME['dark']}]")

                try:
                    with console.status(f"[bold {THEME['lighter']}]AI is thinking & coding...[/bold {THEME['lighter']}]",
                                        spinner_style=THEME["base"]):
                        output = run_agent(llm, user_input)

                    console.print(Panel(
                        Markdown(output),
                        title=f"[bold {THEME['lighter']}]DearEngine ({cfg['name']})[/bold {THEME['lighter']}]",
                        border_style=THEME["base"]
                    ))
                    success = True
                    break

                except Exception as e:
                    err_msg = str(e)
                    low = err_msg.lower()
                    if "429" in err_msg or "quota" in low or "rate limit" in low:
                        console.print(f"[bold yellow]{cfg['name']} reached limit! Switching model...[/bold yellow]")
                        pool.mark_limit_reached(cfg, cooldown_seconds=60)
                    else:
                        console.print(f"[bold red]Error ({cfg['name']}):[/bold red]\n{err_msg}")
                        # Requests that fail for reasons other than rate limits (e.g., 404, 400, or connection failures) will be paused for a longer period to avoid repeated attempts.
                        pool.mark_limit_reached(cfg, cooldown_seconds=300)

            if not success:
                console.print("[bold red]All models hit a limit or encountered an error.[/bold red]\n")

        except KeyboardInterrupt:
            console.print(f"\n[{THEME['lighter']}]Exit program.[/{THEME['lighter']}]")
            break

if __name__ == "__main__":
    main()