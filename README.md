# ♠️♦️ Winadim - Poker Game State Analyzer ♥️♣️

**Winadim** is an intelligent assistant for playing online poker on Winamax. You take a screenshot of your table, and it extracts visual data (cards, bets, stacks, etc.), deduces all the information about the current table state, and send a request to OpenAI’s API such that the GPT model gives advice about the best move to perform as well as the reasons behind it. 

---

## ✨ Features

- 🖥️ GUI interface to capture and analyze poker screen
- 🔍 OCR using EasyOCR to extract numerical and textual data
- 🤖 GPT integration to describe game situations and recommend analysis
- 🃏 Card recognition using NumPy-based template matching
- 🧪 Built-in test suite with sample screenshots
- 📸 `examples/` folder (planned): use-case screenshots during real games

---

## ⚙️ Important Notes & Customization

- ⚠️ **GPU Required**  
  This application **requires a GPU** to function properly.  
  Components like OCR (EasyOCR) and some image processing steps are GPU-accelerated, and may **not work at all** or will be extremely slow on CPU-only systems.  

  You can still remove the imports related to torch and torchvision in the `requirements.txt` but you'll need them to use the GPU. 

  You can also change cuda distribution (see torch official website) to install based on your current device.

- 🌐 **Instruction Language**  
  By default, the initial GPT instruction (defined in `main_gui.py`) asks the assistant to answer **in French**.  
  You can change this prompt to English (or any language you prefer) by modifying the instruction string.

- 🧠 **Reasoning vs Speed**  
  The GPT prompt also asks the model to **explain why a move was chosen**.  
  If you'd like faster results and are okay with just the move recommendation (without the reasoning being explained), 
  you can remove that part of the instruction.  
  This will significantly reduce response time at the cost of interpretability.

- 🔎 **Performance and relaibility**
  While fun and intuitive, asking chatGPT about the move to do does not guarantee great performance.
  He is no expert and can be wrong and misleading more than once so keep being cautious. 
  Further discussion in the Future Directions section.

  Stay cautious and informed during gameplay.  
  Further discussion can be found in the **Future Directions** section.

→ Support for cpu-only usage and offering language and explanation options in the GUI are planned features.

---

## ⚠️ Limitations

This tool currently works only with a specific visual style of poker game (e.g., table layout, card design, and UI structure).  
If you're using a different theme or card pack, results may be inaccurate or inconsistent.
This is the drawback of using template matching based on pixel locations or color, this is fast as hell but not very robust.  

→ Support for multiple styles is a planned feature.

---

## 📸 Screenshot to Intelligence Flow

1. User presses a shortcut (space bar) to trigger screen capture and can exit the app with esc.
2. OCR detects cards, bets, stacks, and other visual elements, which are then used to compute the full current state of the poker table.
3. Data is formatted into structured input for OpenAI API.
4. Sent to OpenAI GPT for grounded advice about the best move to perform.

---

## 🚀 Getting Started

### 📦 1. Clone the Repo

```bash
git clone https://github.com/yourusername/winadim.git
cd winadim
```

### 🐍 2. Create Environment & Install Requirements

➤ Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

➤ Install dependencies

```bash
pip install -r requirements.txt
```

### 🔑 3. Set Your OpenAI Key

Open the file gpt_interface.py and replace the placeholder string with your own OpenAI API key

# gpt_interface.py

```python
# === OpenAI API ===
_OPENAI_KEY = 'my_openai_key'  # 🔁 Replace this with your actual OpenAI key
```

### 🟢 4. Run the App

From the `Winadim` folder, launch the GUI:

```bash
cd Winadim
python main_gui.py
```

## Project Structure

```lua
Winadim/
├── constants.py
├── gpt_interface.py
├── image_processing.py
├── main_gui.py
├── prepare_data.py
├── constants_data/
│   ├── card_rank_masks.npy
│   └── card_symbol_masks.npy
├── tests/
│   └── tests.py
├── examples/          <-- (planned) game screenshots for demo
└── README.md
```

## 🧠 Tech Stack

### 🐍 Python 3.10+ (tested with 3.10.11 and 3.13.1)

### 🧠 OpenAI GPT (via openai API)

### 🧾 EasyOCR for text extraction

### 🧰 NumPy & OpenCV for image processing

### 🖼️ PyQt5 for GUI

### 🎹 Pynput + keyboard for detecting key press events

### 🧪 Pytest for testing

## 🧪 Running Tests

Unit tests are in the `tests/` folder and use `pytest`
Screenshots are available in handscreenshots folder for you 
to validate game state detection. Just change the path used
inside `tests.py`

## 🚀 Future Directions

In addition to the planned features mentioned above, recent research has shown that **fine-tuning a large language model (LLM)** can significantly enhance its performance as a poker assistant.

💡 Instead of relying on ChatGPT, a dedicated LLM could be trained on poker-specific data to better understand in-game situations and deliver more reliable and context-aware recommendations.

📄 This idea is supported by the findings in the paper [*Playing Texas Hold'em Poker with a Fine-tuned Language Model* (2024)](https://arxiv.org/abs/2401.06781), which demonstrates how domain-specific fine-tuning can greatly improve the quality of move suggestions and strategic awareness.

🔁 An alternative — possibly easier to implement and even more robust — would be to use all the information computed about the current poker table state **not to query an LLM**, but to plug directly into a **solver** designed to return **game-theoretic optimal decisions**.  
Such a solution would prioritize correctness and exploitability minimization over natural language reasoning.

✨ These two directions — fine-tuned LLMs and solver-based reasoning — open the door to a more powerful and reliable poker assistant.