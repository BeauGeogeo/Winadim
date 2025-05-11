
import sys
import pyautogui
from threading import Thread
from pynput import keyboard
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QVBoxLayout
from PyQt5.QtGui import QFont, QCloseEvent
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
import openai
from openai import OpenAI
import easyocr
from easyocr import Reader
from PIL.Image import Image
from typing import Union, Optional

from gpt_interface import run_gpt_completion
from prepare_data import PokerState

# === OpenAI API ===
_OPENAI_KEY = 'my_openai_key' # Replace with your actual key

# NOTE: yo ucan modify the prompt below such as to skip the justification part,
# jumping directly to the answer. It makes OpenAI API answer much faster, but
# you will get only the recommended move, not the reasons behind it.

# === System Message ===
initial_instruction = {
    "role": "system",
    "content": (
        "You are an expert poker assistant embedded in a poker-playing AI system. "
        "The game is played on the **Winamax app**, in a **No-Limit Texas Hold'em** format at a **5-max table** (up to 5 players). "
        "All players are humans. You are assisting **Player 1**, the user of this system (also called the hero), who is actively seeking help in decision-making.\n\n"
        "You receive, for each game snapshot, structured **textual** inputs about the table state and players.\n\n"
        "**Key data format:**\n"
        "1. The **table message** includes:\n"
        "   - `'preflop'` or `'postflop'` phase\n"
        "   - If `'preflop'`, no community cards are included\n"
        "   - If `'postflop'`, between 3 and 5 community cards are provided (e.g., `'J♠'`, `'9♥'`, ...)\n"
        "   - Pot information in BB (e.g., `'Pot 30 and Pot total 50'`)\n\n"
        "2. Each **Player N** (from 1 to 5):\n"
        "   - `Status`: `'present'` or `'absent'`\n"
        "   - If present:\n"
        "     - `Move`: `'B'`, `'B-ALLIN'`, `'C'`, `'F'`, or `'NP'`\n"
        "     - `Position`: `'D'`, `'SB'`, `'BB'`, or `'Other'`\n"
        "     - `Stack`: current stack in BB (e.g., `'75 BB'`) unless all-in\n"
        "     - `Bet`: amount of the current bet (if applicable)\n"
        "     - For **Player 1 only**, their two hole cards are given (e.g., `'A♠, K♣'`)\n\n"
        "**Your task:**\n"
        "- Analyze the full poker situation based solely on the data\n"
        "- Recommend the **best move for Player 1 (the hero)**: **Fold**, **Call**, or **Raise**\n"
        "- Your response must be **a single line in FRENCH**, clearly stating the recommended move. "
        "- Add on a newline, after a blank space, a sound explanation for the recommanded move (in french) mentionning precisely the elements"
        "you base your decision on."
        "Only the action itself should be in English (Fold, Call, Raise).\n"
        "- Do **not** include explanation, reasoning, or summary.\n"
        "- Use **only** the given structured input. Do not make assumptions beyond it.\n\n"
        "**Example output:**\n"
        "→ Je recommande de Raise.\n"
    )
}

# === Worker Threads ===

class GPTWorker(QThread):
    """
    Worker thread for running GPT completions without blocking the main UI thread.

    Emits:
        finished (str): Emitted with the GPT response or an error message once the run is complete.
    """
    finished = pyqtSignal(str)

    def __init__(self, 
                 client: OpenAI, 
                 system_prompt: str, 
                 user_prompt: str) -> None:
        """
        Initializes the GPTWorker.

        Args:
            client (OpenAI): An instance of the OpenAI client used to perform the completion.
            system_prompt (str): The system-level instructions guiding GPT behavior.
            user_prompt (str): The user's input prompt.
        """
        super().__init__()
        self.client = client
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def run(self) -> None:
        """
        Executes the GPT completion in a separate thread and emits the result.
        """
        try:
            result = run_gpt_completion(self.client,
                                        self.system_prompt, 
                                        self.user_prompt)
        except Exception as e:
            result = f"Error: {e}"
        self.finished.emit(result)

class OCRWorker(QThread):
    """
    Worker thread for performing OCR and preparing game data without blocking the UI.

    Emits:
        finished (PokerState | None, str): The parsed PokerState object and prompt, or error message.
    """
    finished = pyqtSignal(object, str)

    def __init__(self, screenshot: Image, ocr_reader: Reader) -> None:
        """
        Initializes the OCR worker thread.

        Args:
            screenshot (Image): Screenshot of the game area for OCR.
            ocr_reader (Reader): An instance of EasyOCR reader.
        """
        super().__init__()
        self.screenshot = screenshot
        self.ocr_reader = ocr_reader

    def run(self) -> None:
        """
        Executes OCR and prepares the prompt, emitting the result upon completion.
        """
        try:
            poker_state = PokerState(image=self.screenshot, ocr_reader=self.ocr_reader)
            player_msgs, table_msg = poker_state.prepare_game_data()

            lines = []
            for player in player_msgs:
                lines.extend([item["text"] for item in player["content"]])
            lines.extend([item["text"] for item in table_msg["content"]])
            final_prompt = "\n".join(lines)

            self.finished.emit(poker_state, final_prompt)
        except Exception as e:
            self.finished.emit(None, f"Error during OCR processing: {e}")

# === Communicator ===

class Communicator(QObject):
    """
    A communication helper class to emit signals between threads and GUI components.

    Signals:
        update_text (str): Emitted when text needs to be updated in the GUI.
        trigger_screenshot (): Emitted to signal that a screenshot should be taken.
    """
    update_text = pyqtSignal(str)
    trigger_screenshot = pyqtSignal()

    def __init__(self) -> None:
        """
        Initializes the Communicator with no additional setup.
        """
        super().__init__()

# === GUI ===

class PokerAssistantGUI(QWidget):
    """
    The main GUI window for the Poker Assistant application.
    Handles OCR, GPT integration, and user interactions.
    """
    def __init__(self, client: OpenAI) -> None:
        """
        Initializes the PokerAssistantGUI with OCR capabilities and GPT integration.

        Args:
            client (OpenAI): An instance of the OpenAI client for GPT completions.
        """
        super().__init__()
        self.client = client
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        self.is_closing = False
        self.processing = False

        self.communicator = Communicator()
        self.communicator.update_text.connect(self.show_message)
        self.communicator.trigger_screenshot.connect(self._trigger_analysis_safe)

        self.init_ui()

    def init_ui(self) -> None:
        """
        Initializes and configures the main GUI layout, widgets, and window properties.
        """
        self.setWindowTitle("Poker Assistant")
        self.setGeometry(100, 100, 600, 400)
        assert hasattr(Qt, "WindowStaysOnTopHint") # for mypy typing
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        self.text_area.setFont(QFont("Courier New", 11))

        layout = QVBoxLayout()
        layout.addWidget(self.text_area)
        self.setLayout(layout)

        self.show()
        self.update_text("Waiting for input...")

    def trigger_analysis(self) -> None:
        """
        Initiates the analysis process if the GUI is active and not already processing.
        Emits signals to update the UI and trigger a screenshot.
        """
        if self.is_closing or self.processing:
            return
        self.processing = True
        self.communicator.update_text.emit("Processing...")
        self.communicator.trigger_screenshot.emit()


    def _trigger_analysis_safe(self) -> None:
        """
        Safely triggers analysis by briefly hiding the GUI and scheduling screenshot capture.
        Used to ensure the window isn't included in the screenshot.
        """
        self.hide()
        QTimer.singleShot(200, self.take_screenshot_then_start_processing)

    def take_screenshot_then_start_processing(self) -> None:
        """
        Captures a screenshot, shows the GUI again, and starts the OCR processing
        in a background thread using OCRWorker.
        """
        screenshot = pyautogui.screenshot()
        self.show()

        self.ocr_worker = OCRWorker(screenshot, self.ocr_reader)
        self.ocr_worker.finished.connect(self.handle_ocr_result)
        self.ocr_worker.start()

    def handle_ocr_result(self, poker_state: Union[PokerState, None], prompt_or_error: str) -> None:
        """
        Handles the result from the OCRWorker. If OCR failed, displays the error.
        Otherwise, starts the GPTWorker to process the generated prompt.

        Args:
            poker_state (PokerState | None): The parsed poker state object, or None on error.
            prompt_or_error (str): Either the generated prompt text or an error message.
        """
        if poker_state is None:
            self.communicator.update_text.emit(prompt_or_error)
            self.processing = False
            return

        self.gpt_worker = GPTWorker(self.client,
                                    initial_instruction["content"], 
                                    prompt_or_error)
        self.gpt_worker.finished.connect(self._handle_gpt_result)
        self.gpt_worker.start()


    def _handle_gpt_result(self, response: str) -> None:
        """
        Handles the result from GPTWorker by updating the UI and resetting the processing state.

        Args:
            response (str): The assistant's reply generated by GPT.
        """
        self.communicator.update_text.emit(response)
        self.processing = False


    def show_message(self, text: str) -> None:
        """
        Displays the given message in the text area and ensures 
        the GUI is fully brought to the foreground.

        Args:
            text (str): The message to display in the text area.
        """
        self.text_area.setPlainText(text)
        assert hasattr(Qt, "WindowNoState")
        self.setWindowState(Qt.WindowNoState)
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.repaint()
        QApplication.processEvents()

    def update_text(self, content: str) -> None:
        """
        Updates the text area with the provided content.

        Args:
            content (str): The text to display in the UI text area.
        """
        self.text_area.setPlainText(content)

    def closeEvent(self, event: Optional[QCloseEvent]) -> None:
        """
        Handles the window close event by marking the application as closing
        and initiating a clean shutdown of the Qt application.

        Args:
            event (QCloseEvent): The close event triggered by the user.
        """
        self.is_closing = True
        QApplication.quit()

# === Background Key Listener ===

def start_key_listener(gui_ref: PokerAssistantGUI) -> None:
    """
    Starts a background keyboard listener that triggers GUI actions based on key presses.

    Pressing SPACE triggers analysis.
    Pressing ESC quits the application.

    Args:
        gui_ref (Any): A reference to the GUI instance with `trigger_analysis` and `is_closing` attributes.
    """
    def on_press(key: keyboard.Key) -> bool | None:
        """
        Handles key press events for controlling the GUI.

        - SPACE triggers the analysis process.
        - ESC sets the GUI to closing state and quits the application.

        Args:
            key (keyboard.Key): The key event captured by the listener.

        Returns:
            bool | None: Returns False to stop the listener (on ESC), or None to continue.
        """
        try:
            if key == keyboard.Key.space:
                gui_ref.trigger_analysis()
            elif key == keyboard.Key.esc:
                gui_ref.is_closing = True
                QApplication.quit()
                return False  # Stop the listener
        except Exception as e:
            print(f"[ERROR] Key listener: {e}")
        return None  # Added to satisfy all code paths for mypy typing

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

# === Main Function ===

def main() -> None:
    """
    Entry point for the Poker Assistant application.

    - Initializes the OpenAI client.
    - Sets up the Qt application and main GUI window.
    - Starts a background thread to listen for keyboard shortcuts.
    - Begins the Qt event loop.

    Returns:
        None
    """
    client = openai.OpenAI(api_key=_OPENAI_KEY)
    app = QApplication(sys.argv)
    gui = PokerAssistantGUI(client)

    key_thread = Thread(target=start_key_listener, args=(gui,), daemon=True)
    key_thread.start()

    app.exec_()

if __name__ == "__main__":
    main()
