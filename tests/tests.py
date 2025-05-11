import os
import sys
# === Fix path and initialize objects ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from prepare_data import PokerState
from openai import OpenAI
import easyocr

_SCREENSHOT_TEST_PATH = "handscreenshots/9.jpeg" # NOTE: Update to screenshot path of your choice.
# Path might depend on where (which directory) you execute the code from. Here it was executed from
# Windadim folder, containing handscreenshots folder and tests folders at the same level, the current
# testing script being located in tests folder.

def test_pokerstate_image_analysis() -> None:
    """
    Runs a series of tests on the PokerState class to verify its image-based
    analysis functionality.

    Loads a sample poker screenshot and runs all key analysis methods:
    - Extracting table cards
    - Detecting cards back (unfolded players)
    - Identifying dealer position
    - Detecting player stack presence and all-in state
    - Extracting bets from crop regions
    - Extracting Player 1's cards using OCR

    Output is printed to the console for manual verification.

    Raises:
        Any unexpected exceptions from internal PokerState methods are caught and printed.
    """
    img = Image.open(_SCREENSHOT_TEST_PATH)
    reader = easyocr.Reader(['en'], gpu=True)

    state = PokerState(image=img, ocr_reader=reader)

    print("\n[TEST] extract_table_cards")
    table_cards = state.extract_table_cards()
    print(f"Table cards: {table_cards}")

    print("\n[TEST] get_cards_back_presence")
    cards_back = state.get_cards_back_presence()
    for i, unfolded in enumerate(cards_back, start=2):
        print(f"Player {i} has unfolded: {unfolded}")

    print("\n[TEST] get_dealer_index")
    dealer = state.get_dealer_index()
    print(f"Dealer: Player {dealer + 1}")

    print("\n[TEST] set_player_stacks + player.presence")
    state.set_player_stacks()
    state.set_players_presence_from_stack()
    for p in state.players:
        print(f"Player {p.index + 1} present: {p.presence}, stack: {p.stack}, all-in: {p.has_all_in}")

    print("\n[TEST] get_bet_crops + get_bets")
    present = [p.index for p in state.players if p.presence == "present" and p.index != 0]
    bet_crops = state.get_bet_crops(present)
    bets = state.get_bets(bet_crops)
    for i, bet in zip(present, bets):
        print(f"Player {i + 1}: {'bet' if bet else 'no bet'}")

    print("\n[TEST] extract_player1_cards")
    try:
        cards = state.extract_player1_cards()
        print(f"Player 1 cards: {cards}")
    except Exception as e:
        print(f"Failed to extract Player 1 cards: {e}")

if __name__ == "__main__":
    print("Running PokerState image analysis tests...")
    test_pokerstate_image_analysis()
    print("All tests completed.")
