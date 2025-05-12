import numpy
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
CONSTANTS_DIR = _THIS_DIR / "constants_data"

# Load .npy files
_CARD_RANK_MASKS = numpy.load(CONSTANTS_DIR / "card_rank_masks.npy", allow_pickle=False)
_CARD_SYMBOL_MASKS = numpy.load(CONSTANTS_DIR / "card_symbol_masks.npy", allow_pickle=False)

_RANK_INDICES_MAPPING = {
    0: 'A',
    1: '2',
    2: '3',
    3: '4',
    4: '5',
    5: '6',
    6: '7',
    7: '8',
    8: '9',
    9: '10',
    10: 'J',
    11: 'Q',
    12: 'K'
}

_SYMBOL_INDICES_MAPPING = {
    0: '♦',
    1: '♠',
    2: '♣',
    3: '♥'
}
