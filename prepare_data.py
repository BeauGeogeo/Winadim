from PIL.Image import Image
from io import BytesIO
import numpy
import cv2
from dataclasses import dataclass, field
from typing import ClassVar, Literal, Optional
from easyocr import Reader
from constants import _CARD_RANK_MASKS, _CARD_SYMBOL_MASKS, _RANK_INDICES_MAPPING, _SYMBOL_INDICES_MAPPING
from image_processing import find_best_match, to_ink_mask, apply_diagonal_mask_top_left
import time 

@dataclass
class PlayerInfo:
    """
    Represents the game-related state of a single poker player.

    Attributes:
        index (int): Player index (0-based).
        presence (Literal["present", "absent"]): Whether the player is currently seated.
        stack (str): Stack size as detected from OCR, typically in BB.
        bet_amount (str): Amount of the current bet, if any.
        move (Optional[str]): The current move made by a player ('all-in', 'bet', 'check', 'fold', 
        and 'NP' if he has not played yet).
        position (Optional[str]): Table position ('D', 'SB', 'BB', 'Other').
        has_all_in (bool): Flag indicating if the player is currently all-in.
    """
    index: int
    presence: Literal["present", "absent"]
    stack: str = ''
    bet_amount: str = ''
    move: Optional[str] = None
    position: Optional[str] = None
    has_all_in: bool = field(default=False)

class PokerState:
    """
    Represents the state of a poker game based on image-based OCR and template matching.

    Attributes:
        image (Image): The screenshot or input image from which the game state is parsed.
        ocr_reader (Reader): EasyOCR reader instance used to extract textual data.
        preflop (bool): Whether the current game phase is preflop (no community cards yet).
        players (list[PlayerInfo]): List of player information objects, one per seat.
    """

    max_nb_players: ClassVar[int] = 5

    stack_regions: ClassVar[list[dict]] = [
        {"left": 793, "top": 800, "width": 273, "height": 44}, # NOTE: updated and move to the right as on the left it was taking timer value...
        {"left": 262, "top": 637, "width": 273, "height": 44},
        {"left": 452, "top": 187, "width": 273, "height": 44},
        {"left": 1080, "top": 187, "width": 273, "height": 44},
        {"left": 1278, "top": 635, "width": 273, "height": 44}
    ]

    bet_regions: ClassVar[list[dict]] = [
        {"left": 807, "top": 653, "width": 190, "height": 30},
        {"left": 469, "top": 575, "width": 190, "height": 30},
        {"left": 611, "top": 320, "width": 190, "height": 30},
        {"left": 1009, "top": 319, "width": 190, "height": 30},
        {"left": 1159, "top": 574, "width": 190, "height": 30}
    ]

    card_back_regions: ClassVar[list[dict]] = [
        {"left": 322, "top": 522, "width": 147, "height": 86},
        {"left": 516, "top": 75,  "width": 149, "height": 86},
        {"left": 1144, "top": 78, "width": 149, "height": 84},
        {"left": 1341, "top": 524, "width": 146, "height": 82}
    ]

    button_regions: ClassVar[list[dict]] = [
        {"left": 745, "top": 653, "width": 49, "height": 46},
        {"left": 452, "top": 444, "width": 49, "height": 46},
        {"left": 746, "top": 235, "width": 49, "height": 46},
        {"left": 1238, "top": 272, "width": 49, "height": 46},
        {"left": 1239, "top": 615, "width": 49, "height": 46}
    ]

    pot_region: ClassVar[dict] = {"left": 832, "top": 555, "width": 259, "height": 30}
    table_card_base: ClassVar[dict] = {"left": 618, "top": 352, "width": 112, "height": 164}
    table_shift: ClassVar[int] = table_card_base["width"] + 2
    # player1_cards: ClassVar[dict] = {"left": 828, "top": 688, "width": 150, "height": 86} NOT USEFUL ANYMORE NORMALLY
    # === Table rank (value) crops ===
    table_rank_base: ClassVar[dict] = {"left": 622, "top": 354, "width": 36, "height": 41}
    table_rank_shift: ClassVar[int] = 114  # 106 width + 8 gap

    # === Player 1 hand rank (value) crops ===
    player1_rank_base: ClassVar[dict] = {"left": 830, "top": 689, "width": 36, "height": 41}
    player1_rank_shift: ClassVar[int] = 43  # Distance from mask1's right to card2's left

    # === Table symbol (suit) crops ===
    table_symbol_base: ClassVar[dict] = {"left": 622, "top": 407, "width": 35, "height": 31}
    table_symbol_shift: ClassVar[int] = 114  # 106 card width + 8 gap

    # === Player 1 hand symbol (suit) crops ===
    player1_symbol_base: ClassVar[dict] = {"left": 830, "top": 741, "width": 35, "height": 31}
    player1_symbol_shift: ClassVar[int] = 43  # Mask 1 to Card 2 symbol left edge

    card_rank_masks: ClassVar[numpy.ndarray] = _CARD_RANK_MASKS
    card_symbol_masks: ClassVar[numpy.ndarray] = _CARD_SYMBOL_MASKS
    rank_indices_mapping: dict[int, str] = _RANK_INDICES_MAPPING
    symbol_indices_mapping: dict[int, str] = _SYMBOL_INDICES_MAPPING

    rank_threshold: ClassVar[int] = 240 # FIXME: not good threshold.
    symbol_threshold: ClassVar[int] = 200

    def __init__(self, image: Image, ocr_reader: Reader) -> None:
        """
        Initializes the PokerState with image and OCR reader, sets preflop status, and prepares players.

        Args:
            image (Image): Screenshot of the poker table.
            ocr_reader (Reader): EasyOCR reader instance.
        """
        self.image = image
        self.ocr_reader = ocr_reader
        self.preflop: bool = True
        self.players: list[PlayerInfo] = [
            PlayerInfo(i, "absent") for i in range(self.max_nb_players)
        ]

    def crop_region(self, region: dict[str, int]) -> Image:
        """
        Crops a rectangular region from the current image using the provided bounding box.

        Args:
            region (dict[str, int]): A dictionary with keys 'left', 'top', 'width', and 'height'
                                    defining the rectangular region to crop.

        Returns:
            Image: A PIL Image object representing the cropped region.
        """
        box: tuple[int, int, int, int] = (
            region["left"],
            region["top"],
            region["left"] + region["width"],
            region["top"] + region["height"]
        )

        return self.image.crop(box)

    def has_stack(self, crop: Image, threshold: int = 30) -> numpy.bool_:
        """
        Detects yellow/orange text (e.g., '100 BB') in a cropped image.

        Args:
            crop (Image): Cropped image containing the stack value text.
            threshold (int): Minimum number of yellow/orange pixels to confirm presence.

        Returns:
            bool: True if yellow/orange stack text is detected, False otherwise.
        """
        color_crop = crop.convert("RGB")
        arr = numpy.array(color_crop)

        yellow_mask = (
            (arr[:, :, 0] >= 200) & (arr[:, :, 0] <= 255) &  # Red
            (arr[:, :, 1] >= 150) & (arr[:, :, 1] <= 197) &  # Green
            (arr[:, :, 2] >= 0)   & (arr[:, :, 2] <= 90)     # Blue
        )

        yellow_pixel_count = numpy.sum(yellow_mask)
        has_enough_yellow = yellow_pixel_count > threshold

        return has_enough_yellow

    def has_yellow_bet(self, crop: Image, threshold: int = 30) -> numpy.bool_:
        """
        Detects the presence of bet amount based on the (yellow) pixel color analysis.

        Args:
            crop (Image): Cropped image from the expected bet zone.
            threshold (int): Minimum number of yellow-like pixels required to confirm presence.

        Returns:
            bool: True if a yellow bet label is detected, False otherwise.
        """
        arr = numpy.array(crop.convert("RGB"))
        yellow_mask = (
            (arr[:, :, 0] >= 200) & (arr[:, :, 0] <= 255) &  # Red
            (arr[:, :, 1] >= 170) & (arr[:, :, 1] <= 240) &  # Green
            (arr[:, :, 2] >= 0)   & (arr[:, :, 2] <= 100)    # Blue
        )
        yellow_pixel_count = numpy.sum(yellow_mask)
        has_enough_yellow = yellow_pixel_count > threshold

        return has_enough_yellow

    def has_unfolded(self, crop: Image, threshold: int = 100) -> numpy.bool_:
        """
        Determines whether a player has not folded based on red card back presence.

        Args:
            crop (PIL.Image): Cropped image of the card area.
            threshold (int): Minimum number of red pixels to confirm presence.

        Returns:
            bool: True if the red card backs are present (not folded), else False.
        """
        color_crop = crop.convert("RGB")
        arr = numpy.array(color_crop)
        # Mask for bright/dark red area of card backs (excluding white border)
        red_mask = (
            (arr[:, :, 0] >= 200) & (arr[:, :, 0] <= 255) &
            (arr[:, :, 1] >= 25)  & (arr[:, :, 1] <= 60) &
            (arr[:, :, 2] >= 25)  & (arr[:, :, 2] <= 60)
        )
        red_pixel_count = numpy.sum(red_mask)
        has_enough_red = red_pixel_count > threshold

        return has_enough_red

    def has_card(self, crop: Image, threshold: int = 200) -> bool:
        """
        Determines whether a card is present in a cropped region by detecting white contours.

        This method works by identifying bright white areas (common to card backgrounds)
        and evaluating whether any significant contour areas are detected.

        Args:
            crop (Image): Cropped image of the card zone.
            threshold (int): Minimum contour area required to consider a card present.

        Returns:
            bool: True if at least one large white contour is detected, False otherwise.
        """
        arr = numpy.array(crop.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > threshold]

        return len(significant_contours) > 0

    def has_dealer_button(self, crop: Image, threshold=50) -> bool:
        """
        Detect if the dealer button (orange/yellow circular D) is present.
        
        Parameters:
            crop (PIL.Image): Cropped image of the suspected dealer button zone.
            threshold (int): Minimum number of yellow-orange pixels to confirm presence.
        
        Returns:
            bool: True if dealer button is detected, False otherwise.
        """
        arr = numpy.array(crop.convert("RGB"))
        # Define mask for bright yellow-orange tones
        mask = (
            (arr[:, :, 0] >= 200) &  # High red
            (arr[:, :, 1] >= 130) &  # Mid to high green
            (arr[:, :, 2] <= 100)    # Low blue
        )
        count = numpy.sum(mask)
        has_dealer = count > threshold

        return has_dealer
    
    def has_all_in(self, crop: Image, threshold:int = 40) -> numpy.bool_:
        """
        Detects the presence of ALL-IN red text in a cropped image.

        Args:
            crop (PIL.Image): cropped stack region.
            threshold (int): minimum number of red pixels to confirm ALL-IN.

        Returns:
            bool: True if ALL-IN red text detected, False otherwise.
        """
        arr = numpy.array(crop.convert("RGB"))
        red_mask = (
            (arr[:, :, 0] >= 170) & (arr[:, :, 0] <= 255) &
            (arr[:, :, 1] >= 0)   & (arr[:, :, 1] <= 60) &
            (arr[:, :, 2] >= 0)   & (arr[:, :, 2] <= 60)
        )
        red_pixel_count = numpy.sum(red_mask)
        has_enough_red = red_pixel_count > threshold

        return has_enough_red

    def get_card_rank(self, rank_crop: Image) -> str:
        """
        Matches a cropped card rank region against known rank templates.

        Applies preprocessing (diagonal masking, ink masking) and selects the
        best matching rank index using template matching.

        Args:
            rank_crop (Image): Cropped image of the card rank area.

        Returns:
            str: The matched rank character (e.g., 'A', 'K', '9').
        """
        arr = numpy.array(rank_crop)
        arr_with_diag_mask = apply_diagonal_mask_top_left(arr)
        arr_ink_mask = to_ink_mask(arr_with_diag_mask, self.rank_threshold)
        best_rank_match_index, _ = find_best_match(arr_ink_mask, self.card_rank_masks)
        card_rank = self.rank_indices_mapping[best_rank_match_index]

        return card_rank

    def get_card_symbol(self, symbol_crop: Image) -> str:
        """
        Matches a cropped card symbol region against known suit templates.

        Applies ink masking and selects the best matching suit index using template matching.

        Args:
            symbol_crop (Image): Cropped image of the suit symbol area.

        Returns:
            str: The matched suit character (e.g., '♠', '♦', 'h').
        """
        arr = numpy.array(symbol_crop)
        arr_ink_mask = to_ink_mask(arr, self.symbol_threshold)
        best_symbol_match_index, _ = find_best_match(arr_ink_mask, self.card_symbol_masks)
        card_symbol = self.symbol_indices_mapping[best_symbol_match_index]

        return card_symbol


    def get_card(self, rank_crop: Image, symbol_crop: Image) -> str:
        """
        Constructs a readable card string from cropped rank and suit regions.

        Args:
            rank_crop (Image): Cropped image of the card rank.
            symbol_crop (Image): Cropped image of the card suit.

        Returns:
            str: Combined card notation, e.g., 'A♥', '10♠'.
        """
        rank = self.get_card_rank(rank_crop)
        suit = self.get_card_symbol(symbol_crop)

        return f"{rank}{suit}"
    
    def get_dealer_index(self) -> int:
        """
        Detects and returns the player index (0-4) who has the dealer button.

        Returns:
            int: Index of the player with the dealer button.

        Raises:
            RuntimeError: If no dealer button is found in any player region.
        """
        for player_index, region in enumerate(self.button_regions):
            crop = self.crop_region(region)
            if self.has_dealer_button(crop):
                return player_index

        raise RuntimeError("Dealer button not found in any region.")

    # TODO: wrap in subfunctions
    def extract_table_cards(self) -> list[str]:
        """
        Detects and returns a list of up to 5 community cards.

        Cards are detected locally with pattern matching based on suit and rank.

        Returns:
            list[str]: A list of community cards in the format ['A♥', '7♦', ...].
                       Returns an empty list if no flop is detected (i.e., preflop).
        """
        cards = []

        for i in range(self.max_nb_players):
            # Compute main card crop (for presence check)
            card_region = {
                "left": self.table_card_base["left"] + i * self.table_shift,
                "top": self.table_card_base["top"],
                "width": self.table_card_base["width"],
                "height": self.table_card_base["height"]
            }
            card_crop = self.crop_region(card_region)

            if i == 0 and not self.has_card(card_crop):
                return []  # No cards on table: preflop
            
            elif i < 3 or self.has_card(card_crop):
                # === Crop rank region ===
                rank_region = {
                    "left": self.table_rank_base["left"] + i * self.table_rank_shift,
                    "top": self.table_rank_base["top"],
                    "width": self.table_rank_base["width"],
                    "height": self.table_rank_base["height"]
                }
                rank_crop = self.crop_region(rank_region)

                # === Crop suit region ===
                symbol_region = {
                    "left": self.table_symbol_base["left"] + i * self.table_symbol_shift,
                    "top": self.table_symbol_base["top"],
                    "width": self.table_symbol_base["width"],
                    "height": self.table_symbol_base["height"]
                }
                symbol_crop = self.crop_region(symbol_region)

                # === Get card info ===
                card = self.get_card(rank_crop, symbol_crop)
                cards.append(card)

        return cards

    def extract_player1_cards(self) -> list[str]:
        """
        Detects and returns Player 1's hand cards (rank + suit).
        
        Uses fixed image regions to extract the rank and suit of each of the two hole cards.

        Returns:
            list[str]: A list of two card strings (e.g., ['A♥', '9♣']).
        """
        cards = []
        for i in range(2):
            # === Crop rank ===
            rank_region = {
                "left": self.player1_rank_base["left"] + i * self.player1_rank_shift,
                "top": self.player1_rank_base["top"],
                "width": self.player1_rank_base["width"],
                "height": self.player1_rank_base["height"]
            }
            rank_crop = self.crop_region(rank_region)

            # === Crop symbol ===
            symbol_region = {
                "left": self.player1_symbol_base["left"] + i * self.player1_symbol_shift,
                "top": self.player1_symbol_base["top"],
                "width": self.player1_symbol_base["width"],
                "height": self.player1_symbol_base["height"]
            }
            symbol_crop = self.crop_region(symbol_region)

            # === Get symbolic card ===
            card = self.get_card(rank_crop, symbol_crop)
            cards.append(card)

        return cards

    # TODO: wrap in sub-functions
    def set_player_stacks(self) -> None:
        """
        Detects and sets each player's stack value based on predefined image regions.

        Uses OCR to read stack text for each player. If the player is all-in, sets the
        `has_all_in` flag accordingly. If no stack is detected, sets the value to an empty string.

        Returns:
            None
        """
        for player_index, region in enumerate(self.stack_regions):
            player = self.players[player_index]
            crop = self.crop_region(region)
            crop_arr = numpy.array(crop)

            if self.has_stack(crop):
                ocr_result = self.ocr_reader.readtext(crop_arr, detail=0)
                player.stack = ocr_result[0]
            elif self.has_all_in(crop):
                ocr_result = self.ocr_reader.readtext(crop_arr, detail=0)
                player.stack = ocr_result[0]
                player.has_all_in = True
            else:
                player.stack = ''


    def get_bet_crops(self, present_player_indices: list[int]) -> list[Image]:
        """
        Returns cropped image regions for each present player's betting area.

        Args:
            present_player_indices (list[int]): Indices of players currently active (excluding Player 1).

        Returns:
            list[Image]: List of cropped bet zones matching the order of present players.
        """
        bet_crops = []

        for player_index in present_player_indices:
            region = self.bet_regions[player_index]
            crop = self.crop_region(region)
            bet_crops.append(crop)

        return bet_crops


    def get_bets(self, bet_crops: list[Image]) -> list[Image | None]:
        """
        Detects which bet zones contain a yellow bet value and returns the matching crops.

        Args:
            bet_crops (list[Image]): List of cropped images of bet areas for present players.

        Returns:
            list[Image | None]: A list where each element is either the original crop (if a bet is detected)
                                or None (if no bet is present). Matches the order of input crops.
        """
        bets = []

        for crop in bet_crops:
            has_bet = self.has_yellow_bet(crop)
            bet_crop = crop if has_bet else None
            bets.append(bet_crop)

        return bets
    
    def set_bet_values(
        self,
        present_player_indices: list[int],
        bet_crops: list[Image],
        bets: list[Image | None]
    ) -> None:
        """
        Sets the textual bet amount for each present player based on OCR.

        Args:
            present_player_indices (list[int]): Indices of active players (excluding Player 1).
            bet_crops (list[Image]): Cropped bet regions corresponding to the present players.
            bets (list[bool]): Flags indicating whether a bet was detected for each player.

        Returns:
            None
        """
        for index, bet in enumerate(bets):
            present_player_index = present_player_indices[index]
            player = self.players[present_player_index]

            if bet:
                bet_crop = bet_crops[index]
                bet_crop_arr = numpy.array(bet_crop)
                bet_amount = self.ocr_reader.readtext(bet_crop_arr)
            else:
                bet_amount = ''

            player.bet_amount = bet_amount


    def get_cards_back_presence(self) -> list[numpy.bool_]:
        """
        Checks whether each player (from seat 2 to 5) has unfolded cards
        based on the presence of card back patterns.

        Returns:
            list[bool]: True for each player if their card back is detected (i.e., not folded).
        """
        results = []
        for region in self.card_back_regions:
            crop = self.crop_region(region)
            unfolded = self.has_unfolded(crop)
            results.append(unfolded)

        return results

    def set_players_presence_from_stack(self) -> None:
        """
        Updates each player's presence status based on whether a stack value exists.

        A player is marked as "present" if a stack value is set (non-empty),
        otherwise marked as "absent".

        Returns:
            None
        """
        for player in self.players:
            player.presence = "present" if player.stack else "absent"

    def set_player_positions(
        self,
        dealer_index: int,
        present_player_indices: list[int]
    ) -> None:
        """
        Assigns table positions (D, SB, BB, Other) to present players based on dealer index.

        Args:
            dealer_index (int): The index of the player with the dealer button.
            present_player_indices (list[int]): Ordered list of indices of players who are present.

        Returns:
            None

        Notes:
            If the dealer is not found in the present players list, no positions are assigned.
        """
        try:
            dealer_pos = present_player_indices.index(dealer_index)
        except ValueError:
            print("[WARNING] Dealer not in present players — skipping position assignment.")
            return

        num_present = len(present_player_indices)
        sb_index = present_player_indices[(dealer_pos + 1) % num_present]
        bb_index = present_player_indices[(dealer_pos + 2) % num_present]

        for player_index in range(self.max_nb_players):
            player = self.players[player_index]
            if player_index not in present_player_indices:
                player.position = "absent"
            elif player_index == dealer_index:
                player.position = "D"
            elif player_index == bb_index:
                player.position = "BB"
            elif player_index == sb_index:
                player.position = "SB"
            else:
                player.position = "Other"

    def get_move(
        self,
        player_index: int,
        bet: Optional[Image],
        cards_back: numpy.bool_,
        dealer_index: int,
        present_player_indices: list[int]
    ) -> str:
        """
        Determines the move of a given player based on current game phase and position.

        Preflop logic is based on whether the player has acted after the big blind.
        Postflop logic checks if the player has acted after the dealer.
        If the player has not yet acted, the move is marked as 'NP' (Not Played).

        Args:
            player_index (int): Index of the player being evaluated.
            bet (Image | None): Cropped bet image if a bet was detected, otherwise None.
            cards_back (bool): Whether the player has cards visible (i.e., not folded).
            dealer_index (int): Index of the dealer.
            present_player_indices (list[int]): List of all present players' indices 
            (except the hero player at index 0, that is the one requesting the help of 
            the API for executing his move now).

        Returns:
            str: The inferred move — one of ['all-in', 'bet', 'check', 'fold', 'NP', 'unknown'].
        """
        assert player_index in present_player_indices, "You're requesting a move for an absent player."

        try:
            dealer_pos = present_player_indices.index(dealer_index)
            player_pos = present_player_indices.index(player_index)
        except ValueError:
            print("[WARNING] Dealer or player not in present players")
            return 'unknown'

        if self.preflop:
            # Preflop: players after big blind have acted
            bb_pos = (dealer_pos + 2) % len(present_player_indices)
            has_played = player_pos > bb_pos
        else:
            # Postflop: players after dealer have acted
            has_played = player_pos > dealer_pos

        player_move = self._move_from_state(player_index, bet, cards_back) if has_played else 'NP'

        return player_move

    def _move_from_state(
        self,
        player_index: int,
        bet: Optional[Image],
        cards_back: numpy.bool_
    ) -> str:
        """
        Infers the move of a player based on whether a bet or cards are detected.

        Args:
            player_index (int): Index of the player to evaluate.
            bet (Image | None): Bet image if detected, otherwise None.
            cards_back (bool): True if player has not folded (cards still shown).

        Returns:
            str: Inferred move — one of ['all-in', 'bet', 'check', 'fold'].
        """
        if bet is not None:
            player = self.players[player_index]
            return 'all-in' if player.has_all_in else 'bet'
        elif cards_back:
            return 'check'
        else:
            return 'fold'
        
    def _encode_move(self, move: str) -> str:
        """
        Encodes a player's move into a standardized string format for messaging.

        Args:
            move (str): The raw move string (e.g., 'bet', 'check', 'fold', 'NP', 'all-in').

        Returns:
            str: Encoded representation — e.g., 'B', 'C', 'F', 'NP', 'B-ALLIN', or 'unknown'.
        """
        if move == "all-in":
            move = "B-ALLIN"
        elif move == "bet":
            code = "B"
        elif move == "check":
            code = "C"
        elif move == "fold":
            code = "F"
        elif move == "NP":
            code = "NP"
        else:
            code = "unknown"
        return code

    def set_players_moves(
        self,
        bets: list[Image | None],
        present_player_indices: list[int],
        cards_back_flags: list[numpy.bool_],
        dealer_index: int
    ) -> None:
        """
        Assigns moves to players 2 through 5 based on game context, betting status, and card visibility.

        Player 1 (index 0) is excluded from this process. For him, his move is directly 
        set to 'NP'.
        For each other player:
        - If present, their move is inferred from bet presence and card visibility.
        - If absent, their move is explicitly set to 'absent'.

        Args:
            bets (list[Image | None]): Detected bet crops (or None) for each present player (excluding Player 1).
            present_player_indices (list[int]): Indices of currently active players (excluding Player 1).
            cards_back_flags (list[bool]): Whether each player has visible cards (unfolded).
            dealer_index (int): Index of the player with the dealer button.

        Returns:
            None
        """
        bet_index = 0
        hero_player_index = 0
        hero_player = self.players[hero_player_index]
        hero_player.move = "NP"
        for i, player_index in enumerate(range(1, self.max_nb_players)):
            player = self.players[player_index]
            cards_back = cards_back_flags[i]

            if player_index in present_player_indices:
                bet = bets[bet_index]
                player.move = self.get_move(
                    player_index=player_index,
                    bet=bet,
                    cards_back=cards_back,
                    dealer_index=dealer_index,
                    present_player_indices=present_player_indices
                )
                if player.move == "bet":
                    bet_index += 1
            else:
                player.move = 'absent'
    
    # FIXME: handle fail of ocr and hence modify type 
    def get_pot_values(self, pot_crop: Image) -> tuple[str, str]:
        """
        Extracts the pot and pot total values from a cropped image of the pot area.

        Uses OCR to read text from the cropped region. If two values are found,
        they are assumed to be [pot, pot_total]. If only one value is found, it is assumed
        to be the current pot, and pot_total is returned as an empty string.

        Args:
            pot_crop (Image): Cropped image of the pot value display.

        Returns:
            tuple[str, str]: A tuple containing (pot, pot_total).
        """
        pot, pot_total = '', ''
        pot_crop_arr = numpy.array(pot_crop)
        pot_values = self.ocr_reader.readtext(pot_crop_arr, detail=0)  # value(s) for pot and/or pot total

        if len(pot_values) == 2:
            pot, pot_total = pot_values
        else:
            pot = pot_values[0]

        return pot, pot_total

    def prepare_game_data(self):
        """
        Runs the full OCR + analysis/deduction pipeline on the current screenshot 
        to fully determine the state of the game.

        This includes:
        - Detecting preflop/postflop phase via community cards
        - Extracting player stack, bet, and move information
        - Determining pot and pot total via OCR
        - Assigning dealer/SB/BB/Other positions
        - Formatting output into structured message blocks

        Returns:
            tuple[list[dict], dict]: A tuple containing:
                - player_data: A list of role-tagged dictionaries for each player (present or absent),
                including move, position, stack, and optionally cards and bet.
                - table_data: A single dict containing the game phase, table cards, and pot amounts.
        """
        start = time.time()
        # === Detected table cards === 
        table_cards_block = self.extract_table_cards()
        if table_cards_block:
            self.preflop = False

        # === Players presence based on stack ===
        self.set_player_stacks()
        self.set_players_presence_from_stack() 
        present_player_indices = [player.index for player in self.players 
                                  if player.presence == 'present' and player.index != 0]

        # === Pot ===
        pot_crop = self.crop_region(self.pot_region)
        pot, pot_total = self.get_pot_values(pot_crop)
        # === Cards back presence (for players other than player 1 that has not folded) ===
        cards_back_crops = self.get_cards_back_presence()

        # === Bets ===
        bet_crops = self.get_bet_crops(present_player_indices)
        bets = self.get_bets(bet_crops)
        self.set_bet_values(present_player_indices, 
                            bet_crops, 
                            bets)

        # === Players positions ===
        dealer_index = self.get_dealer_index()
        all_present_player_indices = [0] + present_player_indices
        self.set_player_positions(dealer_index, 
                                  all_present_player_indices)
        # === Present players moves ===
        self.set_players_moves(bets, 
                               present_player_indices,
                               cards_back_crops,
                               dealer_index)
        
        # === Table ===
        table_content = []
        # Indicate game phase
        phase_text = "preflop" if self.preflop else "postflop"
        table_content.append({"type": "text", "text": phase_text})
        # Include community cards only if postflop
        if not self.preflop:
            cards_str = ", ".join(table_cards_block) 
            table_content.append({"type": "text", "text": cards_str})
        # Add pot information
        table_content.append({"type": "text", "text": f"Pot {pot} and Pot total {pot_total}"})

        table_data = {"role": "user", "content": table_content}

        player_data = []

        for player_index in range(self.max_nb_players):
            player = self.players[player_index]
            message = []

            if player.presence == "absent":
                status = "absent"
                text_msg = f"Player {player.index + 1} - Status: {status}"
                message.append({"type": "text", "text": text_msg})
                player_data.append({"role": "user", "content": message})
                continue

            # Present player
            status = "present"
            assert player.move is not None, "if player is present, move must not be None"
            move_code = self._encode_move(player.move)
            position = player.position

            text_parts = [f"Player {player.index + 1}", f"Status: {status}"]
            if player_index != 0:
                text_parts.append(f"Move: {move_code}")
            
            text_parts.append(f"Position: {position}")

            text_msg = " - ".join(text_parts)
            message.append({"type": "text", "text": text_msg})

            # === Stack value (text only, unless all-in) ===
            if not player.has_all_in and player.stack:
                message.append({"type": "text", "text": f"Stack: {player.stack}"})

            # === Bet amount (if move was a bet or all-in) ===
            if player.bet_amount:
                message.append({"type": "text", "text": f"Bet: {player.bet_amount}"})

            # Player 1’s cards
            if player_index == 0:
                cards = self.extract_player1_cards() 
                card_text = ", ".join(cards)
                message.append({"type": "text", "text": f"Player1 cards: {card_text}"})

            player_data.append({"role": "user", "content": message})
        end = time.time()
        elapsed_time = end - start
        print(f"Elasped time local processing : {elapsed_time}")

        return player_data, table_data
    