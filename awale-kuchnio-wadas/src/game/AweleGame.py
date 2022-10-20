##Authors: Karol Kuchnio s21912 and Michał Wadas s20495
from .board import deal_with_position, pick_new_position, print_board, will_starve_player
from .constants import PIT_COUNT, BOARD_ARR, CONST_INDEX
from easyAI import TwoPlayerGame
from .PlayerData import PlayerData


class AweleGame(TwoPlayerGame):
    """
    Main class with game logic.
    """
    def __init__(self, players):
        """Initialize players method"""
        gamePlayers = []
        for i, player in enumerate(players):
            player.score = 0
            player.isstarved = False
            player.camp = i
            gamePlayers.append(
                self.get_complement_properties_player(i, player.score, player.isstarved, player.camp, player.ask_move,
                                                      player, ))
        self.players = gamePlayers
        self.score = [0] * 2
        self.board = BOARD_ARR
        # setting up which player starts first
        self.current_player = 1

    def make_move(self, move):
        """ Make move method including adding points."""
        move = CONST_INDEX.index(move)
        board, score = self.play_turn(self.players[self.player.camp], self.board, move, self.score)
        self.player.score += score[self.player.camp]
        self.player.camp = 1 - self.player.camp

    def possible_moves(self):
        """
        Possible moves method, which returns of all moves allowed
        """
        if self.current_player == 1:
            if max(self.board[:6]) == 0:
                return ["None"]
            possible_moves = [i for i in range(6) if (self.board[i] >= 6 - i)]
            if possible_moves == []:
                possible_moves = [i for i in range(6) if self.board[i] != 0]
        else:
            if max(self.board[6:]) == 0:
                return ["None"]
            possible_moves = [i for i in range(6, 12) if (self.board[i] >= 12 - i)]
            if possible_moves == []:
                possible_moves = [i for i in range(6, 12) if self.board[i] != 0]

        return ["abcdefghijkl"[u] for u in possible_moves]

    def show(self):
        """ Printing boards with all letters&hole numbers"""
        print("Score: %d / %d" % tuple(p.score for p in self.players))
        print_board(self.board)

    def lose(self):
        """ Check losing score method. """
        return self.opponent.score > 24

    def is_over(self):
        """ Check if the game has ended """
        return self.lose() or sum(self.board) < 7 or self.opponent.isstarved

    def get_complement_properties_player(self, number, score, isstarved, camp, ask_move, player=None):
        """ Get method. Retrieving player properties. """
        half_pit = int(PIT_COUNT / 2)
        return PlayerData(number, number * half_pit, (1 + number) * half_pit, (1 - number) * half_pit,
                          (2 - number) * half_pit, player, score, isstarved, camp, ask_move)

    def play_turn(self, current_player, board, position, score):
        """ Play turn method. Check if player is starving and set up position """
        starving = will_starve_player(current_player,
                                      board,
                                      position,
                                      score,
                                      )

        if starving:
            new_board = deal_with_position(board, position)
            return new_board, score
        return pick_new_position(current_player, board, position, score)
