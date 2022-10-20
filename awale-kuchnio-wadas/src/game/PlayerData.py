
##Authors: Karol Kuchnio s21912 and Michał Wadas s20495
class PlayerData:
    """Player data class with necessary properties to initialize"""
    def __init__(self, current_player, min_position,max_position,
                 min_pick,max_pick,player,score, isstarved, camp, ask_move):
        self.current_player = current_player
        self.min_position = min_position
        self.max_position = max_position
        self.min_pick = min_pick
        self.max_pick = max_pick
        self.player = player
        self.score = score
        self.isstarved = isstarved
        self.camp = camp
        self.ask_move=ask_move
