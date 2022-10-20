##Authors: Karol Kuchnio s21912 and Michał Wadas s20495
from easyAI import Human_Player, AI_Player, Negamax
from game.AweleGame import AweleGame

"""
 Game launcher with provided Negamax algorithm.
"""
if __name__ == "__main__":
    try:
        scoring = lambda game: game.player.score - game.opponent.score
        ai = Negamax(6, scoring)
        game = AweleGame([Human_Player(), AI_Player(ai)])
        game.play()
    except KeyboardInterrupt:
        print("\n\nBye bye Player!\n\n")

    except Exception as e:
        print("Something went wrong, sorry.\nError Message: {0}\n"
              .format(str(e)))
