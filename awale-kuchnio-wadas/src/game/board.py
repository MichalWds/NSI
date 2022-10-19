##Authors: Karol Kuchnio s21912 and Michał Wadas s20495
from .constants import PIT_COUNT

def print_board(board):
    print("  ".join("lkjihg"))
    print(" ".join(["%02d" % i for i in board[-1:-7:-1]]))
    print(" ".join(["%02d" % i for i in board[:6]]))
    print("  ".join("abcdef"))

def deal_with_position(board, position):
    stones = board[position]
    board[position] = 0
    i = position

    while stones > 0:
        i += 1
        if i % PIT_COUNT != position:
            board[i % PIT_COUNT] += 1
            stones -= 1

    return i % PIT_COUNT, board


def will_starve_player(player, board, position, score=[0, 0]):
    copy_board = board[:]
    copy_score = score[:]
    new_board, new_score = pick_new_position(player, copy_board, position, copy_score)
    min_pick = player.min_pick
    max_pick = player.max_pick
    starving = (sum(new_board[min_pick:max_pick]) == 0)
    return starving


def pick_new_position(player, board, position, score):
    end_position, new_board = deal_with_position(board, position)
    while is_pick_possible(player,new_board,end_position):
        score[player.current_player] += new_board[end_position]
        new_board[end_position] = 0
        end_position -= 1
    return new_board, score

def is_pick_possible(player,new_board, end_position):
        return (player.min_pick <= end_position < player.max_pick and
                2 <= new_board[end_position] <= 3)