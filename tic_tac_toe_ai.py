"""
Tic-Tac-Toe AI
--------------
An unbeatable Tic-Tac-Toe agent using the Minimax algorithm,
with optional Alpha-Beta pruning.

Board positions are numbered 1-9:

     1 | 2 | 3
    -----------
     4 | 5 | 6
    -----------
     7 | 8 | 9

Run:  python tic_tac_toe_ai.py
"""

import math

HUMAN = "X"
AI = "O"
EMPTY = " "

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]

# Counts how many game-tree nodes were explored (to compare with/without pruning)
nodes_explored = 0


# ---------------------------------------------------------------- Game logic
def new_board():
    return [EMPTY] * 9


def print_board(board):
    print()
    for row in range(3):
        a, b, c = board[row * 3: row * 3 + 3]
        print(f" {a} | {b} | {c}")
        if row < 2:
            print("-----------")
    print()


def winner(board):
    """Return 'X' or 'O' if someone has won, else None."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board):
    return EMPTY not in board


def available_moves(board):
    return [i for i, cell in enumerate(board) if cell == EMPTY]


def game_over(board):
    return winner(board) is not None or is_full(board)


# ------------------------------------------------------------------- Minimax
def evaluate(board, depth):
    """
    Score a terminal board from the AI's point of view.
    Faster wins / slower losses are preferred by using depth.
    """
    w = winner(board)
    if w == AI:
        return 10 - depth
    if w == HUMAN:
        return depth - 10
    return 0  # draw


def minimax(board, depth, is_maximizing):
    """Plain Minimax (no pruning)."""
    global nodes_explored
    nodes_explored += 1

    if game_over(board):
        return evaluate(board, depth)

    if is_maximizing:  # AI's turn
        best = -math.inf
        for move in available_moves(board):
            board[move] = AI
            best = max(best, minimax(board, depth + 1, False))
            board[move] = EMPTY
        return best
    else:              # Human's turn
        best = math.inf
        for move in available_moves(board):
            board[move] = HUMAN
            best = min(best, minimax(board, depth + 1, True))
            board[move] = EMPTY
        return best


def minimax_ab(board, depth, alpha, beta, is_maximizing):
    """Minimax with Alpha-Beta pruning."""
    global nodes_explored
    nodes_explored += 1

    if game_over(board):
        return evaluate(board, depth)

    if is_maximizing:
        best = -math.inf
        for move in available_moves(board):
            board[move] = AI
            best = max(best, minimax_ab(board, depth + 1, alpha, beta, False))
            board[move] = EMPTY
            alpha = max(alpha, best)
            if beta <= alpha:      # prune: the minimizer will never allow this
                break
        return best
    else:
        best = math.inf
        for move in available_moves(board):
            board[move] = HUMAN
            best = min(best, minimax_ab(board, depth + 1, alpha, beta, True))
            board[move] = EMPTY
            beta = min(beta, best)
            if beta <= alpha:      # prune: the maximizer will never allow this
                break
        return best


def best_move(board, use_pruning=True):
    """Pick the AI's best move for the current board."""
    global nodes_explored
    nodes_explored = 0

    best_score = -math.inf
    move_choice = None

    for move in available_moves(board):
        board[move] = AI
        if use_pruning:
            score = minimax_ab(board, 1, -math.inf, math.inf, False)
        else:
            score = minimax(board, 1, False)
        board[move] = EMPTY

        if score > best_score:
            best_score = score
            move_choice = move

    return move_choice


# ------------------------------------------------------------------ Gameplay
def ask_human_move(board):
    while True:
        raw = input("Your move (1-9): ").strip()
        if not raw.isdigit() or not 1 <= int(raw) <= 9:
            print("Please enter a number from 1 to 9.")
            continue
        idx = int(raw) - 1
        if board[idx] != EMPTY:
            print("That square is taken. Try another.")
            continue
        return idx


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def play_game(use_pruning, human_first):
    board = new_board()
    turn = HUMAN if human_first else AI

    print_board(board)

    while not game_over(board):
        if turn == HUMAN:
            board[ask_human_move(board)] = HUMAN
            turn = AI
        else:
            move = best_move(board, use_pruning)
            board[move] = AI
            print(f"AI plays {move + 1}  "
                  f"(nodes explored: {nodes_explored}, "
                  f"{'alpha-beta' if use_pruning else 'plain minimax'})")
            turn = HUMAN
        print_board(board)

    w = winner(board)
    if w == HUMAN:
        print("You win! (This should never happen.)")
    elif w == AI:
        print("The AI wins!")
    else:
        print("It's a draw!")


def main():
    print("=" * 40)
    print("      TIC-TAC-TOE AI  (You: X, AI: O)")
    print("=" * 40)

    use_pruning = ask_yes_no("Use Alpha-Beta pruning? (y/n): ")

    while True:
        human_first = ask_yes_no("Do you want to go first? (y/n): ")
        play_game(use_pruning, human_first)
        if not ask_yes_no("Play again? (y/n): "):
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
