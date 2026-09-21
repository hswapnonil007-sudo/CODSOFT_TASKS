"""
Tic-Tac-Toe AI - Interactive GUI (Tkinter)
------------------------------------------
Click a square to play. The AI (O) uses Minimax, with optional
Alpha-Beta pruning, and cannot be beaten.

Run:  python tic_tac_toe_gui.py   (or press F5 in IDLE)
No extra installs needed - Tkinter comes with Python.
"""

import math
import tkinter as tk

HUMAN = "X"
AI = "O"
EMPTY = " "

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

nodes_explored = 0


# ------------------------------------------------------------ Game logic
def winning_line(board):
    """Return the winning (a, b, c) tuple, or None."""
    for a, b, c in WIN_LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return (a, b, c)
    return None


def winner(board):
    line = winning_line(board)
    return board[line[0]] if line else None


def is_full(board):
    return EMPTY not in board


def available_moves(board):
    return [i for i, cell in enumerate(board) if cell == EMPTY]


def game_over(board):
    return winner(board) is not None or is_full(board)


def evaluate(board, depth):
    w = winner(board)
    if w == AI:
        return 10 - depth
    if w == HUMAN:
        return depth - 10
    return 0


def minimax(board, depth, is_maximizing):
    global nodes_explored
    nodes_explored += 1
    if game_over(board):
        return evaluate(board, depth)

    if is_maximizing:
        best = -math.inf
        for m in available_moves(board):
            board[m] = AI
            best = max(best, minimax(board, depth + 1, False))
            board[m] = EMPTY
        return best
    best = math.inf
    for m in available_moves(board):
        board[m] = HUMAN
        best = min(best, minimax(board, depth + 1, True))
        board[m] = EMPTY
    return best


def minimax_ab(board, depth, alpha, beta, is_maximizing):
    global nodes_explored
    nodes_explored += 1
    if game_over(board):
        return evaluate(board, depth)

    if is_maximizing:
        best = -math.inf
        for m in available_moves(board):
            board[m] = AI
            best = max(best, minimax_ab(board, depth + 1, alpha, beta, False))
            board[m] = EMPTY
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    best = math.inf
    for m in available_moves(board):
        board[m] = HUMAN
        best = min(best, minimax_ab(board, depth + 1, alpha, beta, True))
        board[m] = EMPTY
        beta = min(beta, best)
        if beta <= alpha:
            break
    return best


def best_move(board, use_pruning=True):
    global nodes_explored
    nodes_explored = 0
    best_score, choice = -math.inf, None
    for m in available_moves(board):
        board[m] = AI
        if use_pruning:
            score = minimax_ab(board, 1, -math.inf, math.inf, False)
        else:
            score = minimax(board, 1, False)
        board[m] = EMPTY
        if score > best_score:
            best_score, choice = score, m
    return choice


# ------------------------------------------------------------------- GUI
class TicTacToeApp:
    BG = "#1e1e2e"
    CELL_BG = "#313244"
    CELL_HOVER = "#45475a"
    X_COLOR = "#89b4fa"
    O_COLOR = "#f38ba8"
    WIN_BG = "#a6e3a1"
    TEXT = "#cdd6f4"

    def __init__(self, root):
        self.root = root
        root.title("Tic-Tac-Toe AI")
        root.configure(bg=self.BG)
        root.resizable(False, False)

        self.board = [EMPTY] * 9
        self.game_active = False
        self.pending = None               # scheduled AI move (so New Game can cancel it)
        self.scores = {"You": 0, "AI": 0, "Draws": 0}

        self.use_pruning = tk.BooleanVar(value=True)
        self.human_first = tk.BooleanVar(value=True)

        self._build_widgets()
        self.new_game()

    # -- layout
    def _build_widgets(self):
        tk.Label(self.root, text="TIC-TAC-TOE AI", font=("Segoe UI", 20, "bold"),
                 bg=self.BG, fg=self.TEXT).pack(pady=(14, 2))
        tk.Label(self.root, text="You: X    AI: O", font=("Segoe UI", 11),
                 bg=self.BG, fg=self.TEXT).pack()

        self.status = tk.Label(self.root, text="", font=("Segoe UI", 13, "bold"),
                               bg=self.BG, fg=self.TEXT)
        self.status.pack(pady=8)

        grid = tk.Frame(self.root, bg=self.BG)
        grid.pack(padx=20)
        self.buttons = []
        for i in range(9):
            btn = tk.Button(
                grid, text="", width=4, height=1,
                font=("Segoe UI", 36, "bold"),
                bg=self.CELL_BG, fg=self.TEXT, activebackground=self.CELL_HOVER,
                relief="flat", bd=0, cursor="hand2",
                command=lambda idx=i: self.human_click(idx),
            )
            btn.grid(row=i // 3, column=i % 3, padx=4, pady=4)
            self.buttons.append(btn)

        self.info = tk.Label(self.root, text="", font=("Segoe UI", 10),
                             bg=self.BG, fg=self.TEXT)
        self.info.pack(pady=(10, 2))

        self.score_label = tk.Label(self.root, text="", font=("Segoe UI", 11),
                                    bg=self.BG, fg=self.TEXT)
        self.score_label.pack()

        options = tk.Frame(self.root, bg=self.BG)
        options.pack(pady=8)
        for text, var in (("Alpha-Beta pruning", self.use_pruning),
                          ("I go first", self.human_first)):
            tk.Checkbutton(options, text=text, variable=var,
                           bg=self.BG, fg=self.TEXT, selectcolor=self.CELL_BG,
                           activebackground=self.BG, activeforeground=self.TEXT,
                           font=("Segoe UI", 10)).pack(side="left", padx=8)

        tk.Button(self.root, text="New Game", font=("Segoe UI", 12, "bold"),
                  bg=self.X_COLOR, fg="#11111b", relief="flat", padx=16, pady=4,
                  cursor="hand2", command=self.new_game).pack(pady=(4, 16))

    # -- game flow
    def new_game(self):
        if self.pending is not None:
            self.root.after_cancel(self.pending)
            self.pending = None
        self.board = [EMPTY] * 9
        self.game_active = True
        for btn in self.buttons:
            btn.config(text="", bg=self.CELL_BG, state="normal")
        self.info.config(text="")
        self.update_scores()

        if self.human_first.get():
            self.status.config(text="Your turn")
        else:
            self.game_active = False
            self.status.config(text="AI is thinking...")
            self.pending = self.root.after(400, self.ai_turn)

    def human_click(self, idx):
        if not self.game_active or self.board[idx] != EMPTY:
            return
        self.place(idx, HUMAN)
        if self.check_end():
            return
        self.status.config(text="AI is thinking...")
        self.game_active = False          # block clicks while the AI "thinks"
        self.pending = self.root.after(400, self.ai_turn)

    def ai_turn(self):
        self.pending = None
        move = best_move(self.board, self.use_pruning.get())
        self.place(move, AI)
        mode = "Alpha-Beta" if self.use_pruning.get() else "Plain Minimax"
        self.info.config(text=f"AI explored {nodes_explored:,} positions ({mode})")
        if self.check_end():
            return
        self.game_active = True
        self.status.config(text="Your turn")

    def place(self, idx, player):
        self.board[idx] = player
        color = self.X_COLOR if player == HUMAN else self.O_COLOR
        self.buttons[idx].config(text=player, fg=color)

    def check_end(self):
        line = winning_line(self.board)
        if line:
            for i in line:
                self.buttons[i].config(bg=self.WIN_BG, fg="#11111b")
            who = self.board[line[0]]
            if who == HUMAN:
                self.scores["You"] += 1
                self.status.config(text="You win!")
            else:
                self.scores["AI"] += 1
                self.status.config(text="The AI wins!")
            self.finish()
            return True
        if is_full(self.board):
            self.scores["Draws"] += 1
            self.status.config(text="It's a draw!")
            self.finish()
            return True
        return False

    def finish(self):
        self.game_active = False
        self.update_scores()

    def update_scores(self):
        s = self.scores
        self.score_label.config(
            text=f"You: {s['You']}    AI: {s['AI']}    Draws: {s['Draws']}")


def main():
    root = tk.Tk()
    TicTacToeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
