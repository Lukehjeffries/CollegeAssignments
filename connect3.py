X = "X"
O = "O"
E = None
def initial_state():
    return [[E, E, E],[E, E, E],[E, E, E]]

def player(board):
    """Return player to move (X or O): X starts."""
    ########################################################
    # >>> STUDENT TODO START: player
    # Count X and O; X moves when counts are equal, else O.

    if sum(row.count(X) for row in board) == sum(row.count(O) for row in board):
        return X
    else:
        return O
    


    # <<< STUDENT TODO END
    ########################################################

def actions(board):
    """Return set of legal columns {0,1,2} that are not full."""
    ########################################################
    # >>> STUDENT TODO START: actions
    # A column is legal if board[0][col] is empty.
    for col in range(3):

        if board[0][col] == E:
            yield col # Return legal columns
        

    
    
    # <<< STUDENT TODO END
    ########################################################

def result(board, action):
        
    """Return a NEW board after dropping into column 'action' with gravity."""
    ########################################################
    # >>> STUDENT TODO START: result
    # - Compute whose turn it is
    # - Deep copy board
    # - Find lowest empty row in the chosen column and place the piece
    # - Raise ValueError on illegal column
    if action not in actions(board):
        raise ValueError("Illegal action")
    
    import copy
    new_board = copy.deepcopy(board) # Deep copy to avoid mutating original board)
    turn = player(board) 
    for row in range(2, -1, -1): # Start from bottom row upwards
        if new_board[row][action] == E:
            new_board[row][action] = turn
            return new_board
        

    # <<< STUDENT TODO END
    ########################################################

def winner(board):
    """Return X or O if someone has three in a row, else None."""
    ########################################################
    # >>> STUDENT TODO START: winner
    # Check all rows, columns, and the two diagonals.

    for r in range(3):
        if board[r][0] == board[r][1] == board[r][2] != E: # Check rows
            return board[r][0]
        if board[0][r] == board[1][r] == board[2][r] != E: # Check columns
            return board[0][r]
        
    if board[0][0] == board[1][1] == board[2][2] != E: # Check diagonal
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != E: # Check other diagonal
        return board[0][2]
    return None



    # <<< STUDENT TODO END
    ########################################################

def terminal(board):
    """Game over if winner exists or the board is full."""
    ########################################################
    # >>> STUDENT TODO START: terminal
    if winner(board) is not None:
        return True
    
    if all(cell != E for row in board for cell in row):
        return True
    

    # <<< STUDENT TODO END
    ########################################################

def utility(board):
    """1 if X wins, -1 if O wins, else 0."""
    ########################################################
    # >>> STUDENT TODO START: utility
    
    w = winner(board)

    if w == X:
        return 1
    
    elif w == O:
        return -1
    
    return 0


    # <<< STUDENT TODO END
    ########################################################

def minimax(board):
    """Return optimal column for current player using minimax."""
    ########################################################
    # >>> STUDENT TODO START: minimax
    # Implement classic minimax with recursion:
    # - max_value for X, min_value for O
    # - Stop at terminal(board) and return utility
    # - Choose the action that optimizes the value for current player
    def max_value(b):
        """Return the maximum utility value for player X."""

        if terminal(b): # Base case
            return utility(b) # Return utility of terminal board
        
        v = float('-inf') # Initialize to worst case for max player

        for a in actions(b):
            v = max(v, min_value(result(b, a))) # Maximize over min_value of resulting states

        return v
    
    def min_value(b):
        """Return the minimum utility value for player O."""

        if terminal(b):
            return utility(b)
        v = float('inf')
        for a in actions(b): 
            v = min(v, max_value(result(b, a))) # Minimize over max_value of resulting states
        return v

    turn = player(board)

    if turn == X:
        best_val = float('-inf')
        best_action = None 
        for a in actions(board):
            v = min_value(result(board, a)) # Value if X takes action a
            if v > best_val: # Choose action that maximizes value
                best_val = v
                best_action = a

        return best_action # Return the best action for X
    
    else: # turn == O
        best_val = float('inf')
        best_action = None 
        for a in actions(board):
            v = max_value(result(board, a)) # Value if O takes action a
            if v < best_val:
                best_val = v
                best_action = a

        return best_action # Return the best action for O
    

    # <<< STUDENT TODO END
    ########################################################