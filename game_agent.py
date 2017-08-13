"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import math
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def lose_win_check(fn):
    """Decorator that checks if they player has won or lost before evaluating the current move."""
    def wraps(game, player):
        if game.is_loser(player):
            return float("-inf")
        if game.is_winner(player):
            return float("inf")
        return fn(game, player)
    return wraps

####################################################
# Metrics
####################################################

def pct_board_available(game):
    """Return percentage of board that is blank"""
    return float(len(game.get_blank_spaces())) / float(game.height * game.width)

def distance_to_opponent(game, player):
    """Return the geometric distance the player is from their opponent.

    This has a maximum value of sqrt(game.width**2 + game.height**2)
    """
    opponent = game.get_opponent(player)
    player_x, player_y = game.get_player_location(player)
    opp_x, opp_y = game.get_player_location(opponent)
    return math.sqrt((player_x - opp_x)**2.0 + (player_y - opp_y)**2.0)

def overlap_with_opponent_moves(game, player):
    """Return number of overlapping moves with opponent.

    This has a min value of 0 and a max value of 2.
    """
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    overlap = len(set(opponent_moves).intersection(set(player_moves)))
    return overlap

def num_player_moves(game, player):
    """Return number of moves available to player.

    This has a min value of 0 and a max value of 8, assuming the board is at least 4x4.
    """
    return len(game.get_legal_moves(player))

def num_opponent_moves(game, player):
    """Return number of moves available to opponent.

    This has a min value of 0 and a max value of 8, assuming the board is at least 5x5.
    """
    return len(game.get_legal_moves(game.get_opponent(player)))

def distance_to_center(game, player):
    """Return Euclidean distance to center.

    This has a max value of sqrt(game.width**2 + game.height**2)
    """
    c_x, c_y = game.width / 2.0, game.height / 2.0
    y, x = game.get_player_location(player)
    return math.sqrt(float((c_y - y)**2 + (c_x - x)**2))

def next_round_improved_score_for_player(game, player):
    """Returns the sum of the next available number of legal player moves
    for each of the current legal moves.
    """
    next_round_legal_moves = 0
    for move in game.get_legal_moves(player):
        g = game.forecast_move(move)
        next_round_legal_moves += len(g.get_legal_moves(player))
    return next_round_legal_moves

def next_round_improved_score_for_opponent(game, player):
    """Returns the sum of the next available number of legal opponent moves
    for each of the current legal moves.
    """
    opponent = game.get_opponent(player)
    return next_round_improved_score_for_player(game, opponent)

####################################################
# Heuristic functions
####################################################

## Improved score +/- individual metrics

def improved_score(game, player):
    """Return number of player moves minus number of opponent moves."""
    return num_player_moves(game, player) - num_opponent_moves(game, player)

def improved_center_score(game, player):
    """Return negative center score for player minus the center score for the opponent.

    Goal here is for our player to stay toward the middle while trying to force opponent outside.
    """
    return -1 * distance_to_center(game, player) - distance_to_center(game, game.get_opponent(player))

def improved_score_plus_center_mod2(game, player):
    return improved_score(game, player) - (distance_to_center(game, player) * 2.0)

def improved_score_plus_center(game, player):
    return improved_score(game, player) + distance_to_center(game, player)

def improved_score_minus_center(game, player):
    return improved_score(game, player) - distance_to_center(game, player)

def improved_score_plus_distance_to_opponent(game, player):
    return improved_score(game, player) + distance_to_opponent(game, player)

def improved_score_minus_distance_to_opponent(game, player):
    return improved_score(game, player) - distance_to_opponent(game, player)

def improved_score_plus_overlap_with_opponent(game, player):
    return improved_score(game, player) + overlap_with_opponent_moves(game, player)

def improved_score_minus_overlap_with_opponent(game, player):
    return improved_score(game, player) - overlap_with_opponent_moves(game, player)

def improved_score_plus_improved_center(game, player):
    return improved_score(game, player) + improved_center_score(game, player)

def improved_score_minus_improved_center(game, player):
    return improved_score(game, player) - improved_center_score(game, player)

## Endgame strategy

def center_then_improved_score(game, player):
    """Try to stay toward the center until the endgame, then simply maximize available moves."""
    is_endgame = pct_board_available(game) < 0.35
    if is_endgame:
        return improved_score(game, player)
    return improved_center_score(game, player)

def improved_with_endgame_strategy(game, player):
    """Returns improved score until the endgame, when it returns
    improved score boosted by looking ahead into future rounds for the player only.

    Endgame is set to when the board only has 40% blank squares left.
    """
    ENDGAME_THRESHOLD = 0.40
    score = improved_score(game, player)
    if pct_board_available(game) < ENDGAME_THRESHOLD:
        return score + next_round_improved_score_for_player(game, player)
    return score

def improved_with_improved_endgame_strategy(game, player):
    """Returns improved score until the endgame, when it returns
    improved score boosted by looking ahead into future rounds for the player only.

    Endgame is set to when the board only has 40% blank squares left.
    """
    ENDGAME_THRESHOLD = 0.40
    if pct_board_available(game) < ENDGAME_THRESHOLD:
        return next_round_improved_score_for_player(game, player) - next_round_improved_score_for_opponent(game, player)
    return improved_score(game, player)

####################################################
# Combined Heuristic functions based on pre_tournament.py
####################################################

def improved_score_minus_center_minus_distance_to_opponent(game, player):
    return improved_score(game, player) - distance_to_center(game, player) - distance_to_opponent(game, player)

def improved_score_minus_center_plus_overlap_with_opponent(game, player):
    return improved_score(game, player) - distance_to_center(game, player) + overlap_with_opponent_moves(game, player)

def improved_score_minus_center_plus_improved_center(game, player):
    return improved_score(game, player) - distance_to_center(game, player) + improved_center_score(game, player)

def improved_score_minus_distance_to_opponent_plus_overlap_with_opponent(game, player):
    return improved_score(game, player) - distance_to_opponent(game, player) + overlap_with_opponent_moves(game, player)

def improved_score_minus_distance_to_opponent_plus_improved_center(game, player):
    return improved_score(game, player) - distance_to_opponent(game, player) + improved_center_score(game, player)

def improved_score_plus_overlap_with_opponent_plus_improved_center(game, player):
    return improved_score(game, player) + overlap_with_opponent_moves(game, player) + improved_center_score(game, player)

ALL_HEURISTICS_AS_SCORE_FCNS = [(lose_win_check(f), f.__name__) for f in [
    improved_score_minus_center_minus_distance_to_opponent,
    improved_score_minus_center_plus_overlap_with_opponent,
    improved_score_minus_center_plus_improved_center,
    improved_score_minus_distance_to_opponent_plus_overlap_with_opponent,
    improved_score_minus_distance_to_opponent_plus_improved_center,
    improved_score_plus_overlap_with_opponent_plus_improved_center,
]]

## Actual custom score functions

@lose_win_check
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return improved_score_plus_overlap_with_opponent(game, player)

@lose_win_check
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return improved_score_plus_improved_center(game, player)

@lose_win_check
def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return improved_score_minus_center(game, player)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=20.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def _time_check(self):
        """Raise search timeout if time left is less than threshold."""
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def _is_terminal(self, game):
        """Return true if no legal moves left"""
        self._time_check()
        return not game.get_legal_moves(self)


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            search_depth = 1
            while True:
                best_move = self.minimax(game, search_depth)
                search_depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        self._time_check()

        best_score = float("-inf")  # it can only get better from here!
        best_move = (-1, -1)  # in case no legal moves, we return illegal move
        for move in game.get_legal_moves():
            new_score = self.min_value(game.forecast_move(move), 1, depth)
            if new_score > best_score:
                best_score = new_score
                best_move = move
        return best_move

    def min_value(self, game, current_depth, depth_limit):
        """Return a win if game is over, else return min value over all legal children"""
        self._time_check()
        if self._is_terminal(game) or current_depth >= depth_limit:
            return self.score(game, self)
        value = float("inf")
        for move in game.get_legal_moves():
            value = min(value, self.max_value(game.forecast_move(move), current_depth + 1, depth_limit))
        return value

    def max_value(self, game, current_depth, depth_limit):
        """Return a loss if game is over, else return max value over all legal children"""
        self._time_check()
        if self._is_terminal(game) or current_depth >= depth_limit:
            return self.score(game, self)
        value = float("-inf")
        for move in game.get_legal_moves():
            value = max(value, self.min_value(game.forecast_move(move), current_depth + 1, depth_limit))
        return value


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = next(iter(game.get_legal_moves()), (-1, -1))

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            # implement iterative deepening by incrementing the search depth until we run out of time
            search_depth = 1
            while True:
                best_move = self.alphabeta(game, search_depth)
                search_depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        self._time_check()
        best_score = float("-inf")  # it can only get better from here!
        best_move = (-1, -1)  # in case no legal moves, we return illegal move
        for move in game.get_legal_moves():
            new_score = self.min_value(game.forecast_move(move), 1, depth, alpha, beta)
            if new_score > best_score:
                best_score = new_score
                best_move = move
            if best_score >= beta:
                return move
            alpha = max(alpha, best_score)
        return best_move

    def max_value(self, game, current_depth, depth_limit, alpha, beta):
        """Return a loss if game is over, else return max value over all legal children"""
        self._time_check()
        if self._is_terminal(game) or current_depth >= depth_limit:
            return self.score(game, self)
        value = float("-inf")
        for move in game.get_legal_moves():
            value = max(value, self.min_value(game.forecast_move(move), current_depth + 1, depth_limit, alpha, beta))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, game, current_depth, depth_limit, alpha, beta):
        """Return a win if game is over, else return min value over all legal children"""
        self._time_check()
        if self._is_terminal(game) or current_depth >= depth_limit:
            return self.score(game, self)
        value = float("inf")
        for move in game.get_legal_moves():
            value = min(value, self.max_value(game.forecast_move(move), current_depth + 1, depth_limit, alpha, beta))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value
