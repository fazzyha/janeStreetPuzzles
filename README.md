Robot baseball: jane street oct '25 puzzle (https://www.janestreet.com/puzzles/robot-baseball-index/)
-

a python solution using game theory, backward induction and numerical optimisation. The aim is to find the home-run probability $p$ that maximises the chances of an at-bat reaching full count (three balls and two strikes), assuming both players use optimal strategies.


at each pitch, the pitcher chooses ball or stike as the batter simultaneously chooses wait or swing. the outcome changes the ball strike count, or ends the at-bat. the batter receives:
* 1 point for reaching four balls (a walk)
* 0 points for reaching three strikes (a strikeout)
* 4 points for a home run, which has a probability $p$ when the batter swings at a strike.
the batter seeks to maximise expected points and the pitcher seeks to minimise them. the puzzle asks for the maximum probability $q$ of reaching a full count under optimal pitcher and batter strategies. Maximising $q$ is our objective, not the players'.


my method is to solve the game backward. For fixed $p$, we calculate the batter's expected score $V(b,s)$ at each ball-strike count starting from end results (walk, strikeout and home run). we then find optimal mixed strategies: each non-end state is a two player zero-sum game. For interior mixed strategy equilibirum, symmetry gives the same probability for the batter to swing and the pitcher to throw a strike in its payoff matrix.
we then calculate the full count probability using the strategies. We propagate the probability $R(b,s)$ of reaching (3,2) backward through the states. We set $R(3,2)=1$ as arriving there is our goal. We optimise this over $p$, evaluating $q(p)=R(0,0)$ on a dense grid, refining the interior maximum using parabolic interpolation and golden section steps - a Brent style search.

Running gives p_star=0.226973232207805, q_star=0.295967993374272, which matches the published solution.

python 3.7+, standard library
solve_full_game(p) solves the game state-by-state and returns expected scores, full-count probability and player strategies.
q_of_p(p) gives the probability of reaching a full count from the initial state.
maximise_q_pecise() finds and refines the interior maximum
