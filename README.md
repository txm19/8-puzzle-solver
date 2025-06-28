# 8-puzzle-solver

Programming Language:
- Python 3.8.10 (or the specific version you used)

Code Structure:
- The code is structured as a single Python script (`expense_8_puzzle.py`).
- The script contains a `Node` class to represent states in the search tree.
- Various search algorithms (A*, BFS, DFS, DLS, IDS, UCS, Greedy) are implemented as functions.
- The `main()` function handles command-line arguments, reads the start and goal states from files, and invokes the appropriate search algorithm.

How to Run the Code:
1. Ensure you have Python 3 installed. You can check this by running `python3 --version` in your terminal.
2. Place the `expense_8_puzzle.py` script in your working directory.
3. Create two text files (`start.txt` and `goal.txt`) that contain the initial and goal states of the 8-puzzle, respectively. 

start.txt
2 3 6
1 0 7
4 8 5
END OF FILE

goal.txt
1 2 3
4 5 6
7 8 0
END OF FILE


4. Run the script from the command line with the following syntax:
   
   python3 expense_8_puzzle.py <start-file> <goal-file> <method> <dump-flag>
   
   - `<start-file>`: Path to the file containing the initial state (e.g., `start.txt`).
   - `<goal-file>`: Path to the file containing the goal state (e.g., `goal.txt`).
   - `<method>`: The search algorithm to use (`a*`, `bfs`, `dfs`, `dls`, `ids`, `ucs`, `greedy`).
   - `<dump-flag>`: Set to `true` to enable trace dumping, or `false` to disable it.

   Example command:
   python3 expense_8_puzzle.py start.txt goal.txt a* true


Additional Notes:
- If the `dump-flag` is set to `true`, the script will generate a trace file (`trace-YYYYMMDD-HHMMSS.txt`) in the same directory as the script.
- The script is designed to run on any system with Python 3 installed.


Expected Output:
- The script will print the number of nodes popped, expanded, generated, and the maximum fringe size.
- If a solution is found, it will also print the solution path with the cost and steps.

