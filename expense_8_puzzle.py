#!/usr/bin/env python3

import sys
import datetime
import heapq
import copy
from collections import deque

# Node class to represent a state in the search tree
class Node:
    def __init__(self, state, action="{Start}", g=0, depth=0, parent=None):
        self.state = state
        self.action = action
        self.g = g  # cost so far
        self.depth = depth
        self.parent = parent
        self.f = 0  # f value for A* and Greedy

    def __lt__(self, other):
        if self.f == other.f:
            return self.g < other.g
        return self.f < other.f

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(str(self.state))

    def __str__(self):
        # Match professor's style: full recursive parent node details
        parent_str = "None" if self.parent is None else str(self.parent)
        return (f"< state = {self.state}, action = {self.action} g(n) = {self.g}, "
                f"d = {self.depth}, f(n) = {self.f}, Parent = Pointer to {{{parent_str}}} >")

# Read puzzle from file
def read_puzzle(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    puzzle = []
    for line in lines:
        if line.strip() and "END OF FILE" not in line:
            row = [int(x) for x in line.strip().split()]
            puzzle.append(row)
            if len(puzzle) == 3:
                break
    
    return puzzle

# Find the blank position (0) in the puzzle
def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return (i, j)
    return None

# Get valid moves for a given state (blank-based perspective)
def get_valid_moves(state):
    i, j = find_blank(state)
    moves = []
    
    # Blank moves in these directions (tile moves opposite)
    if i > 0:  # Blank moves up, tile moves down
        moves.append(('Up', i-1, j))
    if i < 2:  # Blank moves down, tile moves up
        moves.append(('Down', i+1, j))
    if j > 0:  # Blank moves left, tile moves right
        moves.append(('Left', i, j-1))
    if j < 2:  # Blank moves right, tile moves left
        moves.append(('Right', i, j+1))
    
    return moves

# Generate successor states
def generate_successors(node):
    successors = []
    blank_i, blank_j = find_blank(node.state)
    
    for direction, i, j in get_valid_moves(node.state):
        new_state = copy.deepcopy(node.state)
        tile = new_state[i][j]
        new_state[blank_i][blank_j], new_state[i][j] = new_state[i][j], new_state[blank_i][blank_j]
        move_cost = tile
        action = f"{{Move {tile} {direction}}}"
        child = Node(new_state, action, node.g + move_cost, node.depth + 1, node)
        successors.append(child)
    
    direction_order = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
    successors.sort(key=lambda x: direction_order[x.action.strip('{}').split()[2]])
    
    return successors

# Heuristic for A* and Greedy search
def calculate_heuristic(state, goal):
    total = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:
                tile = state[i][j]
                goal_i, goal_j = None, None
                for gi in range(3):
                    for gj in range(3):
                        if goal[gi][gj] == tile:
                            goal_i, goal_j = gi, gj
                            break
                    if goal_i is not None:
                        break
                manhattan = abs(i - goal_i) + abs(j - goal_j)
                total += tile * manhattan
    return total

# Check if goal state is reached
def is_goal(state, goal):
    return state == goal

# Trace the path from start to goal
def trace_path(node):
    path = []
    current = node
    
    while current.parent is not None:
        path.append(current.action)
        current = current.parent
    
    path.reverse()
    return path

# Dump search trace to file
def dump_trace(dump_file, current_node, closed, fringe, successors_count=None, is_goal_state=False, 
               nodes_popped=0, nodes_expanded=0, nodes_generated=0, max_fringe_size=0, path=None):
    if dump_file:
        if successors_count is not None:
            dump_file.write(f"Generating successors to {current_node}:\n")
            dump_file.write(f"\t{successors_count} successors generated\n")
        
        closed_states = [node.state for node in closed]
        dump_file.write(f"\tClosed: {closed_states}\n")
        
        dump_file.write("\tFringe: [\n")
        for node in fringe:
            dump_file.write(f"\t\t{node}\n")
        dump_file.write("\t]\n")
        
        if is_goal_state:
            dump_file.write("\nGoal Found:\n")
            dump_file.write(f"{current_node}\n")
            dump_file.write("Path to Goal:\n")
            for step in path:
                dump_file.write(f" {step.strip('{}')}\n")
            dump_file.write(f"Nodes Popped: {nodes_popped}\n")
            dump_file.write(f"Nodes Expanded: {nodes_expanded}\n")
            dump_file.write(f"Nodes Generated: {nodes_generated}\n")
            dump_file.write(f"Max Fringe Size: {max_fringe_size}\n")

# A* Search
def a_star(start, goal, dump_flag):
    if dump_flag:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dump_file = open(f"trace-{timestamp}.txt", "w")
        dump_file.write("Command-Line Arguments : ['" + "', '".join(sys.argv[1:]) + "']\n")
        dump_file.write("Method Selected: a*\n")
        dump_file.write("Running a*\n")
    else:
        dump_file = None
    
    start_node = Node(start)
    h = calculate_heuristic(start, goal)
    start_node.f = start_node.g + h
    
    if is_goal(start, goal):
        if dump_file:
            dump_file.close()
        return start_node, 1, 0, 1, 1, dump_file
    
    fringe = []
    heapq.heappush(fringe, (start_node.f, 0, start_node))
    fringe_counter = 1
    
    closed = set()
    closed_list = []
    
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    max_fringe_size = 1
    
    while fringe:
        max_fringe_size = max(max_fringe_size, len(fringe))
        
        _, _, current = heapq.heappop(fringe)
        nodes_popped += 1
        
        if is_goal(current.state, goal):
            if dump_flag:
                dump_file.write("\nGoal Found:\n")
                dump_file.write(f"{str(current)}\n")
                path = trace_path(current)
                dump_file.write("Path to Goal:\n")
                for step in path:
                    dump_file.write(f" {step.strip('{}')}\n")
                dump_file.write(f"Nodes Popped: {nodes_popped}\n")
                dump_file.write(f"Nodes Expanded: {nodes_expanded}\n")
                dump_file.write(f"Nodes Generated: {nodes_generated}\n")
                dump_file.write(f"Max Fringe Size: {max_fringe_size}\n")
            if dump_file:
                dump_file.close()
            return current, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file
        
        if str(current.state) in closed:
            continue
        
        closed.add(str(current.state))
        closed_list.append(current)
        
        successors = generate_successors(current)
        nodes_expanded += 1
        nodes_generated += len(successors)
        
        for successor in successors:
            if str(successor.state) not in closed:
                h = calculate_heuristic(successor.state, goal)
                successor.f = successor.g + h
                heapq.heappush(fringe, (successor.f, fringe_counter, successor))
                fringe_counter += 1
        
        if dump_flag:
            fringe_for_dump = [node for _, _, node in fringe]
            dump_trace(dump_file, current, closed_list, fringe_for_dump, len(successors))
    
    if dump_file:
        dump_file.write("No solution found.\n")
        dump_file.close()
    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file

# BFS Search
def bfs(start, goal, dump_flag):
    if dump_flag:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dump_file = open(f"trace-{timestamp}.txt", "w")
        dump_file.write("Command-Line Arguments : ['" + "', '".join(sys.argv[1:]) + "']\n")
        dump_file.write("Method Selected: bfs\n")
        dump_file.write("Running bfs\n")
    else:
        dump_file = None
    
    start_node = Node(start)
    
    if is_goal(start, goal):
        if dump_file:
            dump_file.close()
        return start_node, 1, 0, 1, 1, dump_file
    
    fringe = deque([start_node])
    closed = set()
    closed_list = []
    
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    max_fringe_size = 1
    
    while fringe:
        max_fringe_size = max(max_fringe_size, len(fringe))
        
        current = fringe.popleft()
        nodes_popped += 1
        
        if is_goal(current.state, goal):
            path = trace_path(current)
            if dump_flag:
                dump_file.write("\nGoal Found:\n")
                dump_file.write(f"{str(current)}\n")
                dump_file.write("Path to Goal:\n")
                for step in path:
                    dump_file.write(f" {step.strip('{}')}\n")
                dump_file.write(f"Nodes Popped: {nodes_popped}\n")
                dump_file.write(f"Nodes Expanded: {nodes_expanded}\n")
                dump_file.write(f"Nodes Generated: {nodes_generated}\n")
                dump_file.write(f"Max Fringe Size: {max_fringe_size}\n")
            if dump_file:
                dump_file.close()
            return current, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file
        
        if str(current.state) in closed:
            continue
        
        closed.add(str(current.state))
        closed_list.append(current)
        
        successors = generate_successors(current)
        nodes_expanded += 1
        nodes_generated += len(successors)
        
        for successor in successors:
            if str(successor.state) not in closed:
                fringe.append(successor)
        
        if dump_flag:
            fringe_for_dump = list(fringe)
            dump_trace(dump_file, current, closed_list, fringe_for_dump, len(successors))
    
    if dump_file:
        dump_file.write("No solution found.\n")
        dump_file.close()
    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file

# DFS Search
def dfs(start, goal, dump_flag):
    if dump_flag:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dump_file = open(f"trace-{timestamp}.txt", "w")
        dump_file.write("Command-Line Arguments : ['" + "', '".join(sys.argv[1:]) + "']\n")
        dump_file.write("Method Selected: dfs\n")
        dump_file.write("Running dfs\n")
    else:
        dump_file = None
    
    start_node = Node(start)
    
    if is_goal(start, goal):
        return start_node, 1, 0, 1, 1, dump_file
    
    fringe = [start_node]
    closed = set()
    closed_list = []
    
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    max_fringe_size = 1
    
    while fringe:
        max_fringe_size = max(max_fringe_size, len(fringe))
        
        current = fringe.pop()
        nodes_popped += 1
        
        if is_goal(current.state, goal):
            if dump_flag:
                dump_file.write("\nGoal Found:\n")
                dump_file.write(f"{str(current)}\n")
                path = trace_path(current)
                dump_file.write("Path to Goal:\n")
                for step in path:
                    dump_file.write(f" {step.strip('{}')}\n")
                dump_file.write(f"Nodes Popped: {nodes_popped}\n")
                dump_file.write(f"Nodes Expanded: {nodes_expanded}\n")
                dump_file.write(f"Nodes Generated: {nodes_generated}\n")
                dump_file.write(f"Max Fringe Size: {max_fringe_size}\n")
            if dump_file:
                dump_file.close()
            return current, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file
        
        if str(current.state) in closed:
            continue
        
        closed.add(str(current.state))
        closed_list.append(current)
        
        successors = generate_successors(current)
        nodes_expanded += 1
        nodes_generated += len(successors)
        
        for successor in successors:
            if str(successor.state) not in closed:
                fringe.append(successor)
        
        if dump_flag:
            fringe_for_dump = list(fringe)
            dump_trace(dump_file, current, closed_list, fringe_for_dump, len(successors))
    
    if dump_file:
        dump_file.write("No solution found.\n")
        dump_file.close()
    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file

# DLS Search - FIXED
def dls(start, goal, limit, dump_file=None, write_goal=True, dump_flag=False):
    if dump_file:
        dump_file.write(f"Running DLS with depth limit = {limit}\n")
    
    start_node = Node(start)
    
    if is_goal(start, goal):
        return start_node, 1, 0, 1, 1, dump_file
    
    fringe = [start_node]
    closed = set()
    closed_list = []
    
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    max_fringe_size = 1
    
    while fringe:
        max_fringe_size = max(max_fringe_size, len(fringe))
        
        current = fringe.pop()
        nodes_popped += 1
        
        if is_goal(current.state, goal):
            if dump_file and write_goal:
                dump_file.write("\nGoal Found:\n")
                dump_file.write(f"{str(current)}\n")
                path = trace_path(current)
                dump_file.write("Path to Goal:\n")
                for step in path:
                    dump_file.write(f" {step.strip('{}')}\n")
                dump_file.write(f"Nodes Popped: {nodes_popped}\n")
                dump_file.write(f"Nodes Expanded: {nodes_expanded}\n")
                dump_file.write(f"Nodes Generated: {nodes_generated}\n")
                dump_file.write(f"Max Fringe Size: {max_fringe_size}\n")
            return current, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file
        
        if str(current.state) in closed:
            continue
        
        closed.add(str(current.state))
        closed_list.append(current)
        
        successors = []
        
        if current.depth < limit:
            successors = generate_successors(current)
            nodes_expanded += 1
            nodes_generated += len(successors)
            for successor in successors:
                if str(successor.state) not in closed:
                    fringe.append(successor)
        
        if dump_flag and dump_file:  # Check both dump_flag and dump_file
            fringe_for_dump = list(fringe)
            dump_trace(dump_file, current, closed_list, fringe_for_dump, len(successors))
    
    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file

# IDS Search
def ids(start, goal, dump_flag):
    if dump_flag:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dump_file = open(f"trace-{timestamp}.txt", "w")
        dump_file.write("Command-Line Arguments : ['" + "', '".join(sys.argv[1:]) + "']\n")
        dump_file.write("Method Selected: ids\n")
        dump_file.write("Running ids\n")
    else:
        dump_file = None
    
    depth = 0
    total_nodes_popped = 0
    total_nodes_expanded = 0
    total_nodes_generated = 0
    total_max_fringe_size = 0
    
    while True:
        result, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, _ = dls(
            start, goal, depth, dump_file, write_goal=False, dump_flag=dump_flag)
        
        total_nodes_popped += nodes_popped
        total_nodes_expanded += nodes_expanded
        total_nodes_generated += nodes_generated
        total_max_fringe_size = max(total_max_fringe_size, max_fringe_size)
        
        if result:
            if dump_flag:
                dump_file.write("\nGoal Found:\n")
                dump_file.write(f"{str(result)}\n")
                path = trace_path(result)
                dump_file.write("Path to Goal:\n")
                for step in path:
                    dump_file.write(f" {step.strip('{}')}\n")
                dump_file.write(f"Nodes Popped: {total_nodes_popped}\n")
                dump_file.write(f"Nodes Expanded: {total_nodes_expanded}\n")
                dump_file.write(f"Nodes Generated: {total_nodes_generated}\n")
                dump_file.write(f"Max Fringe Size: {total_max_fringe_size}\n")
            if dump_file:
                dump_file.close()
            return result, total_nodes_popped, total_nodes_expanded, total_nodes_generated, total_max_fringe_size, dump_file
        
        depth += 1

# UCS Search
def ucs(start, goal, dump_flag):
    if dump_flag:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dump_file = open(f"trace-{timestamp}.txt", "w")
        dump_file.write("Command-Line Arguments : ['" + "', '".join(sys.argv[1:]) + "']\n")
        dump_file.write("Method Selected: ucs\n")
        dump_file.write("Running ucs\n")
    else:
        dump_file = None
    
    start_node = Node(start)
    
    if is_goal(start, goal):
        return start_node, 1, 0, 1, 1, dump_file
    
    fringe = []
    heapq.heappush(fringe, (start_node.g, 0, start_node))
    fringe_counter = 1
    
    closed = set()
    closed_list = []
    
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    max_fringe_size = 1
    
    while fringe:
        max_fringe_size = max(max_fringe_size, len(fringe))
        
        _, _, current = heapq.heappop(fringe)
        nodes_popped += 1
        
        if is_goal(current.state, goal):
            if dump_flag:
                dump_file.write("\nGoal Found:\n")
                dump_file.write(f"{str(current)}\n")
                path = trace_path(current)
                dump_file.write("Path to Goal:\n")
                for step in path:
                    dump_file.write(f" {step.strip('{}')}\n")
                dump_file.write(f"Nodes Popped: {nodes_popped}\n")
                dump_file.write(f"Nodes Expanded: {nodes_expanded}\n")
                dump_file.write(f"Nodes Generated: {nodes_generated}\n")
                dump_file.write(f"Max Fringe Size: {max_fringe_size}\n")
            if dump_file:
                dump_file.close()
            return current, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file
        
        if str(current.state) in closed:
            continue
        
        closed.add(str(current.state))
        closed_list.append(current)
        
        successors = generate_successors(current)
        nodes_expanded += 1
        nodes_generated += len(successors)
        
        for successor in successors:
            if str(successor.state) not in closed:
                heapq.heappush(fringe, (successor.g, fringe_counter, successor))
                fringe_counter += 1
        
        if dump_flag:
            fringe_for_dump = [node for _, _, node in fringe]
            dump_trace(dump_file, current, closed_list, fringe_for_dump, len(successors))
    
    if dump_file:
        dump_file.write("No solution found.\n")
        dump_file.close()
    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file

# Greedy Search
def greedy(start, goal, dump_flag):
    if dump_flag:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dump_file = open(f"trace-{timestamp}.txt", "w")
        dump_file.write("Command-Line Arguments : ['" + "', '".join(sys.argv[1:]) + "']\n")
        dump_file.write("Method Selected: greedy\n")
        dump_file.write("Running greedy\n")
    else:
        dump_file = None
    
    start_node = Node(start)
    h = calculate_heuristic(start, goal)
    start_node.f = h
    
    if is_goal(start, goal):
        return start_node, 1, 0, 1, 1, dump_file
    
    fringe = []
    heapq.heappush(fringe, (start_node.f, 0, start_node))
    fringe_counter = 1
    
    closed = set()
    closed_list = []
    
    nodes_popped = 0
    nodes_expanded = 0
    nodes_generated = 1
    max_fringe_size = 1
    
    while fringe:
        max_fringe_size = max(max_fringe_size, len(fringe))
        
        _, _, current = heapq.heappop(fringe)
        nodes_popped += 1
        
        if is_goal(current.state, goal):
            if dump_flag:
                dump_file.write("\nGoal Found:\n")
                dump_file.write(f"{str(current)}\n")
                path = trace_path(current)
                dump_file.write("Path to Goal:\n")
                for step in path:
                    dump_file.write(f" {step.strip('{}')}\n")
                dump_file.write(f"Nodes Popped: {nodes_popped}\n")
                dump_file.write(f"Nodes Expanded: {nodes_expanded}\n")
                dump_file.write(f"Nodes Generated: {nodes_generated}\n")
                dump_file.write(f"Max Fringe Size: {max_fringe_size}\n")
            if dump_file:
                dump_file.close()
            return current, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file
        
        if str(current.state) in closed:
            continue
        
        closed.add(str(current.state))
        closed_list.append(current)
        
        successors = generate_successors(current)
        nodes_expanded += 1
        nodes_generated += len(successors)
        
        for successor in successors:
            if str(successor.state) not in closed:
                h = calculate_heuristic(successor.state, goal)
                successor.f = h
                heapq.heappush(fringe, (successor.f, fringe_counter, successor))
                fringe_counter += 1
        
        if dump_flag:
            fringe_for_dump = [node for _, _, node in fringe]
            dump_trace(dump_file, current, closed_list, fringe_for_dump, len(successors))
    
    if dump_file:
        dump_file.write("No solution found.\n")
        dump_file.close()
    return None, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file

# Main function
def main():
    if len(sys.argv) < 3:
        print("Usage: expense_8_puzzle.py <start-file> <goal-file> <method> <dump-flag>")
        return
    
    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    
    method = "a*"  # default
    if len(sys.argv) > 3:
        method = sys.argv[3].lower()
    
    dump_flag = False
    if len(sys.argv) > 4:
        dump_flag = sys.argv[4].lower() == "true"
    
    # Read start and goal files
    start = read_puzzle(start_file)
    goal = read_puzzle(goal_file)
    
    # Choose search method
    if method == "a*":
        result, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file = a_star(start, goal, dump_flag)
    elif method == "bfs":
        result, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file = bfs(start, goal, dump_flag)
    elif method == "dfs":
        result, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file = dfs(start, goal, dump_flag)
    elif method == "dls":
        limit = int(input("Enter depth limit for DLS: "))
        if dump_flag:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            dump_file = open(f"trace-{timestamp}.txt", "w")
            dump_file.write("Command-Line Arguments : ['" + "', '".join(sys.argv[1:]) + "']\n")
            dump_file.write("Method Selected: dls\n")
        else:
            dump_file = None
        result, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file = dls(
            start, goal, limit, dump_file, True, dump_flag)
        if dump_file:
            dump_file.close()
    elif method == "ids":
        result, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file = ids(start, goal, dump_flag)
    elif method == "ucs":
        result, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file = ucs(start, goal, dump_flag)
    elif method == "greedy":
        result, nodes_popped, nodes_expanded, nodes_generated, max_fringe_size, dump_file = greedy(start, goal, dump_flag)
    else:
        print(f"Unknown method: {method}. Available methods: a*, bfs, dfs, dls, ids, ucs, greedy")
        return
    
    # Print results to console in professor's format
    if result:
        path = trace_path(result)
        print(f"Nodes Popped: {nodes_popped}")
        print(f"Nodes Expanded: {nodes_expanded}")
        print(f"Nodes Generated: {nodes_generated}")
        print(f"Max Fringe Size: {max_fringe_size}")
        print(f"Solution Found at depth {result.depth} with cost of {result.g}.")
        print("Steps:")
        for step in path:
            print(f" {step.strip('{}')}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()