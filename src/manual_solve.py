#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
"""def solve_6a1e5592(x):
    return x

def solve_b2862040(x):
    return x"""

def ded97339(coord, bricks, x):
    x[coord] = 0
    lost = []

    if not np.where(np.array(x)==8)[0].size:
        return []

    bricks.append([np.where(np.array(x)==8)[0][0], np.where(np.array(x)==8)[1][0]] if np.where(np.array(x)==8)[0].size else 0)

    lost = ded97339((bricks[-1][0], bricks[-1][1]), bricks, x)

    pointA = bricks.pop()

    def draw_point(a, b):
        x[a, b] = 8

    def draw_line(l, x):
        [ draw_point(l[0][0][0], i) for i in range(l[0][0][1], l[0][1][1]) if l[0][0][0] == l[0][1][0] ]
        [ draw_point(i, l[0][1][1]) for i in range(l[0][0][0], l[0][1][0]) if l[0][0][1] == l[0][1][1] ]
        x[l[0][1][0],l[0][1][1]] = 8

    pointB = []
    if np.where(np.array(bricks) == pointA[0])[0].size and bricks[np.where(np.array(bricks) == pointA[0])[0][0]][0] == pointA[0]:
        pointB = [bricks[np.where(np.array(bricks) == pointA[0])[0][0]][0], bricks[np.where(np.array(bricks) == pointA[0])[0][0]][1]]

        draw_line([[pointB,pointA] if pointA>pointB else [pointA,pointB]], x)

    if np.where(np.array(bricks) == pointA[1])[0].size and bricks[np.where(np.array(bricks) == pointA[1])[0][0]][1] == pointA[1]:
        pointB = [bricks[np.where(np.array(bricks) == pointA[1])[0][0]][0], bricks[np.where(np.array(bricks) == pointA[1])[0][0]][1]]

        draw_line([[pointB,pointA] if pointA>pointB else [pointA,pointB]], x)

    if not pointB:
        lost.append(pointA)

    [draw_point(i[0],i[1]) for i in bricks]

    return lost

def solve_ded97339(x):
    bricks = []
    if np.where(np.array(x)==8)[0].size:
        bricks.append([np.where(np.array(x)==8)[0][0], np.where(np.array(x)==8)[1][0]] if np.where(np.array(x)==8)[0].size else 0)
        lost = ded97339((bricks[-1][0], bricks[-1][1]), bricks, x)
        def draw(p): x[p[0],p[1]] = 8
        [draw(l) for l in lost if x[l[0],l[1]] == 0]

    return x

def step_a2fd1cf0(X, Y, walker):
    endPoint = [np.where(walker==2)[0][0], np.where(walker==2)[1][0]]
    walker[Y][X] = 8
    free = 0

    # Check whether we have find our goal.
    if (endPoint[1] in [X, X-1, X+1] and endPoint[0] in [Y, Y-1, Y+1]):
        return True

    # Recursive search.
    if (walker[Y - 1][X] == free and Y > endPoint[0] and step_a2fd1cf0(X, Y - 1,walker)):
        return True
    if (walker[Y + 1][X] == free and Y != endPoint[0] and step_a2fd1cf0(X, Y + 1,walker)):
        return True
    if (walker[Y][X - 1] == free and X > endPoint[1] and step_a2fd1cf0(X - 1, Y,walker)):
        return True
    if (walker[Y][X + 1] == free and X != endPoint[1] and step_a2fd1cf0(X + 1, Y,walker)):
        return True

def solve_a2fd1cf0(x):
    walker = np.array(x)
    endPoint = [np.where(walker==3)[0][0], np.where(walker==3)[1][0]]
    step_a2fd1cf0(np.where(walker==3)[1][0], np.where(walker==3)[0][0], walker)
    walker[endPoint[0],endPoint[1]] = 3
    return walker


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

