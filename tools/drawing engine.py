def draw_field(n, queen, *funcs):
    cell_w = len(str(n - 1)) + 1  # width for each cell
    # Print column headers
    print(' '.rjust(cell_w), end='')
    for col in range(n):
        print(str(col).rjust(cell_w), end='')
    print()
    for row in range(n):
        print(str(row).rjust(cell_w), end='')
        for col in range(n):
            if (col, row) == queen:
                ch = 'Q'
            elif any(f(col, row) for f in funcs):
                ch = 'X'
            else:
                ch = '.'
            print(ch.rjust(cell_w), end='')
        print()

# Example boolean functions
def column(col, row):
    global queen
    return queen[0] == col

def row_func(col, row):
    global queen
    return queen[1] == row

def diagonals(col, row):
    global queen
    return (queen[0] - col) ** 2 == (queen[1] - row) ** 2

# Usage
n = 40
queen = (10, 6)
draw_field(n, queen, diagonals, row_func, column)