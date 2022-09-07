import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_map(file_path):
    grid = []
    # Load from the file
    with open(file_path, 'r') as map_file:
        reader = csv.reader(map_file)
        for i, row in enumerate(reader):
            # load the map
            int_row = [int(col) for col in row]
            grid.append(int_row)
    return grid

def draw_queens(grid):
    fig, ax = plt.subplots(1)
    # ax.margins()
    # Draw map
    row = len(grid)     # map size
    col = len(grid)  # map size
    for i in range(row):
        for j in range(col):
            ax.add_patch(Rectangle((j-0.5, i-0.5),1,1,edgecolor='k',facecolor='w'))  # free space
    # Graph settings
    plt.axis('scaled')

if __name__ == "__main__":
    grid = load_map('Matrix.csv')
    draw_queens(grid)
    plt.show()

