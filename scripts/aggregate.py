import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from swp.utils.grid_search import grid_search_aggregate

if __name__ == "__main__":
    grid_search_aggregate()
