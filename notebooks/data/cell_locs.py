import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data import copy_registered_cell_centers


if __name__ == "__main__":
    copy_registered_cell_centers("datalist_huc_ablation.csv")
