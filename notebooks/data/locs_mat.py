import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data import extract_locs_cam


if __name__ == "__main__":
    extract_locs_cam("datalist_gfap_gc6f_v2.csv", max_index=10)
