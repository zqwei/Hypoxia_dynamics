import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data import export_segmented_data


if __name__ == "__main__":
    export_segmented_data("datalist.csv", start_index=38)
