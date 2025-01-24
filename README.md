# Multi-environment-Topic-Models
The code is being updated.


**Preprocessing:**
1. Specify the path to your training and test data.
2. Run preprocessor.py to generate .npz files and specify the string you want to name the file/folder path:
python data/preprocessor.py --filename xxx

**Training:**

Train the MTM and specify the filename where your data was saved:
python MTM.py --data_prefix xxx
