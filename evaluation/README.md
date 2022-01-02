## Get evaluation result on custom dataset.
```
# First, run `prepare.py` to prepare the `imgnamefile.txt` and `gt_labels` folder.
python prepare.py

# Second, put the `gt_labels` in `/evaluation` folder.

# Third, get detection results on val dataset.
python batch_inference.py

# Forth, get metrics result
python eval.py
```
