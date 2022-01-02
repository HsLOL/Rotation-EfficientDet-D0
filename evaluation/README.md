## Get evaluation result on custom dataset.
```
# first, run `prepare.py` to get the `imgnamefile.txt` and `gt_labels` folder.
python prepare.py

# second, put the `gt_labels` in `/evaluation` folder.

# third, get detection results (i.e. `result_classname` folder) on val dataset.
python batch_inference.py

# forth, get metrics result
python eval.py
```

```
# file structure should be like this.

evaluation/
    -gt_labels/
        -*.txt
    -result_classname
        -Task1_{category_name}.txt
    -batch_inference.py
    -eval.py
    -imgnamefile.txt
    -prepare.py
```
