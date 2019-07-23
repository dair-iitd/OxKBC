# Data

We are releasing the annotated data, which includes the valid data used for training and validation, and the test set on which we reported the numbers for both fb15k and yago dataset.

The full original data, along with the relation names and wikipedia links (used for MTurk evaluation) can be found at [http://www.cse.iitd.ernet.in/~cs1150210/btp/data.zip](http://www.cse.iitd.ernet.in/~cs1150210/btp/data.zip)

Run the commands to get the copy of data

```
wget http://www.cse.iitd.ernet.in/~cs1150210/btp/data.zip
unzip data.zip ## password = TeXKBC_rocks19
```

After this, you have the following structure:

```
├── fb15k
│   ├── labelled_train
│   │   ├── labelled_train_x.txt
│   │   └── labelled_train_y.txt
│   ├── mid2wikipedia_cleaned.tsv
│   ├── relation_names.txt
│   ├── test
│   │   ├── test_hits10_x.txt
│   │   ├── test_hits10_y.txt
│   │   ├── test_hits1_x.txt
│   │   └── test_hits1_y.txt
│   ├── test.txt
│   ├── train.txt
│   └── valid.txt
└── yago....
```

* The labelled train is a subset from `valid.txt` wich we use for semi-supervised training.
* The `test` directory contains the subset of `test.txt` for which the model, gave correct answer in the file `test_hits1_[x/y].txt` and where the correct answer lied in the top 10 predictions of model in the file `test_hist10_[x/y].txt`


A similar structure is for yago.
