# DigiLut Challenge: Advancing Lung Transplant Rejection Detection
==============================

A Foch hospital challenge.

Project Organization
------------

    ├── README.md                 <- The top-level README for developers using this project
    │
    ├── data
    │   ├── raw                   <- Raw data, immutable, not to be modified
    │   │
    │   ├── processed             <- The final, canonical data sets for modeling
    │   │
    │   └── predictions           <- Predictions made on the test set, stored in a .csv file
    │
    ├── models                    <- Trained and serialized models
    │
    ├── reports                   <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── notebooks                 <- Jupyter notebooks
    │   │
    │   ├── preprocess_data.ipynb <- Notebook to turn raw data into features for modeling
    │   │
    │   ├── train.ipynb           <- Notebook to train model(s)
    │   │
    │   └── predict.ipynb         <- Notebook to make predictions using trained models
    │
    ├── utils
    │   └── schemas.py            <- A script defining the schema of the expected submission
    │
    ├── Makefile                  <- Makefile with commands like `make train` or `make predict`
    │
    ├── .env                      <- File to store environment variables
    │
    └── requirements.txt          <- The requirements file for reproducing the environment



Evaluation process
------------
Update the directory paths in the ENV file then run the following commands:

```make workflow```

This command will run the following steps:

1. `make create_environment` to create a new python environment.
2. `source .venv/bin/activate` to activate this environment.
3. `make install_dependencies` to install the dependencies & reproduce your working environment.
4. `make preprocess_data` to preprocess the data.
5. `make train` to train the model(s) & save it.
6. `make predict` to infer on the test set.
