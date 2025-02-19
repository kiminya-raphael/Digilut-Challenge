# [DigiLut Challenge: Advancing Lung Transplant Rejection Detection](https://www.trustii.io/en/post/join-the-digilut-challenge-advancing-lung-transplant-rejection-detection)
=============================

Lung graft rejection from digitized lung biopsy slides

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
