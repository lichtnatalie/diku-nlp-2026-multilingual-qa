| model               | hyperparameter                                                        |   dev_macro_f1 |   best_epoch |   seconds |
|:--------------------|:----------------------------------------------------------------------|---------------:|-------------:|----------:|
| logistic regression | C=0.03                                                                |         0.7908 |     nan      |  nan      |
| logistic regression | C=0.1                                                                 |         0.7755 |     nan      |  nan      |
| logistic regression | C=0.3                                                                 |         0.7680 |     nan      |  nan      |
| logistic regression | C=1.0                                                                 |         0.7505 |     nan      |  nan      |
| logistic regression | C=3.0                                                                 |         0.7654 |     nan      |  nan      |
| recurrent           | {"hidden_dim": 64, "embedding_dim": 64, "dropout": 0.3, "lr": 0.001}  |         0.8220 |       5.0000 |  312.8638 |
| recurrent           | {"hidden_dim": 32, "embedding_dim": 32, "dropout": 0.3, "lr": 0.001}  |         0.8168 |       5.0000 |  161.8474 |
| recurrent           | {"hidden_dim": 64, "embedding_dim": 64, "dropout": 0.5, "lr": 0.0005} |         0.7810 |       5.0000 |  232.4037 |
| recurrent           | {"hidden_dim": 32, "embedding_dim": 64, "dropout": 0.5, "lr": 0.001}  |         0.7964 |       5.0000 |  185.6920 |
