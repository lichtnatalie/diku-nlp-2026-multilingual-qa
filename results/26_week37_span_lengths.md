| split      | model                                   |   n_gold_spans |   mean_gold_length |   n_predicted_spans |   mean_predicted_length |   median_predicted_length |
|:-----------|:----------------------------------------|---------------:|-------------------:|--------------------:|------------------------:|--------------------------:|
| train      | empty-span baseline                     |           5972 |             3.8208 |                   0 |                  0.0000 |                    0.0000 |
| train      | lexical-overlap span baseline           |           5972 |             3.8208 |                 126 |                  1.1667 |                    1.0000 |
| train      | labeller (independent argmax)           |           5972 |             3.8208 |                5996 |                  1.3753 |                    1.0000 |
| train      | labeller (constrained greedy)           |           5972 |             3.8208 |                5929 |                  1.5180 |                    1.0000 |
| train      | labeller (constrained beam 4)           |           5972 |             3.8208 |                5942 |                  2.0818 |                    1.0000 |
| train      | ablation: no question features (beam 4) |           5972 |             3.8208 |                5929 |                  1.6235 |                    1.0000 |
| train      | labeller (beam 4 + abstention)          |           5972 |             3.8208 |                4519 |                  2.3972 |                    1.0000 |
| validation | empty-span baseline                     |            991 |             3.6549 |                   0 |                  0.0000 |                    0.0000 |
| validation | lexical-overlap span baseline           |            991 |             3.6549 |                  30 |                  1.2000 |                    1.0000 |
| validation | labeller (independent argmax)           |            991 |             3.6549 |                1017 |                  1.2134 |                    1.0000 |
| validation | labeller (constrained greedy)           |            991 |             3.6549 |                 996 |                  1.3193 |                    1.0000 |
| validation | labeller (constrained beam 4)           |            991 |             3.6549 |                 999 |                  1.7578 |                    1.0000 |
| validation | ablation: no question features (beam 4) |            991 |             3.6549 |                1060 |                  1.5594 |                    1.0000 |
| validation | labeller (beam 4 + abstention)          |            991 |             3.6549 |                 693 |                  2.0707 |                    1.0000 |
