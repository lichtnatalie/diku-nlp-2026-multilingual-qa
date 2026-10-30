| check                              | required_by_brief   | status       | labels_ok   | round_trip_ok   | recovers_gold   | recovered                      | passed   |
|:-----------------------------------|:--------------------|:-------------|:------------|:----------------|:----------------|:-------------------------------|:---------|
| answer at character 0              | True                | exact        | True        | True            | True            | Ada Lovelace                   | True     |
| multi-token answer                 | True                | exact        | True        | True            | True            | 1 July 2000                    | True     |
| punctuation adjacent to the answer | True                | exact        | True        | True            | True            | New Jersey                     | True     |
| unanswerable example               | True                | unanswerable | True        | True            | True            |                                | True     |
| punctuation inside the answer      | False               | exact        | True        | True            | True            | Broadcasting Corporation (BBC) | True     |
| answer followed by a comma         | False               | exact        | True        | True            | True            | Aarhus                         | True     |
| answer ends the context            | False               | exact        | True        | True            | True            | Copenhagen                     | True     |
| single-token answer                | False               | exact        | True        | True            | True            | blue                           | True     |
