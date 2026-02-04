# Pruning Experiment Summary

## Results Summary by Task

|                                                                                                                               |   original |   pruning |   quantization |
|:------------------------------------------------------------------------------------------------------------------------------|-----------:|----------:|---------------:|
| ('Qwen/Qwen3-1.7B', 'arc_challenge', 'acc,none', 0.5, 128, 'winogrande')                                                      |   0.398464 |  0.277944 |     nan        |
| ('Qwen/Qwen3-1.7B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'winogrande')                                                 |   0.429181 |  0.307594 |     nan        |
| ('Qwen/Qwen3-1.7B', 'boolq', 'acc,none', 0.5, 128, 'winogrande')                                                              |   0.774924 |  0.655841 |     nan        |
| ('Qwen/Qwen3-1.7B', 'winogrande', 'acc,none', 0.5, 128, 'winogrande')                                                         |   0.610892 |  0.570481 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 128, 'arc_challenge')                                                    |   0.55802  |  0.550341 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 128, 'boolq')                                                            |   0.55802  |  0.545862 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                       |   0.55802  |  0.55347  |       0.510239 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande')                                                       |   0.55802  |  0.549275 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                           |   0.55802  |  0.551574 |       0.540102 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                 |   0.55802  |  0.37244  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 512, 'arc_challenge')                                                    |   0.55802  |  0.547568 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 512, 'boolq')                                                            |   0.55802  |  0.549061 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                       |   0.55802  |  0.554892 |       0.53413  |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande')                                                       |   0.55802  |  0.546502 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                           |   0.55802  |  0.548824 |       0.550341 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                 |   0.55802  |  0.369454 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 1024, 'arc_challenge')                                                   |   0.55802  |  0.547568 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 1024, 'boolq')                                                           |   0.55802  |  0.548422 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                      |   0.55802  |  0.552996 |       0.53413  |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande')                                                      |   0.55802  |  0.547568 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                          |   0.55802  |  0.548066 |       0.539249 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                |   0.55802  |  0.370307 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'boolq')                                                             |   0.55802  |  0.498507 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'commonsense_qa')                                                    |   0.55802  |  0.418089 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                        |   0.55802  |  0.460324 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'ds1000')                                                            |   0.55802  |  0.422782 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'gsm8k')                                                             |   0.55802  |  0.460324 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'mawps')                                                             |   0.55802  |  0.187073 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'race')                                                              |   0.55802  |  0.505973 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'rte')                                                               |   0.55802  |  0.489334 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'winogrande')                                                        |   0.55802  |  0.4407   |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                            |   0.55802  |  0.450085 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'boolq')                                                             |   0.55802  |  0.506826 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'commonsense_qa')                                                    |   0.55802  |  0.421502 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                        |   0.55802  |  0.46971  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'ds1000')                                                            |   0.55802  |  0.220776 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'gsm8k')                                                             |   0.55802  |  0.476323 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'mawps')                                                             |   0.55802  |  0.199232 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'race')                                                              |   0.55802  |  0.508817 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'rte')                                                               |   0.55802  |  0.492321 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'winogrande')                                                        |   0.55802  |  0.442619 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                            |   0.55802  |  0.459044 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'boolq')                                                            |   0.55802  |  0.5032   |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'commonsense_qa')                                                   |   0.55802  |  0.421715 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                       |   0.55802  |  0.453498 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'gsm8k')                                                            |   0.55802  |  0.474403 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'mawps')                                                            |   0.55802  |  0.199232 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'race')                                                             |   0.55802  |  0.512514 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'rte')                                                              |   0.55802  |  0.49552  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'winogrande')                                                       |   0.55802  |  0.448592 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                           |   0.55802  |  0.46587  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                               |   0.564846 |  0.567619 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'boolq')                                                       |   0.564846 |  0.560367 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                  |   0.564846 |  0.56295  |       0.517065 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande')                                                  |   0.564846 |  0.564206 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                      |   0.564846 |  0.565889 |       0.554608 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')            |   0.564846 |  0.397184 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                               |   0.564846 |  0.56634  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'boolq')                                                       |   0.564846 |  0.559087 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                  |   0.564846 |  0.564846 |       0.540102 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande')                                                  |   0.564846 |  0.5657   |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                      |   0.564846 |  0.563804 |       0.554608 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')            |   0.564846 |  0.398038 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                              |   0.564846 |  0.566126 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'boolq')                                                      |   0.564846 |  0.56122  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                 |   0.564846 |  0.564752 |       0.537543 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande')                                                 |   0.564846 |  0.563993 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                     |   0.564846 |  0.564467 |       0.537543 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')           |   0.564846 |  0.398038 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'boolq')                                                        |   0.564846 |  0.513865 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'commonsense_qa')                                               |   0.564846 |  0.440486 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                   |   0.564846 |  0.482082 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'ds1000')                                                       |   0.564846 |  0.443686 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'gsm8k')                                                        |   0.564846 |  0.482295 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'mawps')                                                        |   0.564846 |  0.226536 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'race')                                                         |   0.564846 |  0.522753 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'rte')                                                          |   0.564846 |  0.509172 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'winogrande')                                                   |   0.564846 |  0.459471 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                       |   0.564846 |  0.476962 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'boolq')                                                        |   0.564846 |  0.515998 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'commonsense_qa')                                               |   0.564846 |  0.438567 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                   |   0.564846 |  0.489334 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'ds1000')                                                       |   0.564846 |  0.260026 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'gsm8k')                                                        |   0.564846 |  0.490188 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'mawps')                                                        |   0.564846 |  0.238055 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'race')                                                         |   0.564846 |  0.527019 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'rte')                                                          |   0.564846 |  0.510026 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'winogrande')                                                   |   0.564846 |  0.453498 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                       |   0.564846 |  0.47099  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'boolq')                                                       |   0.564846 |  0.519411 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'commonsense_qa')                                              |   0.564846 |  0.43622  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                  |   0.564846 |  0.468003 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'gsm8k')                                                       |   0.564846 |  0.492534 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'mawps')                                                       |   0.564846 |  0.238055 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'race')                                                        |   0.564846 |  0.527304 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'rte')                                                         |   0.564846 |  0.510239 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'winogrande')                                                  |   0.564846 |  0.457978 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                      |   0.564846 |  0.484215 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 128, 'arc_challenge')                                                         |   0.835859 |  0.830492 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 128, 'boolq')                                                                 |   0.835859 |  0.830282 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                            |   0.835859 |  0.827207 |       0.802189 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande')                                                            |   0.835859 |  0.829125 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                                |   0.835859 |  0.825991 |       0.821128 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                      |   0.835859 |  0.563973 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 512, 'arc_challenge')                                                         |   0.835859 |  0.829545 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 512, 'boolq')                                                                 |   0.835859 |  0.829651 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                            |   0.835859 |  0.827768 |       0.816919 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande')                                                            |   0.835859 |  0.828914 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                                |   0.835859 |  0.829265 |       0.824916 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                      |   0.835859 |  0.566919 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 1024, 'arc_challenge')                                                        |   0.835859 |  0.830808 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 1024, 'boolq')                                                                |   0.835859 |  0.829335 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                           |   0.835859 |  0.827955 |       0.818182 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande')                                                           |   0.835859 |  0.82944  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                               |   0.835859 |  0.826178 |       0.80303  |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                     |   0.835859 |  0.566498 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                                    |   0.809343 |  0.808502 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'boolq')                                                            |   0.809343 |  0.808502 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                       |   0.809343 |  0.806631 |       0.770623 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande')                                                       |   0.809343 |  0.807239 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                           |   0.809343 |  0.803404 |       0.800505 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                 |   0.809343 |  0.549663 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                                    |   0.809343 |  0.809975 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'boolq')                                                            |   0.809343 |  0.809975 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                       |   0.809343 |  0.807286 |       0.779461 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande')                                                       |   0.809343 |  0.80766  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                           |   0.809343 |  0.803919 |       0.796717 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                 |   0.809343 |  0.554293 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                                   |   0.809343 |  0.809449 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'boolq')                                                           |   0.809343 |  0.81008  |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                      |   0.809343 |  0.807426 |       0.776515 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande')                                                      |   0.809343 |  0.806187 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                          |   0.809343 |  0.805228 |       0.796296 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                |   0.809343 |  0.553662 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 128, 'arc_challenge')                                                            |   0.866055 |  0.862615 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 128, 'boolq')                                                                    |   0.866055 |  0.863532 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                               |   0.866055 |  0.865274 |       0.859939 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 128, 'winogrande')                                                               |   0.866055 |  0.863303 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                                   |   0.866055 |  0.863744 |       0.859633 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                         |   0.866055 |  0.621865 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 512, 'arc_challenge')                                                            |   0.866055 |  0.863532 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 512, 'boolq')                                                                    |   0.866055 |  0.863303 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                               |   0.866055 |  0.8649   |       0.870336 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 512, 'winogrande')                                                               |   0.866055 |  0.863609 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                                   |   0.866055 |  0.863948 |       0.864526 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                         |   0.866055 |  0.621713 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 1024, 'arc_challenge')                                                           |   0.866055 |  0.86315  |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 1024, 'boolq')                                                                   |   0.866055 |  0.863303 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                              |   0.866055 |  0.864798 |       0.86789  |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 1024, 'winogrande')                                                              |   0.866055 |  0.863685 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                                  |   0.866055 |  0.863982 |       0.867278 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                        |   0.866055 |  0.620795 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'boolq')                                                                     |   0.866055 |  0.846636 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'commonsense_qa')                                                            |   0.866055 |  0.809862 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                                |   0.866055 |  0.823242 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'ds1000')                                                                    |   0.866055 |  0.790291 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'gsm8k')                                                                     |   0.866055 |  0.838761 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'mawps')                                                                     |   0.866055 |  0.405352 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'race')                                                                      |   0.866055 |  0.844444 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'rte')                                                                       |   0.866055 |  0.845336 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'winogrande')                                                                |   0.866055 |  0.834709 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                                    |   0.866055 |  0.819725 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'boolq')                                                                     |   0.866055 |  0.846483 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'commonsense_qa')                                                            |   0.866055 |  0.815596 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                                |   0.866055 |  0.83685  |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'ds1000')                                                                    |   0.866055 |  0.609786 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'gsm8k')                                                                     |   0.866055 |  0.845336 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'mawps')                                                                     |   0.866055 |  0.38081  |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'race')                                                                      |   0.866055 |  0.842915 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'rte')                                                                       |   0.866055 |  0.844266 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'winogrande')                                                                |   0.866055 |  0.825153 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                                    |   0.866055 |  0.830275 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'boolq')                                                                    |   0.866055 |  0.845413 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'commonsense_qa')                                                           |   0.866055 |  0.818196 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                               |   0.866055 |  0.837003 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'gsm8k')                                                                    |   0.866055 |  0.843119 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'mawps')                                                                    |   0.866055 |  0.38081  |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'race')                                                                     |   0.866055 |  0.843629 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'rte')                                                                      |   0.866055 |  0.841208 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'winogrande')                                                               |   0.866055 |  0.810245 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                                   |   0.866055 |  0.816055 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 128, 'arc_challenge')                                                        |   0.571301 |  0.56321  |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 128, 'boolq')                                                                |   0.571301 |  0.562985 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                           |   0.571301 |  0.562062 |       0.556861 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande')                                                           |   0.571301 |  0.562587 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                               |   0.571301 |  0.56205  |       0.555069 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                     |   0.571301 |  0.413563 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 512, 'arc_challenge')                                                        |   0.571301 |  0.563085 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 512, 'boolq')                                                                |   0.571301 |  0.562587 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                           |   0.571301 |  0.562283 |       0.563035 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande')                                                           |   0.571301 |  0.56184  |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                               |   0.571301 |  0.56163  |       0.560147 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                     |   0.571301 |  0.412418 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 1024, 'arc_challenge')                                                       |   0.571301 |  0.56316  |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 1024, 'boolq')                                                               |   0.571301 |  0.563135 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                          |   0.571301 |  0.561995 |       0.562139 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande')                                                          |   0.571301 |  0.562189 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                              |   0.571301 |  0.562272 |       0.562239 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                    |   0.571301 |  0.413264 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                                   |   0.749054 |  0.741884 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'boolq')                                                           |   0.749054 |  0.740988 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                      |   0.749054 |  0.74163  |       0.732723 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande')                                                      |   0.749054 |  0.742382 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                          |   0.749054 |  0.741076 |       0.734415 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                |   0.749054 |  0.50946  |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                                   |   0.749054 |  0.742357 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'boolq')                                                           |   0.749054 |  0.741486 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                      |   0.749054 |  0.742028 |       0.737502 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande')                                                      |   0.749054 |  0.741909 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                          |   0.749054 |  0.741076 |       0.737204 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                |   0.749054 |  0.510556 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                                  |   0.749054 |  0.742457 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'boolq')                                                          |   0.749054 |  0.741785 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                     |   0.749054 |  0.741065 |       0.738797 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande')                                                     |   0.749054 |  0.742058 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                         |   0.749054 |  0.741917 |       0.7382   |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.749054 |  0.510157 |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 128, 'arc_challenge')                                                       |   0.31     |  0.3175   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 128, 'boolq')                                                               |   0.31     |  0.3165   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                          |   0.31     |  0.318444 |       0.318    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande')                                                          |   0.31     |  0.3145   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                              |   0.31     |  0.317556 |       0.318    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                    |   0.31     |  0.226    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 512, 'arc_challenge')                                                       |   0.31     |  0.317    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 512, 'boolq')                                                               |   0.31     |  0.3165   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                          |   0.31     |  0.318222 |       0.304    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande')                                                          |   0.31     |  0.317    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                              |   0.31     |  0.317778 |       0.302    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                    |   0.31     |  0.231    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 1024, 'arc_challenge')                                                      |   0.31     |  0.318    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 1024, 'boolq')                                                              |   0.31     |  0.3175   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                         |   0.31     |  0.313556 |       0.302    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande')                                                         |   0.31     |  0.315    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                             |   0.31     |  0.318444 |       0.32     |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                   |   0.31     |  0.227    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                                  |   0.414    |  0.4175   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'boolq')                                                          |   0.414    |  0.422    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                     |   0.414    |  0.416444 |       0.408    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande')                                                     |   0.414    |  0.4245   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                         |   0.414    |  0.416889 |       0.406    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.414    |  0.341    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                                  |   0.414    |  0.4175   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'boolq')                                                          |   0.414    |  0.421    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                     |   0.414    |  0.416444 |       0.412    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande')                                                     |   0.414    |  0.424    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                         |   0.414    |  0.417778 |       0.406    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.414    |  0.344    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                                 |   0.414    |  0.4195   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'boolq')                                                         |   0.414    |  0.4205   |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                    |   0.414    |  0.413111 |       0.406    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande')                                                    |   0.414    |  0.423    |     nan        |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                        |   0.414    |  0.419556 |       0.412    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')              |   0.414    |  0.343    |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 128, 'arc_challenge')                                                              |   0.783394 |  0.752708 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 128, 'boolq')                                                                      |   0.783394 |  0.750903 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                                 |   0.783394 |  0.766546 |       0.790614 |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 128, 'winogrande')                                                                 |   0.783394 |  0.754513 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                                     |   0.783394 |  0.76414  |       0.754513 |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                           |   0.783394 |  0.633574 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 512, 'arc_challenge')                                                              |   0.783394 |  0.759928 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 512, 'boolq')                                                                      |   0.783394 |  0.752708 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                                 |   0.783394 |  0.762134 |       0.758123 |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 512, 'winogrande')                                                                 |   0.783394 |  0.755415 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                                     |   0.783394 |  0.761332 |       0.768953 |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                           |   0.783394 |  0.628159 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 1024, 'arc_challenge')                                                             |   0.783394 |  0.759025 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 1024, 'boolq')                                                                     |   0.783394 |  0.755415 |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                                |   0.783394 |  0.762936 |       0.754513 |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 1024, 'winogrande')                                                                |   0.783394 |  0.75361  |     nan        |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                                    |   0.783394 |  0.760128 |       0.765343 |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                          |   0.783394 |  0.633574 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 128, 'arc_challenge')                                                       |   0.67719  |  0.683899 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 128, 'boolq')                                                               |   0.67719  |  0.682518 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                          |   0.67719  |  0.686661 |       0.682715 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 128, 'winogrande')                                                          |   0.67719  |  0.680545 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                              |   0.67719  |  0.684557 |       0.67719  |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                    |   0.67719  |  0.602999 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 512, 'arc_challenge')                                                       |   0.67719  |  0.686267 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 512, 'boolq')                                                               |   0.67719  |  0.682518 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                          |   0.67719  |  0.68824  |       0.692186 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 512, 'winogrande')                                                          |   0.67719  |  0.683899 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                              |   0.67719  |  0.684381 |       0.685872 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                    |   0.67719  |  0.599448 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 1024, 'arc_challenge')                                                      |   0.67719  |  0.684294 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 1024, 'boolq')                                                              |   0.67719  |  0.685478 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                         |   0.67719  |  0.689643 |       0.677979 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande')                                                         |   0.67719  |  0.681926 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                             |   0.67719  |  0.686837 |       0.693765 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                   |   0.67719  |  0.598658 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'boolq')                                                                |   0.67719  |  0.671073 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'commonsense_qa')                                                       |   0.67719  |  0.665154 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                           |   0.67719  |  0.66693  |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'ds1000')                                                               |   0.67719  |  0.632399 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'gsm8k')                                                                |   0.67719  |  0.680939 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'mawps')                                                                |   0.67719  |  0.493883 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'race')                                                                 |   0.67719  |  0.679295 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'rte')                                                                  |   0.67719  |  0.670087 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'winogrande')                                                           |   0.67719  |  0.675809 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                               |   0.67719  |  0.673639 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'boolq')                                                                |   0.67719  |  0.673836 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'commonsense_qa')                                                       |   0.67719  |  0.659432 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                           |   0.67719  |  0.670087 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'ds1000')                                                               |   0.67719  |  0.517167 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'gsm8k')                                                                |   0.67719  |  0.670284 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'mawps')                                                                |   0.67719  |  0.502762 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'race')                                                                 |   0.67719  |  0.683767 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'rte')                                                                  |   0.67719  |  0.666338 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'winogrande')                                                           |   0.67719  |  0.672849 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                               |   0.67719  |  0.668114 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'boolq')                                                               |   0.67719  |  0.671073 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'commonsense_qa')                                                      |   0.67719  |  0.663181 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                          |   0.67719  |  0.673639 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'gsm8k')                                                               |   0.67719  |  0.668903 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'mawps')                                                               |   0.67719  |  0.502762 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'race')                                                                |   0.67719  |  0.683241 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'rte')                                                                 |   0.67719  |  0.677782 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'winogrande')                                                          |   0.67719  |  0.673441 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                              |   0.67719  |  0.664957 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 128, 'arc_challenge')                                                  |   0.498294 |  0.508106 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 128, 'boolq')                                                          |   0.498294 |  0.508746 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                     |   0.498294 |  0.501784 |       0.467577 |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande')                                                     |   0.498294 |  0.501706 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                         |   0.498294 |  0.496354 |       0.483788 |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.498294 |  0.501706 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 512, 'arc_challenge')                                                  |   0.498294 |  0.509172 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 512, 'boolq')                                                          |   0.498294 |  0.510666 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                     |   0.498294 |  0.500388 |       0.483788 |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande')                                                     |   0.498294 |  0.504906 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                         |   0.498294 |  0.496509 |       0.46587  |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.498294 |  0.50384  |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 1024, 'arc_challenge')                                                 |   0.498294 |  0.508106 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 1024, 'boolq')                                                         |   0.498294 |  0.510239 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                    |   0.498294 |  0.50032  |       0.469283 |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande')                                                    |   0.498294 |  0.504266 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                        |   0.498294 |  0.49457  |       0.467577 |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')              |   0.498294 |  0.507679 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'boolq')                                                           |   0.498294 |  0.397327 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'commonsense_qa')                                                  |   0.498294 |  0.292093 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                      |   0.498294 |  0.294369 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'ds1000')                                                          |   0.498294 |  0.376849 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'gsm8k')                                                           |   0.498294 |  0.363766 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'mawps')                                                           |   0.498294 |  0.214733 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'race')                                                            |   0.498294 |  0.404437 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'rte')                                                             |   0.498294 |  0.366041 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'winogrande')                                                      |   0.498294 |  0.298635 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                          |   0.498294 |  0.302048 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'boolq')                                                           |   0.498294 |  0.402162 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'commonsense_qa')                                                  |   0.498294 |  0.298066 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                      |   0.498294 |  0.300341 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'ds1000')                                                          |   0.498294 |  0.372582 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'gsm8k')                                                           |   0.498294 |  0.363766 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'mawps')                                                           |   0.498294 |  0.213026 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'race')                                                            |   0.498294 |  0.405717 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'rte')                                                             |   0.498294 |  0.368885 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'winogrande')                                                      |   0.498294 |  0.298635 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                          |   0.498294 |  0.296928 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'boolq')                                                          |   0.498294 |  0.402446 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'commonsense_qa')                                                 |   0.498294 |  0.294937 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                     |   0.498294 |  0.291809 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'gsm8k')                                                          |   0.498294 |  0.36661  |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'mawps')                                                          |   0.498294 |  0.213026 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'race')                                                           |   0.498294 |  0.409983 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'rte')                                                            |   0.498294 |  0.368032 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'winogrande')                                                     |   0.498294 |  0.304323 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                         |   0.498294 |  0.304608 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                             |   0.538396 |  0.541596 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'boolq')                                                     |   0.538396 |  0.547355 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                |   0.538396 |  0.532113 |       0.498294 |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande')                                                |   0.538396 |  0.540529 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                    |   0.538396 |  0.529398 |       0.508532 |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.538396 |  0.533276 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                             |   0.538396 |  0.542662 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'boolq')                                                     |   0.538396 |  0.545435 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                |   0.538396 |  0.529863 |       0.494027 |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande')                                                |   0.538396 |  0.539036 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                    |   0.538396 |  0.529476 |       0.49744  |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.538396 |  0.536263 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                            |   0.538396 |  0.540102 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'boolq')                                                    |   0.538396 |  0.545222 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                               |   0.538396 |  0.527837 |       0.508532 |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande')                                               |   0.538396 |  0.538609 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                   |   0.538396 |  0.53188  |       0.494027 |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')         |   0.538396 |  0.544795 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'boolq')                                                      |   0.538396 |  0.417235 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'commonsense_qa')                                             |   0.538396 |  0.298919 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                 |   0.538396 |  0.302901 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'ds1000')                                                     |   0.538396 |  0.403015 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'gsm8k')                                                      |   0.538396 |  0.391638 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'mawps')                                                      |   0.538396 |  0.228953 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'race')                                                       |   0.538396 |  0.428328 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'rte')                                                        |   0.538396 |  0.382821 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'winogrande')                                                 |   0.538396 |  0.308589 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                     |   0.538396 |  0.309727 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'boolq')                                                      |   0.538396 |  0.430319 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'commonsense_qa')                                             |   0.538396 |  0.303185 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                 |   0.538396 |  0.307167 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'ds1000')                                                     |   0.538396 |  0.403584 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'gsm8k')                                                      |   0.538396 |  0.394482 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'mawps')                                                      |   0.538396 |  0.228385 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'race')                                                       |   0.538396 |  0.430887 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'rte')                                                        |   0.538396 |  0.387088 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'winogrande')                                                 |   0.538396 |  0.31058  |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                     |   0.538396 |  0.311433 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'boolq')                                                     |   0.538396 |  0.424346 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'commonsense_qa')                                            |   0.538396 |  0.299772 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                |   0.538396 |  0.298635 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'gsm8k')                                                     |   0.538396 |  0.391069 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'mawps')                                                     |   0.538396 |  0.228385 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'race')                                                      |   0.538396 |  0.43942  |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'rte')                                                       |   0.538396 |  0.386519 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'winogrande')                                                |   0.538396 |  0.317122 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                    |   0.538396 |  0.316553 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 128, 'arc_challenge')                                                       |   0.826178 |  0.821023 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 128, 'boolq')                                                               |   0.826178 |  0.821759 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                          |   0.826178 |  0.810989 |       0.75463  |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande')                                                          |   0.826178 |  0.818813 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                              |   0.826178 |  0.810606 |       0.763047 |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                    |   0.826178 |  0.813973 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 512, 'arc_challenge')                                                       |   0.826178 |  0.821338 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 512, 'boolq')                                                               |   0.826178 |  0.823338 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                          |   0.826178 |  0.810683 |       0.771886 |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande')                                                          |   0.826178 |  0.819024 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                              |   0.826178 |  0.810415 |       0.760522 |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                    |   0.826178 |  0.816288 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 1024, 'arc_challenge')                                                      |   0.826178 |  0.818813 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 1024, 'boolq')                                                              |   0.826178 |  0.822601 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                         |   0.826178 |  0.812132 |       0.774832 |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande')                                                         |   0.826178 |  0.818813 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                             |   0.826178 |  0.812443 |       0.771465 |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                   |   0.826178 |  0.819444 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                                  |   0.808502 |  0.802189 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 128, 'boolq')                                                          |   0.808502 |  0.80545  |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                     |   0.808502 |  0.795837 |       0.737374 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande')                                                     |   0.808502 |  0.800295 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                         |   0.808502 |  0.793848 |       0.739899 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.808502 |  0.796507 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                                  |   0.808502 |  0.803451 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 512, 'boolq')                                                          |   0.808502 |  0.806187 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                     |   0.808502 |  0.79358  |       0.748737 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande')                                                     |   0.808502 |  0.800084 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                         |   0.808502 |  0.794575 |       0.747896 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.808502 |  0.797559 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                                 |   0.808502 |  0.802715 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'boolq')                                                         |   0.808502 |  0.802925 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                    |   0.808502 |  0.794402 |       0.746633 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande')                                                    |   0.808502 |  0.800926 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                        |   0.808502 |  0.795263 |       0.752525 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')              |   0.808502 |  0.801347 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 128, 'arc_challenge')                                                          |   0.836086 |  0.823165 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 128, 'boolq')                                                                  |   0.836086 |  0.823394 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                             |   0.836086 |  0.818404 |       0.747706 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 128, 'winogrande')                                                             |   0.836086 |  0.81659  |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                                 |   0.836086 |  0.815624 |       0.751988 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                       |   0.836086 |  0.822936 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 512, 'arc_challenge')                                                          |   0.836086 |  0.819954 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 512, 'boolq')                                                                  |   0.836086 |  0.824312 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                             |   0.836086 |  0.821018 |       0.786239 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 512, 'winogrande')                                                             |   0.836086 |  0.812538 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                                 |   0.836086 |  0.819878 |       0.775535 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                       |   0.836086 |  0.825382 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 1024, 'arc_challenge')                                                         |   0.836086 |  0.821177 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 1024, 'boolq')                                                                 |   0.836086 |  0.822171 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                            |   0.836086 |  0.820948 |       0.791743 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 1024, 'winogrande')                                                            |   0.836086 |  0.814755 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                                |   0.836086 |  0.818043 |       0.740061 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                      |   0.836086 |  0.824006 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'boolq')                                                                   |   0.836086 |  0.712436 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'commonsense_qa')                                                          |   0.836086 |  0.490724 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                              |   0.836086 |  0.507645 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'ds1000')                                                                  |   0.836086 |  0.649643 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'gsm8k')                                                                   |   0.836086 |  0.67472  |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'mawps')                                                                   |   0.836086 |  0.378287 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'race')                                                                    |   0.836086 |  0.656116 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'rte')                                                                     |   0.836086 |  0.684302 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'winogrande')                                                              |   0.836086 |  0.604995 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                                  |   0.836086 |  0.610703 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'boolq')                                                                   |   0.836086 |  0.717533 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'commonsense_qa')                                                          |   0.836086 |  0.507645 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                              |   0.836086 |  0.525994 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'ds1000')                                                                  |   0.836086 |  0.649439 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'gsm8k')                                                                   |   0.836086 |  0.667788 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'mawps')                                                                   |   0.836086 |  0.378287 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'race')                                                                    |   0.836086 |  0.69526  |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'rte')                                                                     |   0.836086 |  0.688175 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'winogrande')                                                              |   0.836086 |  0.595413 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                                  |   0.836086 |  0.599694 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'boolq')                                                                  |   0.836086 |  0.718145 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'commonsense_qa')                                                         |   0.836086 |  0.488787 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                             |   0.836086 |  0.514679 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'gsm8k')                                                                  |   0.836086 |  0.661468 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'mawps')                                                                  |   0.836086 |  0.378287 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'race')                                                                   |   0.836086 |  0.697706 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'rte')                                                                    |   0.836086 |  0.693986 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'winogrande')                                                             |   0.836086 |  0.602243 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                                 |   0.836086 |  0.613761 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 128, 'arc_challenge')                                                      |   0.606552 |  0.595449 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 128, 'boolq')                                                              |   0.606552 |  0.597092 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                         |   0.606552 |  0.594241 |       0.570703 |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande')                                                         |   0.606552 |  0.595574 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                             |   0.606552 |  0.594186 |       0.56911  |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                   |   0.606552 |  0.598337 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 512, 'arc_challenge')                                                      |   0.606552 |  0.596694 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 512, 'boolq')                                                              |   0.606552 |  0.597341 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                         |   0.606552 |  0.594766 |       0.579267 |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande')                                                         |   0.606552 |  0.595897 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                             |   0.606552 |  0.594413 |       0.576877 |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                   |   0.606552 |  0.598287 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 1024, 'arc_challenge')                                                     |   0.606552 |  0.595474 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 1024, 'boolq')                                                             |   0.606552 |  0.597341 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                        |   0.606552 |  0.594877 |       0.580064 |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande')                                                        |   0.606552 |  0.595598 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                            |   0.606552 |  0.594295 |       0.577873 |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                  |   0.606552 |  0.599084 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                                 |   0.8095   |  0.796281 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 128, 'boolq')                                                         |   0.8095   |  0.798222 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                    |   0.8095   |  0.794509 |       0.764788 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande')                                                    |   0.8095   |  0.795509 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                        |   0.8095   |  0.794337 |       0.759809 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')              |   0.8095   |  0.801085 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                                 |   0.8095   |  0.797252 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 512, 'boolq')                                                         |   0.8095   |  0.7976   |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                    |   0.8095   |  0.794219 |       0.780721 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande')                                                    |   0.8095   |  0.794961 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                        |   0.8095   |  0.7943   |       0.778331 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')              |   0.8095   |  0.800388 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                                |   0.8095   |  0.796779 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'boolq')                                                        |   0.8095   |  0.798646 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                   |   0.8095   |  0.794849 |       0.778431 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande')                                                   |   0.8095   |  0.794986 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                       |   0.8095   |  0.794409 |       0.778331 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')             |   0.8095   |  0.801982 |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 128, 'arc_challenge')                                                     |   0.32     |  0.3355   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 128, 'boolq')                                                             |   0.32     |  0.3365   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                        |   0.32     |  0.321273 |       0.29     |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande')                                                        |   0.32     |  0.3365   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                            |   0.32     |  0.322182 |       0.3      |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                  |   0.32     |  0.329    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 512, 'arc_challenge')                                                     |   0.32     |  0.338    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 512, 'boolq')                                                             |   0.32     |  0.3365   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                        |   0.32     |  0.32     |       0.306    |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande')                                                        |   0.32     |  0.334    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                            |   0.32     |  0.321091 |       0.298    |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                  |   0.32     |  0.329    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 1024, 'arc_challenge')                                                    |   0.32     |  0.339    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 1024, 'boolq')                                                            |   0.32     |  0.3375   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                       |   0.32     |  0.3205   |       0.32     |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande')                                                       |   0.32     |  0.3345   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                           |   0.32     |  0.321091 |       0.308    |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                 |   0.32     |  0.329    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                                |   0.442    |  0.4545   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 128, 'boolq')                                                        |   0.442    |  0.4555   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                   |   0.442    |  0.450182 |       0.414    |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande')                                                   |   0.442    |  0.4465   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                       |   0.442    |  0.450182 |       0.43     |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')             |   0.442    |  0.457    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                                |   0.442    |  0.4505   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 512, 'boolq')                                                        |   0.442    |  0.4545   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                   |   0.442    |  0.451818 |       0.434    |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande')                                                   |   0.442    |  0.4505   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                       |   0.442    |  0.450364 |       0.45     |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')             |   0.442    |  0.459    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                               |   0.442    |  0.452    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'boolq')                                                       |   0.442    |  0.4555   |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                  |   0.442    |  0.453    |       0.454    |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande')                                                  |   0.442    |  0.449    |     nan        |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                      |   0.442    |  0.449273 |       0.424    |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')            |   0.442    |  0.462    |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 128, 'arc_challenge')                                                            |   0.6787   |  0.636282 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 128, 'boolq')                                                                    |   0.6787   |  0.65343  |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                               |   0.6787   |  0.601904 |       0.613718 |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 128, 'winogrande')                                                               |   0.6787   |  0.634477 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                                   |   0.6787   |  0.572694 |       0.642599 |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                         |   0.6787   |  0.631769 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 512, 'arc_challenge')                                                            |   0.6787   |  0.643502 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 512, 'boolq')                                                                    |   0.6787   |  0.648014 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                               |   0.6787   |  0.592058 |       0.628159 |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 512, 'winogrande')                                                               |   0.6787   |  0.633574 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                                   |   0.6787   |  0.569741 |       0.599278 |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                         |   0.6787   |  0.648014 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 1024, 'arc_challenge')                                                           |   0.6787   |  0.629061 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 1024, 'boolq')                                                                   |   0.6787   |  0.648917 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                              |   0.6787   |  0.595217 |       0.65343  |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 1024, 'winogrande')                                                              |   0.6787   |  0.623646 |     nan        |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                                  |   0.6787   |  0.577289 |       0.635379 |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                        |   0.6787   |  0.642599 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 128, 'arc_challenge')                                                     |   0.753749 |  0.740726 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 128, 'boolq')                                                             |   0.753749 |  0.737569 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                        |   0.753749 |  0.737964 |       0.719811 |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 128, 'winogrande')                                                        |   0.753749 |  0.736385 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                            |   0.753749 |  0.734591 |       0.73086  |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                  |   0.753749 |  0.740331 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 512, 'arc_challenge')                                                     |   0.753749 |  0.739937 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 512, 'boolq')                                                             |   0.753749 |  0.740529 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                        |   0.753749 |  0.737461 |       0.73165  |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 512, 'winogrande')                                                        |   0.753749 |  0.736385 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                            |   0.753749 |  0.735165 |       0.719021 |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                  |   0.753749 |  0.739148 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 1024, 'arc_challenge')                                                    |   0.753749 |  0.73974  |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 1024, 'boolq')                                                            |   0.753749 |  0.739937 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                       |   0.753749 |  0.738358 |       0.718232 |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande')                                                       |   0.753749 |  0.737174 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                           |   0.753749 |  0.735955 |       0.726914 |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                 |   0.753749 |  0.735201 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'boolq')                                                              |   0.753749 |  0.660089 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'commonsense_qa')                                                     |   0.753749 |  0.552749 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                         |   0.753749 |  0.554065 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'ds1000')                                                             |   0.753749 |  0.658774 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'gsm8k')                                                              |   0.753749 |  0.639569 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'mawps')                                                              |   0.753749 |  0.497764 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'race')                                                               |   0.753749 |  0.675217 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'rte')                                                                |   0.753749 |  0.654301 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'winogrande')                                                         |   0.753749 |  0.575638 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                             |   0.753749 |  0.576164 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'boolq')                                                              |   0.753749 |  0.664562 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'commonsense_qa')                                                     |   0.753749 |  0.544857 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                         |   0.753749 |  0.546961 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'ds1000')                                                             |   0.753749 |  0.658248 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'gsm8k')                                                              |   0.753749 |  0.644567 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'mawps')                                                              |   0.753749 |  0.495659 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'race')                                                               |   0.753749 |  0.668114 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'rte')                                                                |   0.753749 |  0.643515 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'winogrande')                                                         |   0.753749 |  0.569587 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                             |   0.753749 |  0.573007 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'boolq')                                                             |   0.753749 |  0.661405 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'commonsense_qa')                                                    |   0.753749 |  0.54512  |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                        |   0.753749 |  0.545383 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'gsm8k')                                                             |   0.753749 |  0.640095 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'mawps')                                                             |   0.753749 |  0.495659 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'race')                                                              |   0.753749 |  0.674033 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'rte')                                                               |   0.753749 |  0.651671 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'winogrande')                                                        |   0.753749 |  0.568272 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                            |   0.753749 |  0.56985  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 128, 'arc_challenge')                                          |   0.427474 |  0.422355 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 128, 'boolq')                                                  |   0.427474 |  0.419369 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                             |   0.427474 |  0.422639 |       0.401024 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande')                                             |   0.427474 |  0.421288 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                 |   0.427474 |  0.422853 |       0.41041  |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')       |   0.427474 |  0.426195 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 512, 'arc_challenge')                                          |   0.427474 |  0.422995 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 512, 'boolq')                                                  |   0.427474 |  0.421928 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                             |   0.427474 |  0.426123 |       0.414676 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande')                                             |   0.427474 |  0.421075 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                 |   0.427474 |  0.421004 |       0.416382 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')       |   0.427474 |  0.423208 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 1024, 'arc_challenge')                                         |   0.427474 |  0.422782 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 1024, 'boolq')                                                 |   0.427474 |  0.420648 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                            |   0.427474 |  0.425768 |       0.422355 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande')                                            |   0.427474 |  0.420435 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                |   0.427474 |  0.424417 |       0.418089 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')      |   0.427474 |  0.422355 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'boolq')                                                   |   0.427474 |  0.350853 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'commonsense_qa')                                          |   0.427474 |  0.294539 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                              |   0.427474 |  0.299275 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'ds1000')                                                  |   0.427474 |  0.338396 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'gsm8k')                                                   |   0.427474 |  0.324061 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'mawps')                                                   |   0.427474 |  0.187713 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'race')                                                    |   0.427474 |  0.354437 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'rte')                                                     |   0.427474 |  0.336348 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'winogrande')                                              |   0.427474 |  0.283276 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                  |   0.427474 |  0.294369 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'boolq')                                                   |   0.427474 |  0.354096 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'commonsense_qa')                                          |   0.427474 |  0.289932 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                              |   0.427474 |  0.300128 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'ds1000')                                                  |   0.427474 |  0.343686 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'gsm8k')                                                   |   0.427474 |  0.321502 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'mawps')                                                   |   0.427474 |  0.190444 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'race')                                                    |   0.427474 |  0.351536 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'rte')                                                     |   0.427474 |  0.329693 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'winogrande')                                              |   0.427474 |  0.280887 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                  |   0.427474 |  0.287969 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024, 'boolq')                                                  |   0.427474 |  0.356826 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024, 'commonsense_qa')                                         |   0.427474 |  0.283788 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                             |   0.427474 |  0.290956 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024, 'gsm8k')                                                  |   0.427474 |  0.331741 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024, 'mawps')                                                  |   0.427474 |  0.190444 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024, 'race')                                                   |   0.427474 |  0.352048 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024, 'winogrande')                                             |   0.427474 |  0.280375 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                 |   0.427474 |  0.284983 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                     |   0.462457 |  0.448166 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'boolq')                                             |   0.462457 |  0.449232 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                        |   0.462457 |  0.453498 |       0.454778 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande')                                        |   0.462457 |  0.442193 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                            |   0.462457 |  0.451721 |       0.430887 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')  |   0.462457 |  0.461177 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                     |   0.462457 |  0.446459 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'boolq')                                             |   0.462457 |  0.450299 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                        |   0.462457 |  0.453569 |       0.464164 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande')                                        |   0.462457 |  0.446246 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                            |   0.462457 |  0.450085 |       0.452218 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')  |   0.462457 |  0.459471 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                    |   0.462457 |  0.446032 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'boolq')                                            |   0.462457 |  0.448166 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                       |   0.462457 |  0.454565 |       0.452218 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande')                                       |   0.462457 |  0.445392 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                           |   0.462457 |  0.452929 |       0.455631 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte') |   0.462457 |  0.453498 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'boolq')                                              |   0.462457 |  0.37628  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'commonsense_qa')                                     |   0.462457 |  0.311604 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                         |   0.462457 |  0.316766 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'ds1000')                                             |   0.462457 |  0.359727 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'gsm8k')                                              |   0.462457 |  0.350171 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'mawps')                                              |   0.462457 |  0.227474 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'race')                                               |   0.462457 |  0.374061 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'rte')                                                |   0.462457 |  0.36041  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'winogrande')                                         |   0.462457 |  0.31058  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128, 'winogrande_gsm8k_boolq')                             |   0.462457 |  0.318686 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'boolq')                                              |   0.462457 |  0.374573 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'commonsense_qa')                                     |   0.462457 |  0.312969 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                         |   0.462457 |  0.324445 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'ds1000')                                             |   0.462457 |  0.362799 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'gsm8k')                                              |   0.462457 |  0.349147 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'mawps')                                              |   0.462457 |  0.239078 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'race')                                               |   0.462457 |  0.375256 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'rte')                                                |   0.462457 |  0.352048 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'winogrande')                                         |   0.462457 |  0.30529  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512, 'winogrande_gsm8k_boolq')                             |   0.462457 |  0.310367 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'boolq')                                             |   0.462457 |  0.373208 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'commonsense_qa')                                    |   0.462457 |  0.30529  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                        |   0.462457 |  0.311647 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'gsm8k')                                             |   0.462457 |  0.358874 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'mawps')                                             |   0.462457 |  0.239078 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'race')                                              |   0.462457 |  0.379181 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'winogrande')                                        |   0.462457 |  0.311263 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                            |   0.462457 |  0.31442  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 128, 'arc_challenge')                                               |   0.744949 |  0.731902 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 128, 'boolq')                                                       |   0.744949 |  0.732534 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                  |   0.744949 |  0.734287 |       0.731902 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande')                                                  |   0.744949 |  0.733165 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                      |   0.744949 |  0.730535 |       0.729377 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')            |   0.744949 |  0.733796 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 512, 'arc_challenge')                                               |   0.744949 |  0.729167 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 512, 'boolq')                                                       |   0.744949 |  0.733165 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                  |   0.744949 |  0.731973 |       0.741582 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande')                                                  |   0.744949 |  0.731061 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                      |   0.744949 |  0.730885 |       0.727273 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')            |   0.744949 |  0.732744 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 1024, 'arc_challenge')                                              |   0.744949 |  0.730324 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 1024, 'boolq')                                                      |   0.744949 |  0.731797 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                 |   0.744949 |  0.733761 |       0.744108 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande')                                                 |   0.744949 |  0.730114 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                     |   0.744949 |  0.732709 |       0.732323 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')           |   0.744949 |  0.731061 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                          |   0.720539 |  0.699285 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'boolq')                                                  |   0.720539 |  0.699811 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                             |   0.720539 |  0.709245 |       0.707912 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande')                                             |   0.720539 |  0.696338 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                 |   0.720539 |  0.707211 |       0.683081 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')       |   0.720539 |  0.708544 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                          |   0.720539 |  0.697496 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'boolq')                                                  |   0.720539 |  0.698969 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                             |   0.720539 |  0.709175 |       0.708754 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande')                                             |   0.720539 |  0.695286 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                 |   0.720539 |  0.705913 |       0.693603 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')       |   0.720539 |  0.706019 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                         |   0.720539 |  0.695707 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'boolq')                                                 |   0.720539 |  0.699916 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                            |   0.720539 |  0.708123 |       0.710017 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande')                                            |   0.720539 |  0.69476  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                |   0.720539 |  0.708228 |       0.702441 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')      |   0.720539 |  0.706019 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 128, 'arc_challenge')                                                  |   0.74159  |  0.721407 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 128, 'boolq')                                                          |   0.74159  |  0.716743 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                     |   0.74159  |  0.732926 |       0.731804 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 128, 'winogrande')                                                     |   0.74159  |  0.725535 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                         |   0.74159  |  0.728542 |       0.714067 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.74159  |  0.730428 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 512, 'arc_challenge')                                                  |   0.74159  |  0.721636 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 512, 'boolq')                                                          |   0.74159  |  0.718654 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                     |   0.74159  |  0.730836 |       0.759021 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 512, 'winogrande')                                                     |   0.74159  |  0.723624 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                         |   0.74159  |  0.725994 |       0.725688 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')               |   0.74159  |  0.734098 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 1024, 'arc_challenge')                                                 |   0.74159  |  0.72026  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 1024, 'boolq')                                                         |   0.74159  |  0.720031 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                    |   0.74159  |  0.735219 |       0.734557 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 1024, 'winogrande')                                                    |   0.74159  |  0.72393  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                        |   0.74159  |  0.724924 |       0.740367 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')              |   0.74159  |  0.734404 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'boolq')                                                           |   0.74159  |  0.659755 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'commonsense_qa')                                                  |   0.74159  |  0.649297 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                      |   0.74159  |  0.652829 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'ds1000')                                                          |   0.74159  |  0.678471 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'gsm8k')                                                           |   0.74159  |  0.651254 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'mawps')                                                           |   0.74159  |  0.37841  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'race')                                                            |   0.74159  |  0.687339 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'rte')                                                             |   0.74159  |  0.693578 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'winogrande')                                                      |   0.74159  |  0.647278 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                          |   0.74159  |  0.643425 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'boolq')                                                           |   0.74159  |  0.687829 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'commonsense_qa')                                                  |   0.74159  |  0.637615 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                      |   0.74159  |  0.624924 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'ds1000')                                                          |   0.74159  |  0.660917 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'gsm8k')                                                           |   0.74159  |  0.644648 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'mawps')                                                           |   0.74159  |  0.378654 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'race')                                                            |   0.74159  |  0.682324 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'rte')                                                             |   0.74159  |  0.67792  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'winogrande')                                                      |   0.74159  |  0.655841 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                          |   0.74159  |  0.666667 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024, 'boolq')                                                          |   0.74159  |  0.670031 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024, 'commonsense_qa')                                                 |   0.74159  |  0.647706 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                     |   0.74159  |  0.642737 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024, 'gsm8k')                                                          |   0.74159  |  0.632477 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024, 'mawps')                                                          |   0.74159  |  0.378654 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024, 'race')                                                           |   0.74159  |  0.693639 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024, 'winogrande')                                                     |   0.74159  |  0.65474  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                         |   0.74159  |  0.639067 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 128, 'arc_challenge')                                              |   0.558156 |  0.546928 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 128, 'boolq')                                                      |   0.558156 |  0.547177 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                 |   0.558156 |  0.547418 |       0.541227 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande')                                                 |   0.558156 |  0.547052 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                     |   0.558156 |  0.546281 |       0.537941 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')           |   0.558156 |  0.550289 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 512, 'arc_challenge')                                              |   0.558156 |  0.547077 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 512, 'boolq')                                                      |   0.558156 |  0.547476 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                 |   0.558156 |  0.54716  |       0.542621 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande')                                                 |   0.558156 |  0.546903 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                     |   0.558156 |  0.54682  |       0.545608 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')           |   0.558156 |  0.550239 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 1024, 'arc_challenge')                                             |   0.558156 |  0.547077 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 1024, 'boolq')                                                     |   0.558156 |  0.547152 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                |   0.558156 |  0.547293 |       0.543916 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande')                                                |   0.558156 |  0.546306 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                    |   0.558156 |  0.54643  |       0.542621 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.558156 |  0.551235 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                         |   0.741585 |  0.729337 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'boolq')                                                 |   0.741585 |  0.72849  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                            |   0.741585 |  0.731129 |       0.722665 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande')                                            |   0.741585 |  0.72971  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                |   0.741585 |  0.730623 |       0.722266 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')      |   0.741585 |  0.733967 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                         |   0.741585 |  0.729163 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'boolq')                                                 |   0.741585 |  0.729113 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                            |   0.741585 |  0.731353 |       0.728839 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande')                                            |   0.741585 |  0.730034 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                |   0.741585 |  0.73059  |       0.728441 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')      |   0.741585 |  0.734913 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                        |   0.741585 |  0.728889 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'boolq')                                                |   0.741585 |  0.729362 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                           |   0.741585 |  0.731038 |       0.727246 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande')                                           |   0.741585 |  0.729461 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                               |   0.741585 |  0.730706 |       0.726748 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')     |   0.741585 |  0.733718 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 128, 'arc_challenge')                                             |   0.312    |  0.3105   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 128, 'boolq')                                                     |   0.312    |  0.3115   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                |   0.312    |  0.3065   |       0.302    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande')                                                |   0.312    |  0.3095   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                    |   0.312    |  0.3105   |       0.29     |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.312    |  0.315    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 512, 'arc_challenge')                                             |   0.312    |  0.3095   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 512, 'boolq')                                                     |   0.312    |  0.31     |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                |   0.312    |  0.311    |       0.298    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande')                                                |   0.312    |  0.3125   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                    |   0.312    |  0.311167 |       0.32     |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.312    |  0.317    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 1024, 'arc_challenge')                                            |   0.312    |  0.3055   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 1024, 'boolq')                                                    |   0.312    |  0.3115   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                               |   0.312    |  0.310667 |       0.31     |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande')                                               |   0.312    |  0.3115   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                   |   0.312    |  0.3135   |       0.3      |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')         |   0.312    |  0.315    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'arc_challenge')                                        |   0.41     |  0.388    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'boolq')                                                |   0.41     |  0.3855   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                           |   0.41     |  0.384167 |       0.422    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande')                                           |   0.41     |  0.3865   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq')                               |   0.41     |  0.385667 |       0.4      |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')     |   0.41     |  0.388    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'arc_challenge')                                        |   0.41     |  0.3875   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'boolq')                                                |   0.41     |  0.3855   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                           |   0.41     |  0.388833 |       0.396    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande')                                           |   0.41     |  0.387    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq')                               |   0.41     |  0.383333 |       0.404    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')     |   0.41     |  0.389    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'arc_challenge')                                       |   0.41     |  0.3925   |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'boolq')                                               |   0.41     |  0.39     |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                          |   0.41     |  0.388833 |       0.414    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande')                                          |   0.41     |  0.389    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                              |   0.41     |  0.3855   |       0.4      |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')    |   0.41     |  0.389    |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 128, 'arc_challenge')                                                    |   0.541516 |  0.559567 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 128, 'boolq')                                                            |   0.541516 |  0.554152 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                       |   0.541516 |  0.571901 |       0.472924 |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 128, 'winogrande')                                                       |   0.541516 |  0.527076 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                           |   0.541516 |  0.567088 |       0.458484 |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                 |   0.541516 |  0.563177 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 512, 'arc_challenge')                                                    |   0.541516 |  0.563177 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 512, 'boolq')                                                            |   0.541516 |  0.553249 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                       |   0.541516 |  0.577316 |       0.476534 |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 512, 'winogrande')                                                       |   0.541516 |  0.546931 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                           |   0.541516 |  0.56769  |       0.555957 |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                 |   0.541516 |  0.563177 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 1024, 'arc_challenge')                                                   |   0.541516 |  0.565884 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 1024, 'boolq')                                                           |   0.541516 |  0.555957 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                      |   0.541516 |  0.571901 |       0.494585 |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 1024, 'winogrande')                                                      |   0.541516 |  0.544224 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                          |   0.541516 |  0.560469 |       0.577617 |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')                |   0.541516 |  0.570397 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 128, 'arc_challenge')                                             |   0.693765 |  0.69929  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 128, 'boolq')                                                     |   0.693765 |  0.699684 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                |   0.693765 |  0.702184 |       0.684294 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 128, 'winogrande')                                                |   0.693765 |  0.699882 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq')                                    |   0.693765 |  0.700605 |       0.682715 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.693765 |  0.703631 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 512, 'arc_challenge')                                             |   0.693765 |  0.699882 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 512, 'boolq')                                                     |   0.693765 |  0.700671 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                |   0.693765 |  0.701329 |       0.690608 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 512, 'winogrande')                                                |   0.693765 |  0.700079 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq')                                    |   0.693765 |  0.697777 |       0.692976 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.693765 |  0.698895 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 1024, 'arc_challenge')                                            |   0.693765 |  0.701657 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 1024, 'boolq')                                                    |   0.693765 |  0.700474 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                               |   0.693765 |  0.701592 |       0.696133 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande')                                               |   0.693765 |  0.700474 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq')                                   |   0.693765 |  0.699553 |       0.700868 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')         |   0.693765 |  0.698106 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'boolq')                                                      |   0.693765 |  0.650039 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'commonsense_qa')                                             |   0.693765 |  0.597948 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                 |   0.693765 |  0.599448 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'ds1000')                                                     |   0.693765 |  0.654301 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'gsm8k')                                                      |   0.693765 |  0.662983 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'mawps')                                                      |   0.693765 |  0.500552 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'race')                                                       |   0.693765 |  0.657932 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'rte')                                                        |   0.693765 |  0.645146 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'winogrande')                                                 |   0.693765 |  0.601105 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128, 'winogrande_gsm8k_boolq')                                     |   0.693765 |  0.608524 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'boolq')                                                      |   0.693765 |  0.64925  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'commonsense_qa')                                             |   0.693765 |  0.600631 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                 |   0.693765 |  0.608327 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'ds1000')                                                     |   0.693765 |  0.644515 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'gsm8k')                                                      |   0.693765 |  0.658564 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'mawps')                                                      |   0.693765 |  0.50371  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'race')                                                       |   0.693765 |  0.662352 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'rte')                                                        |   0.693765 |  0.646251 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'winogrande')                                                 |   0.693765 |  0.608998 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512, 'winogrande_gsm8k_boolq')                                     |   0.693765 |  0.612865 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024, 'boolq')                                                     |   0.693765 |  0.655722 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024, 'commonsense_qa')                                            |   0.693765 |  0.591318 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                |   0.693765 |  0.604972 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024, 'gsm8k')                                                     |   0.693765 |  0.652092 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024, 'mawps')                                                     |   0.693765 |  0.50371  |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024, 'race')                                                      |   0.693765 |  0.661405 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024, 'winogrande')                                                |   0.693765 |  0.608051 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024, 'winogrande_gsm8k_boolq')                                    |   0.693765 |  0.614838 |     nan        |

## Average Performance across Tasks

|                                                                                             |   original |   pruning |   quantization |
|:--------------------------------------------------------------------------------------------|-----------:|----------:|---------------:|
| ('Qwen/Qwen3-1.7B', 0.5, 128, 'winogrande')                                                 |   0.56887  |  0.470765 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 128, 'arc_challenge')                                               |   0.649006 |  0.645115 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 128, 'boolq')                                                       |   0.649006 |  0.64404  |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                  |   0.649006 |  0.64612  |       0.631724 |
| ('Qwen/Qwen3-8B', 0.25, 128, 'winogrande')                                                  |   0.649006 |  0.644743 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 128, 'winogrande_gsm8k_boolq')                                      |   0.649006 |  0.64517  |       0.638288 |
| ('Qwen/Qwen3-8B', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')            |   0.649006 |  0.475611 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 512, 'arc_challenge')                                               |   0.649006 |  0.645736 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 512, 'boolq')                                                       |   0.649006 |  0.644352 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                  |   0.649006 |  0.646277 |       0.637072 |
| ('Qwen/Qwen3-8B', 0.25, 512, 'winogrande')                                                  |   0.649006 |  0.645132 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 512, 'winogrande_gsm8k_boolq')                                      |   0.649006 |  0.644885 |       0.641026 |
| ('Qwen/Qwen3-8B', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')            |   0.649006 |  0.476    |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 1024, 'arc_challenge')                                              |   0.649006 |  0.645776 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 1024, 'boolq')                                                      |   0.649006 |  0.645106 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                                 |   0.649006 |  0.645476 |       0.634153 |
| ('Qwen/Qwen3-8B', 0.25, 1024, 'winogrande')                                                 |   0.649006 |  0.644423 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 1024, 'winogrande_gsm8k_boolq')                                     |   0.649006 |  0.645189 |       0.63954  |
| ('Qwen/Qwen3-8B', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')           |   0.649006 |  0.475905 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'boolq')                                                        |   0.666528 |  0.63252  |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'commonsense_qa')                                               |   0.666528 |  0.583398 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                   |   0.666528 |  0.608144 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'ds1000')                                                       |   0.666528 |  0.572289 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'gsm8k')                                                        |   0.666528 |  0.61558  |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'mawps')                                                        |   0.666528 |  0.328211 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'race')                                                         |   0.666528 |  0.638116 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'rte')                                                          |   0.666528 |  0.628483 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'winogrande')                                                   |   0.666528 |  0.602672 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 128, 'winogrande_gsm8k_boolq')                                       |   0.666528 |  0.605103 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'boolq')                                                        |   0.666528 |  0.635786 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'commonsense_qa')                                               |   0.666528 |  0.583774 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                   |   0.666528 |  0.616495 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'ds1000')                                                       |   0.666528 |  0.401939 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'gsm8k')                                                        |   0.666528 |  0.620533 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'mawps')                                                        |   0.666528 |  0.330215 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'race')                                                         |   0.666528 |  0.64063  |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'rte')                                                          |   0.666528 |  0.628238 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'winogrande')                                                   |   0.666528 |  0.59853  |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512, 'winogrande_gsm8k_boolq')                                       |   0.666528 |  0.607106 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'boolq')                                                       |   0.666528 |  0.634774 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'commonsense_qa')                                              |   0.666528 |  0.584828 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                  |   0.666528 |  0.608036 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'gsm8k')                                                       |   0.666528 |  0.61974  |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'mawps')                                                       |   0.666528 |  0.330215 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'race')                                                        |   0.666528 |  0.641672 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'rte')                                                         |   0.666528 |  0.631187 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'winogrande')                                                  |   0.666528 |  0.597564 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024, 'winogrande_gsm8k_boolq')                                      |   0.666528 |  0.607774 |     nan        |
| ('google/gemma-7b', 0.25, 128, 'arc_challenge')                                             |   0.647087 |  0.641347 |     nan        |
| ('google/gemma-7b', 0.25, 128, 'boolq')                                                     |   0.647087 |  0.644093 |     nan        |
| ('google/gemma-7b', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                                |   0.647087 |  0.632654 |       0.598055 |
| ('google/gemma-7b', 0.25, 128, 'winogrande')                                                |   0.647087 |  0.638443 |     nan        |
| ('google/gemma-7b', 0.25, 128, 'winogrande_gsm8k_boolq')                                    |   0.647087 |  0.628546 |       0.607239 |
| ('google/gemma-7b', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.647087 |  0.63872  |     nan        |
| ('google/gemma-7b', 0.25, 512, 'arc_challenge')                                             |   0.647087 |  0.642042 |     nan        |
| ('google/gemma-7b', 0.25, 512, 'boolq')                                                     |   0.647087 |  0.644038 |     nan        |
| ('google/gemma-7b', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                                |   0.647087 |  0.631441 |       0.613134 |
| ('google/gemma-7b', 0.25, 512, 'winogrande')                                                |   0.647087 |  0.638264 |     nan        |
| ('google/gemma-7b', 0.25, 512, 'winogrande_gsm8k_boolq')                                    |   0.647087 |  0.628721 |       0.606252 |
| ('google/gemma-7b', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')          |   0.647087 |  0.641197 |     nan        |
| ('google/gemma-7b', 0.25, 1024, 'arc_challenge')                                            |   0.647087 |  0.64027  |     nan        |
| ('google/gemma-7b', 0.25, 1024, 'boolq')                                                    |   0.647087 |  0.643727 |     nan        |
| ('google/gemma-7b', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                               |   0.647087 |  0.63204  |       0.617744 |
| ('google/gemma-7b', 0.25, 1024, 'winogrande')                                               |   0.647087 |  0.63748  |     nan        |
| ('google/gemma-7b', 0.25, 1024, 'winogrande_gsm8k_boolq')                                   |   0.647087 |  0.629501 |       0.606923 |
| ('google/gemma-7b', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')         |   0.647087 |  0.642467 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'boolq')                                                      |   0.656631 |  0.546772 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'commonsense_qa')                                             |   0.656631 |  0.408621 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                                 |   0.656631 |  0.414745 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'ds1000')                                                     |   0.656631 |  0.52207  |     nan        |
| ('google/gemma-7b', 0.5, 128, 'gsm8k')                                                      |   0.656631 |  0.517423 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'mawps')                                                      |   0.656631 |  0.329934 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'race')                                                       |   0.656631 |  0.541024 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'rte')                                                        |   0.656631 |  0.521866 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'winogrande')                                                 |   0.656631 |  0.446964 |     nan        |
| ('google/gemma-7b', 0.5, 128, 'winogrande_gsm8k_boolq')                                     |   0.656631 |  0.449661 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'boolq')                                                      |   0.656631 |  0.553644 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'commonsense_qa')                                             |   0.656631 |  0.413438 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                                 |   0.656631 |  0.420116 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'ds1000')                                                     |   0.656631 |  0.520963 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'gsm8k')                                                      |   0.656631 |  0.517651 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'mawps')                                                      |   0.656631 |  0.328839 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'race')                                                       |   0.656631 |  0.549994 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'rte')                                                        |   0.656631 |  0.521916 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'winogrande')                                                 |   0.656631 |  0.443554 |     nan        |
| ('google/gemma-7b', 0.5, 512, 'winogrande_gsm8k_boolq')                                     |   0.656631 |  0.445266 |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'boolq')                                                     |   0.656631 |  0.551585 |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'commonsense_qa')                                            |   0.656631 |  0.407154 |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                                |   0.656631 |  0.412626 |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'gsm8k')                                                     |   0.656631 |  0.51481  |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'mawps')                                                     |   0.656631 |  0.328839 |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'race')                                                      |   0.656631 |  0.555286 |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'rte')                                                       |   0.656631 |  0.525052 |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'winogrande')                                                |   0.656631 |  0.44799  |     nan        |
| ('google/gemma-7b', 0.5, 1024, 'winogrande_gsm8k_boolq')                                    |   0.656631 |  0.451193 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 128, 'arc_challenge')                                     |   0.577639 |  0.568794 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 128, 'boolq')                                             |   0.577639 |  0.567654 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 128, 'commonsense_qa_gsm8k_boolq')                        |   0.577639 |  0.572354 |       0.561139 |
| ('meta-llama/Llama-3.2-3B', 0.25, 128, 'winogrande')                                        |   0.577639 |  0.565295 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 128, 'winogrande_gsm8k_boolq')                            |   0.577639 |  0.571057 |       0.550839 |
| ('meta-llama/Llama-3.2-3B', 0.25, 128, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')  |   0.577639 |  0.574019 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 512, 'arc_challenge')                                     |   0.577639 |  0.56855  |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 512, 'boolq')                                             |   0.577639 |  0.568093 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 512, 'commonsense_qa_gsm8k_boolq')                        |   0.577639 |  0.573515 |       0.565527 |
| ('meta-llama/Llama-3.2-3B', 0.25, 512, 'winogrande')                                        |   0.577639 |  0.56734  |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 512, 'winogrande_gsm8k_boolq')                            |   0.577639 |  0.570114 |       0.569286 |
| ('meta-llama/Llama-3.2-3B', 0.25, 512, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte')  |   0.577639 |  0.573524 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 1024, 'arc_challenge')                                    |   0.577639 |  0.568783 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 1024, 'boolq')                                            |   0.577639 |  0.568636 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 1024, 'commonsense_qa_gsm8k_boolq')                       |   0.577639 |  0.573524 |       0.568103 |
| ('meta-llama/Llama-3.2-3B', 0.25, 1024, 'winogrande')                                       |   0.577639 |  0.566872 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 1024, 'winogrande_gsm8k_boolq')                           |   0.577639 |  0.570851 |       0.572428 |
| ('meta-llama/Llama-3.2-3B', 0.25, 1024, 'winogrande_gsm8k_boolq_ds1000_race_mawps_wmt_rte') |   0.577639 |  0.573163 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'boolq')                                              |   0.581322 |  0.509232 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'commonsense_qa')                                     |   0.581322 |  0.463347 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'commonsense_qa_gsm8k_boolq')                         |   0.581322 |  0.467079 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'ds1000')                                             |   0.581322 |  0.507724 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'gsm8k')                                              |   0.581322 |  0.497117 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'mawps')                                              |   0.581322 |  0.323537 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'race')                                               |   0.581322 |  0.518442 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'rte')                                                |   0.581322 |  0.50887  |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'winogrande')                                         |   0.581322 |  0.46056  |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 128, 'winogrande_gsm8k_boolq')                             |   0.581322 |  0.466251 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'boolq')                                              |   0.581322 |  0.516437 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'commonsense_qa')                                     |   0.581322 |  0.460287 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'commonsense_qa_gsm8k_boolq')                         |   0.581322 |  0.464456 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'ds1000')                                             |   0.581322 |  0.502979 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'gsm8k')                                              |   0.581322 |  0.493465 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'mawps')                                              |   0.581322 |  0.327972 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'race')                                               |   0.581322 |  0.517867 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'rte')                                                |   0.581322 |  0.501478 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'winogrande')                                         |   0.581322 |  0.462754 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512, 'winogrande_gsm8k_boolq')                             |   0.581322 |  0.469467 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024, 'boolq')                                             |   0.581322 |  0.513947 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024, 'commonsense_qa')                                    |   0.581322 |  0.457026 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024, 'commonsense_qa_gsm8k_boolq')                        |   0.581322 |  0.462578 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024, 'gsm8k')                                             |   0.581322 |  0.493796 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024, 'mawps')                                             |   0.581322 |  0.327972 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024, 'race')                                              |   0.581322 |  0.521568 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024, 'winogrande')                                        |   0.581322 |  0.463607 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024, 'winogrande_gsm8k_boolq')                            |   0.581322 |  0.463327 |     nan        |

## Average across Calibration Datasets (per Task)

|                                                                           |   original |   pruning |   quantization |
|:--------------------------------------------------------------------------|-----------:|----------:|---------------:|
| ('Qwen/Qwen3-1.7B', 'arc_challenge', 'acc,none', 0.5, 128)                |   0.398464 |  0.277944 |     nan        |
| ('Qwen/Qwen3-1.7B', 'arc_challenge', 'acc_norm,none', 0.5, 128)           |   0.429181 |  0.307594 |     nan        |
| ('Qwen/Qwen3-1.7B', 'boolq', 'acc,none', 0.5, 128)                        |   0.774924 |  0.655841 |     nan        |
| ('Qwen/Qwen3-1.7B', 'winogrande', 'acc,none', 0.5, 128)                   |   0.610892 |  0.570481 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 128)                 |   0.55802  |  0.539756 |       0.525171 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 512)                 |   0.55802  |  0.538903 |       0.542235 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.25, 1024)                |   0.55802  |  0.538263 |       0.536689 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 128)                  |   0.55802  |  0.428742 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 512)                  |   0.55802  |  0.412067 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc,none', 0.5, 1024)                 |   0.55802  |  0.436998 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 128)            |   0.564846 |  0.553834 |       0.535836 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 512)            |   0.564846 |  0.553701 |       0.547355 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.25, 1024)           |   0.564846 |  0.553888 |       0.537543 |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 128)             |   0.564846 |  0.451097 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 512)             |   0.564846 |  0.432204 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_challenge', 'acc_norm,none', 0.5, 1024)            |   0.564846 |  0.454971 |     nan        |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 128)                      |   0.835859 |  0.811448 |       0.811658 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 512)                      |   0.835859 |  0.812487 |       0.820918 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc,none', 0.25, 1024)                     |   0.835859 |  0.811829 |       0.810606 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 128)                 |   0.809343 |  0.790207 |       0.785564 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 512)                 |   0.809343 |  0.791246 |       0.788089 |
| ('Qwen/Qwen3-8B', 'arc_easy', 'acc_norm,none', 0.25, 1024)                |   0.809343 |  0.791377 |       0.786406 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 128)                         |   0.866055 |  0.848834 |       0.859786 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 512)                         |   0.866055 |  0.848901 |       0.867431 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.25, 1024)                        |   0.866055 |  0.848786 |       0.867584 |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 128)                          |   0.866055 |  0.780087 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 512)                          |   0.866055 |  0.758078 |     nan        |
| ('Qwen/Qwen3-8B', 'boolq', 'acc,none', 0.5, 1024)                         |   0.866055 |  0.773967 |     nan        |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 128)                     |   0.571301 |  0.553102 |       0.555965 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 512)                     |   0.571301 |  0.552816 |       0.561591 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc,none', 0.25, 1024)                    |   0.571301 |  0.55309  |       0.562189 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 128)                |   0.749054 |  0.727009 |       0.733569 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 512)                |   0.749054 |  0.727252 |       0.737353 |
| ('Qwen/Qwen3-8B', 'hellaswag', 'acc_norm,none', 0.25, 1024)               |   0.749054 |  0.727261 |       0.738498 |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 128)                    |   0.31     |  0.311563 |       0.318    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 512)                    |   0.31     |  0.312125 |       0.303    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc,none', 0.25, 1024)                   |   0.31     |  0.31075  |       0.311    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 128)               |   0.414    |  0.413687 |       0.407    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 512)               |   0.414    |  0.413937 |       0.409    |
| ('Qwen/Qwen3-8B', 'openbookqa', 'acc_norm,none', 0.25, 1024)              |   0.414    |  0.4135   |       0.409    |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 128)                           |   0.783394 |  0.752369 |       0.772563 |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 512)                           |   0.783394 |  0.751241 |       0.763538 |
| ('Qwen/Qwen3-8B', 'rte', 'acc,none', 0.25, 1024)                          |   0.783394 |  0.751467 |       0.759928 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 128)                    |   0.67719  |  0.679213 |       0.679953 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 512)                    |   0.67719  |  0.680101 |       0.689029 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.25, 1024)                   |   0.67719  |  0.681013 |       0.685872 |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 128)                     |   0.67719  |  0.647897 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 512)                     |   0.67719  |  0.633668 |     nan        |
| ('Qwen/Qwen3-8B', 'winogrande', 'acc,none', 0.5, 1024)                    |   0.67719  |  0.650177 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 128)               |   0.498294 |  0.501588 |       0.475683 |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 512)               |   0.498294 |  0.502015 |       0.474829 |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.25, 1024)              |   0.498294 |  0.501474 |       0.46843  |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 128)                |   0.498294 |  0.333345 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 512)                |   0.498294 |  0.334403 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc,none', 0.5, 1024)               |   0.498294 |  0.330205 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 128)          |   0.538396 |  0.53503  |       0.503413 |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 512)          |   0.538396 |  0.534272 |       0.495734 |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.25, 1024)         |   0.538396 |  0.535112 |       0.50128  |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 128)           |   0.538396 |  0.350512 |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 512)           |   0.538396 |  0.35413  |     nan        |
| ('google/gemma-7b', 'arc_challenge', 'acc_norm,none', 0.5, 1024)          |   0.538396 |  0.347076 |     nan        |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 128)                    |   0.826178 |  0.814219 |       0.758838 |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 512)                    |   0.826178 |  0.814429 |       0.766204 |
| ('google/gemma-7b', 'arc_easy', 'acc,none', 0.25, 1024)                   |   0.826178 |  0.815567 |       0.773148 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 128)               |   0.808502 |  0.797536 |       0.738636 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 512)               |   0.808502 |  0.797325 |       0.748316 |
| ('google/gemma-7b', 'arc_easy', 'acc_norm,none', 0.25, 1024)              |   0.808502 |  0.797942 |       0.749579 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 128)                       |   0.836086 |  0.818688 |       0.749847 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 512)                       |   0.836086 |  0.820217 |       0.780887 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.25, 1024)                      |   0.836086 |  0.81959  |       0.765902 |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 128)                        |   0.836086 |  0.600636 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 512)                        |   0.836086 |  0.605162 |     nan        |
| ('google/gemma-7b', 'boolq', 'acc,none', 0.5, 1024)                       |   0.836086 |  0.597845 |     nan        |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 128)                   |   0.606552 |  0.595051 |       0.569906 |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 512)                   |   0.606552 |  0.59548  |       0.578072 |
| ('google/gemma-7b', 'hellaswag', 'acc,none', 0.25, 1024)                  |   0.606552 |  0.595396 |       0.578968 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 128)              |   0.8095   |  0.795542 |       0.762298 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 512)              |   0.8095   |  0.795382 |       0.779526 |
| ('google/gemma-7b', 'hellaswag', 'acc_norm,none', 0.25, 1024)             |   0.8095   |  0.795845 |       0.778381 |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 128)                  |   0.32     |  0.326944 |       0.295    |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 512)                  |   0.32     |  0.326222 |       0.302    |
| ('google/gemma-7b', 'openbookqa', 'acc,none', 0.25, 1024)                 |   0.32     |  0.327212 |       0.314    |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 128)             |   0.442    |  0.451222 |       0.422    |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 512)             |   0.442    |  0.451778 |       0.442    |
| ('google/gemma-7b', 'openbookqa', 'acc_norm,none', 0.25, 1024)            |   0.442    |  0.452    |       0.439    |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 128)                         |   0.6787   |  0.607802 |       0.628159 |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 512)                         |   0.6787   |  0.604894 |       0.613718 |
| ('google/gemma-7b', 'rte', 'acc,none', 0.25, 1024)                        |   0.6787   |  0.60617  |       0.644404 |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 128)                  |   0.753749 |  0.737153 |       0.725335 |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 512)                  |   0.753749 |  0.73735  |       0.725335 |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.25, 1024)                 |   0.753749 |  0.737581 |       0.722573 |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 128)                   |   0.753749 |  0.607893 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 512)                   |   0.753749 |  0.604767 |     nan        |
| ('google/gemma-7b', 'winogrande', 'acc,none', 0.5, 1024)                  |   0.753749 |  0.597725 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 128)       |   0.427474 |  0.422377 |       0.405717 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 512)       |   0.427474 |  0.423051 |       0.415529 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.25, 1024)      |   0.427474 |  0.423747 |       0.420222 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 128)        |   0.427474 |  0.306723 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 512)        |   0.427474 |  0.305443 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc,none', 0.5, 1024)       |   0.427474 |  0.296839 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 128)  |   0.462457 |  0.451141 |       0.442833 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 512)  |   0.462457 |  0.450916 |       0.458191 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.25, 1024) |   0.462457 |  0.451455 |       0.453925 |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 128)   |   0.462457 |  0.331111 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 512)   |   0.462457 |  0.331147 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_challenge', 'acc_norm,none', 0.5, 1024)  |   0.462457 |  0.324704 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 128)            |   0.744949 |  0.732523 |       0.73064  |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 512)            |   0.744949 |  0.731404 |       0.734428 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc,none', 0.25, 1024)           |   0.744949 |  0.732334 |       0.738215 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 128)       |   0.720539 |  0.705166 |       0.695497 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 512)       |   0.720539 |  0.704213 |       0.701178 |
| ('meta-llama/Llama-3.2-3B', 'arc_easy', 'acc_norm,none', 0.25, 1024)      |   0.720539 |  0.704468 |       0.706229 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 128)               |   0.74159  |  0.727716 |       0.722936 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 512)               |   0.74159  |  0.726469 |       0.742355 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.25, 1024)              |   0.74159  |  0.727563 |       0.737462 |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 128)                |   0.74159  |  0.633582 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 512)                |   0.74159  |  0.631148 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'boolq', 'acc,none', 0.5, 1024)               |   0.74159  |  0.618775 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 128)           |   0.558156 |  0.547094 |       0.539584 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 512)           |   0.558156 |  0.547212 |       0.544115 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc,none', 0.25, 1024)          |   0.558156 |  0.547086 |       0.543268 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 128)      |   0.741585 |  0.730503 |       0.722466 |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 512)      |   0.741585 |  0.730694 |       0.72864  |
| ('meta-llama/Llama-3.2-3B', 'hellaswag', 'acc_norm,none', 0.25, 1024)     |   0.741585 |  0.730506 |       0.726997 |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 128)          |   0.312    |  0.309474 |       0.296    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 512)          |   0.312    |  0.311263 |       0.309    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc,none', 0.25, 1024)         |   0.312    |  0.311421 |       0.305    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 128)     |   0.41     |  0.385632 |       0.411    |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 512)     |   0.41     |  0.386421 |       0.4      |
| ('meta-llama/Llama-3.2-3B', 'openbookqa', 'acc_norm,none', 0.25, 1024)    |   0.41     |  0.388316 |       0.407    |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 128)                 |   0.541516 |  0.562037 |       0.465704 |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 512)                 |   0.541516 |  0.566312 |       0.516245 |
| ('meta-llama/Llama-3.2-3B', 'rte', 'acc,none', 0.25, 1024)                |   0.541516 |  0.562987 |       0.536101 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 128)          |   0.693765 |  0.700951 |       0.683504 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 512)          |   0.693765 |  0.699726 |       0.691792 |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.25, 1024)         |   0.693765 |  0.700536 |       0.6985   |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 128)           |   0.693765 |  0.618373 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 512)           |   0.693765 |  0.619919 |     nan        |
| ('meta-llama/Llama-3.2-3B', 'winogrande', 'acc,none', 0.5, 1024)          |   0.693765 |  0.611598 |     nan        |

## Global Average (Across Tasks and Calibration Datasets)

|                                         |   original |   pruning |   quantization |
|:----------------------------------------|-----------:|----------:|---------------:|
| ('Qwen/Qwen3-1.7B', 0.5, 128)           |   0.56887  |  0.470765 |     nan        |
| ('Qwen/Qwen3-8B', 0.25, 128)            |   0.649006 |  0.634638 |       0.635006 |
| ('Qwen/Qwen3-8B', 0.25, 512)            |   0.649006 |  0.634792 |       0.639049 |
| ('Qwen/Qwen3-8B', 0.25, 1024)           |   0.649006 |  0.634657 |       0.636847 |
| ('Qwen/Qwen3-8B', 0.5, 128)             |   0.666528 |  0.576956 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 512)             |   0.666528 |  0.559004 |     nan        |
| ('Qwen/Qwen3-8B', 0.5, 1024)            |   0.666528 |  0.579028 |     nan        |
| ('google/gemma-7b', 0.25, 128)          |   0.647087 |  0.634616 |       0.602647 |
| ('google/gemma-7b', 0.25, 512)          |   0.647087 |  0.634488 |       0.609693 |
| ('google/gemma-7b', 0.25, 1024)         |   0.647087 |  0.634899 |       0.612333 |
| ('google/gemma-7b', 0.5, 128)           |   0.656631 |  0.473096 |     nan        |
| ('google/gemma-7b', 0.5, 512)           |   0.656631 |  0.474615 |     nan        |
| ('google/gemma-7b', 0.5, 1024)          |   0.656631 |  0.468213 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.25, 128)  |   0.577639 |  0.570419 |       0.555989 |
| ('meta-llama/Llama-3.2-3B', 0.25, 512)  |   0.577639 |  0.570698 |       0.567407 |
| ('meta-llama/Llama-3.2-3B', 0.25, 1024) |   0.577639 |  0.570947 |       0.570265 |
| ('meta-llama/Llama-3.2-3B', 0.5, 128)   |   0.581322 |  0.472447 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 512)   |   0.581322 |  0.471914 |     nan        |
| ('meta-llama/Llama-3.2-3B', 0.5, 1024)  |   0.581322 |  0.462979 |     nan        |