# FedHK-MVFC: Federated Heat Kernel Multi-View Clustering

**Repository Maintainer:** Kristina P. Sinaga (Independent Researcher)

> **Notice:** Coding modifications are currently underway in preparation for publication. The repository may change frequently during this process.

## Overview

This repository provides MATLAB implementations for HK-MVFC (Heat Kernel Multi-View Fuzzy Clustering) and its federated version, FedHK-MVFC, along with several baseline multi-view clustering algorithms. The implementations are based on the following papers:
1. Sinaga, K. P. (2025). FedHK-MVFC: Federated Heat Kernel Multi-View Clustering. **. [Link]()
                                                     |

## Dataset

The datasets used in this repository is generated synthetically using the `generate_synthetic_data.m` script. You can modify the parameters in the script to create datasets with different characteristics. 

In the original paper, the author used only synthetic datasets for evaluation. However, you can also use real-world datasets by loading them into MATLAB and formatting them as required by the clustering algorithms. To accomodate real-world datasets, you may need to preprocess the data to ensure that it is in the correct format (e.g., cell arrays for multi-view data). I added a placeholder function `load_real_world_data.m` that you can modify to load your own datasets. For this purpose, I used Iris datasets as an example. I encourage you to explore and experiment with different real-world datasets to evaluate the performance of the clustering algorithms.

I will consider adding some real-world datasets in future updates. In the meantime, feel free to contribute real-world datasets if you have any.

## Contact

**Kristina P. Sinaga**

Email: [kristinasinaga41@gmail.com](mailto:kristinasinaga41@gmail.com)

Website: [https://kristinap09.github.io/](https://kristinap09.github.io/)

## Citation
If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@article{sinaga2025fedhk,
  title={FedHK-MVFC: Federated Heat Kernel Multi-View Clustering},
  author={Sinaga, Kristina P},
  journal={},
  year={2025},
  publisher={}
}
```


