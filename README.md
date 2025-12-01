# FedHK-MVFC: Federated Heat Kernel Multi-View Clustering

**Repository Maintainer:** Kristina P. Sinaga (Independent Researcher)  
**Paper:** [FedHK-MVFC: Federated Heat Kernel Multi-View Clustering](https://arxiv.org/abs/2509.15844) (arXiv:2509.15844)

> **Important Notice**  
> This repository is under active development in preparation for journal submission. Code and documentation may change frequently during this period.

## Overview

This repository contains MATLAB implementations of:
- **HK-MVFC** – Heat Kernel Multi-View Fuzzy Clustering  
- **FedHK-MVFC** – Its privacy-preserving federated learning extension  
- Several baseline multi-view clustering algorithms for comparison

The methods are introduced in the following paper:

**K. P. Sinaga**, "FedHK-MVFC: Federated Heat Kernel Multi-View Clustering," arXiv preprint arXiv:2509.15844, 2025.  
[arXiv link](https://arxiv.org/abs/2509.15844)

- Clean, well-commented MATLAB code
- Synthetic data generation with customizable parameters
- Support for real-world datasets (with example for Iris)
- Ready-to-run experiments and result visualization scripts
- Baseline methods included for fair comparison

## Datasets

### Synthetic Data
Synthetic multi-view datasets can be generated using `generate_synthetic_data.m`.  
Feel free to adjust parameters (number of views, clusters, noise level, etc.) to create datasets suited to your experiments.

### Real-World Data
The repository includes a template function `load_real_world_data.m`.  
As an example, the Iris dataset is already configured and ready to use.  
You can easily extend this function to load other popular multi-view datasets (e.g., BBC, Reuters, Caltech101-20, etc.) by formatting them as cell arrays.

Contributions of preprocessed real-world datasets are very welcome!


## Quick Start

```matlab
% 1. Generate or load data
data = generate_synthetic_data();        % or load_real_world_data()

% 2. Run FedHK-MVFC (example with 5 clients)
results = FedHK_MVFC(data, num_clusters, num_clients=5);

% 3. Evaluate and visualize
evaluate_clustering(results);
```

Detailed usage examples are provided in the demo/ folder.

## Citation
If you use this code or the FedHK-MVFC method in your research, please cite:

```citation
@misc{sinaga2025fedhkmvfc,
  title={FedHK-MVFC: Federated Heat Kernel Multi-View Clustering},
  author={Kristina P. Sinaga},
  year={2025},
  eprint={2509.15844},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.15844}
}
```


## Contact
Kristina P. Sinaga
Email (primary): kristinasinaga41@gmail.com

Email (will stay active forever, in case I lose access to my Gmail account or another one): kristinapestaria.sinaga@isti.cnr.it

Website: https://kristinap09.github.io/

GitHub: https://github.com/KristinaP09

Feel free to open an issue or contact me directly for questions, suggestions, or collaboration opportunities.


## License
MIT License, because I’m not a monster 😉

Thank you for your interest in FedHK-MVFC! ⭐ Star the repository if you find it useful.






