# FedTopoTuner - A Federated Topology-Guided Tuning Framework for Automated Proximity Graph Configuration

## Installation

Python >= 3.8 is recommended. Installing into a dedicated virtual environment is advised.

```bash
# 1. Deep learning framework
#    (torchvision is used for ResNet50 feature extraction, tensorboard for training logs)
#    For CUDA builds, pick the matching index-url from https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install tensorboard

# 2. Graph neural networks
pip install torch-geometric

# 3. Numerical computing and machine learning
pip install numpy pandas scipy scikit-learn

# 4. Approximate nearest neighbor search (proximity graph construction)
pip install faiss-cpu          # or faiss-gpu, which must match your local CUDA version

# 5. Graph analysis and plotting
pip install networkx matplotlib

# 6. Image I/O, progress bars, resource monitoring, and the web visualization server
pip install pillow tqdm psutil flask
```

Install everything in one command:

```bash
pip install torch torchvision tensorboard torch-geometric \
            numpy pandas scipy scikit-learn faiss-cpu \
            networkx matplotlib pillow tqdm psutil flask
```

### Run federated training

```bash
cd src_pgformer
python fedv3.py --aggregation_strategy performance_based --num_rounds 20
```