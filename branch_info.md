# GLBL Experiment Results Branch

## Branch Information
- **Branch Name**: `glbl-experiment-results`
- **Created From**: `cursor/reperform-gelu-glbl-experiment-and-plot-9105`
- **Purpose**: Clean branch containing GLBL experiment results without large files

## Excluded Files (Due to Size)
The following large files have been excluded from this branch to keep it lightweight:

### Data Files
- `data/MNIST/raw/*.ubyte` (45MB + 7.5MB MNIST image data)
- `data/MNIST/raw/*.gz` (9.5MB + 1.6MB compressed MNIST data)

### Generated Files
- `*.png` (visualization plots)
- `glbl_venv/` (virtual environment)

### Model Files
- `*.pt`, `*.pth`, `*.pkl` (PyTorch model files)
- `*.h5`, `*.hdf5` (Keras/HDF5 model files)

## Key Results Included
- **Enhanced GLBL experiment code** (`mlp-routing.py`)
- **Comprehensive analysis reports** (`.md` files)
- **Training configurations and improvements**
- **Performance metrics and comparisons**

## Achievement Summary
✅ **96%+ Accuracy Target Met**
- Standard MLP: 98.22% test accuracy
- GLBL MLP: 98.13% test accuracy
- Enhanced with loss/accuracy reporting every 2 epochs
- Optimized pathway structure (2×2×2 = 8 pathways)

## To Recreate Results
1. Create virtual environment: `python3 -m venv glbl_venv`
2. Install dependencies: `pip install torch torchvision matplotlib seaborn numpy scipy`
3. Run experiment: `python mlp-routing.py`
4. MNIST data will be automatically downloaded (~60MB)

## Files in This Branch
- Core experiment code and analysis
- Documentation and reports
- Configuration files
- No large binary files or data