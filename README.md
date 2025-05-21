# CNN Interpretability Project

This project implements and evaluates different interpretability methods for Convolutional Neural Networks (CNNs) in computer vision tasks.

## Project Structure

```
proyecto_interpretabilidad/
├── data/                     # Dataset storage
│   ├── raw/                 # Original data
│   └── processed/           # Preprocessed data
├── notebooks/                # Jupyter notebooks for exploration and rapid prototyping
├── src/                     # Main source code
│   ├── models/             # CNN model definitions
│   ├── interpretability/   # Interpretability frameworks and methods
│   ├── training/          # Model training scripts
│   ├── evaluation/        # Model and interpretability evaluation
│   └── utils/             # Utility functions
├── configs/                 # Configuration files
│   ├── dataset_configs/    # Dataset configurations
│   ├── model_configs/      # Model configurations
│   └── experiment_configs/ # Experiment configurations
├── results/                 # Experiment results
│   ├── trained_models/     # Saved trained models
│   ├── interpretability_outputs/ # Visualization and method outputs
│   └── metrics/           # Evaluation metrics
└── tests/                  # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd proyecto_interpretabilidad
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

To train a model:

```bash
python -m src.training.train_model --config configs/model_configs/model1.yaml
```

### Interpretability Evaluation

To evaluate interpretability methods:

```bash
python -m src.evaluation.evaluate --model path/to/model --method lime
```

## Implemented Interpretability Methods

- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Grad-CAM (Gradient-weighted Class Activation Mapping)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. 