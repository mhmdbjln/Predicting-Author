# Author Attribution with a Neural Network

This project trains a multi-class neural network to predict the author of a text sample from binary word-presence features. The implementation is provided in `Notebook/Authors.ipynb`.

## Repository contents

- `Notebook/Authors.ipynb`: preprocessing, model training, validation, and prediction workflow.
- `Data/train.csv`: labeled training data.
- `Data/test.csv`: separate test data.

## Method

- Separates the author label from binary word-feature columns.
- Encodes author labels using Scikit-learn's `LabelEncoder`.
- Uses a stratified 80/20 training-validation split with `random_state=42`.
- Builds a Keras Sequential neural network with Dense layers of 256, 128, and 64 units, ReLU activations, and Dropout layers (`0.3`).
- Uses a Softmax output layer, Adam optimizer, and sparse categorical cross-entropy loss.
- Trains for up to 20 epochs with a batch size of 32.
- Generates author predictions for the separate test dataset.

## Reported result

The evaluation cell computes a **macro F1-score of 0.838 (83.8%)** on the validation split.

**Metric clarification:** The notebook's introductory summary mentions 85.3% accuracy, while the explicit evaluation cell reports macro F1 = 0.838. The two metrics are not interchangeable; this README reports the metric computed by the evaluation code.

## Run

Install dependencies and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook Notebook/Authors.ipynb
```

The notebook's dataset path strings currently contain placeholders. Update them to point to `Data/train.csv` and `Data/test.csv` on your machine before running it.

## Notes

The current notebook displays predictions in a DataFrame; it does not save a trained model artifact. Confirm the dataset's source and redistribution terms before further distribution.

## License

No license has been specified.
