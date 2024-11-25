# Polyps Segmentation

## Installation

Clone the repository:

```bash
git clone https://github.com/IamKrill1n/Polyps-Segmentation.git
```

Download the model from this link and put it in folder named model_checkpoint:

```bash
https://drive.google.com/file/d/1y75J_3kNfIf6l2243SdYOd1iDBjOHHKr/view?usp=sharing
```

Other models that were used in ensemble submission can be found in:
```bash
https://www.kaggle.com/datasets/doppelganger77/unetplusplus-resnet34/data
https://www.kaggle.com/datasets/anhtu77/unetplusplus-resnest
https://www.kaggle.com/datasets/doppelganger77/unetplusplus-resnet152/settings
```
## Usage

To run inference on an image, use the following command:

```bash
python3 infer.py --image_path image.jpeg
```
## Training

To checkout training details, switch to the `feature/train` branch: