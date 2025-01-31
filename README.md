# Unlearning Source Exclusive Classes in Domain Adaptation via Adversarial Optimization

## Setup

### Requirements
Ensure you have Python **3.12.2** installed.

#### Required Packages:
```txt
matplotlib==3.9.2
numba==0.59.1
numpy==1.26.4
pillow==10.4.0
prettytable==3.10.0
scikit-learn==1.5.0
scipy==1.14.1
timm==1.0.3
torch==2.4.1+cu118
torchvision==0.19.1+cu118
tqdm==4.66.2
```

## Dataset Setup
Download the **OfficeHome** dataset from [this link](https://www.hemanthdv.org/officeHomeDataset.html) and place it in the `data/OfficeHome` directory.

Alternatively, if you already have the dataset stored elsewhere, update the `data_path` in the configuration dictionary found in `main.py`.

---

## Running DASEC Unlearning Methods on the OfficeHome Dataset

### Running the Proposed Method
To run the proposed method for classes `{1,2,3}` on a single OfficeHome Task, say **(Art → Product)**, use the following command:

```bash
python main.py -d OfficeHome -s Art -t Product -m minimax -fc 1,2,3
```

### Running the proposed method for All Tasks
To run the proposed method for classes `{1,2,3}` across all tasks in OfficeHome, execute:

```bash
bash minimax.sh
```

### Running Baseline Methods
To run baseline models, execute:

```bash
bash baselines.sh
```
