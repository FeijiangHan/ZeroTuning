# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training

Experience our latest version directly in Google Colab.

## Quick Start

### Prerequisites
- Python 3.8+
- transformers==4.53.2 (This specific version is required)

### Installation

1. Download our core components using gdown:
```bash
# Download our modified LLama model
!gdown --id 1JYr9Do94hfzc91NyxKBSJCwsTNw5ygK6

# Download evaluation pipeline
!gdown --id 17PhF8wGp9X5puN_kr0WzW7r5_ynlShXs

# Download evaluation configuration file
!gdown --id 1ZNbNV_ePNckVuNbkzQjJAqJoODy-GSW8

# Download and extract dataset
!rm -r ./data
!mkdir data
!gdown --id 1LjcvWQ84JqZQKMQrsxxIVnv0uug8hKWP -O data.zip
!unzip data.zip -d data
```

2. Import and open the notebook using the Colab 

3. Follow the step-by-step cells to test the model
4. (Optional) Experiment with your own inputs or datasets by modifying the code
