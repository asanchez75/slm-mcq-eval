# Evaluating Small and Medium Language Models for High-Quality Multi-Choice Question Generation

## Project Structure
```
raiforukraine-MARIA/
├── data/                   # Data: lisa sheets, generated mcqs
├── notebooks/              # Jupyter notebooks
├── src/                    
│   ├── eval/               # Evaluation criterion
│   ├── ollama/             # Inference and convertion with Ollama
│   ├── prepare_data/       # Data preparation
│   ├── __init__.py         
│   ├── dpo_training_runner.py  # Direct Preference Optimization training script
│   ├── inference.py        # Model inference
│   ├── main.py             # Main application entry point
│   └── rag.py              # Retrieval-Augmented Generation
├── LICENSE                 # Project license
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
└── run_dpo.sh              # Shell script to run DPO training
```

## Dataset
The Lisa sheets are available at ```data\lisa_sheets```.

They were split into folders, and this is available in ```data\train_test_split```.
You can determine whether a file is for training or testing by the folder name.

## Installation

### Prerequisites
```
Python =< 3.10
```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Its-OP/raiforukraine-MARIA.git
   cd raiforukraine-MARIA
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Create a `.env` file in the project root directory
   - Add the following required variables:
     ```
     OPENAI_API_KEY=Your_open_ai_key
     LISA_SHEETS_PATH=path_to_lisa_sheet_csv
     MODEL_MCQ_PATH=path_to_generated_mcqs_csv
     MODEL_MCQ_EVAL_EXPORT_PATH=path_to_export_csv
     ```

## License
This project is licensed under the terms of the LICENSE file included in the repository.
