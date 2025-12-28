# AI Council

A system for evaluating responses from multiple AI models through a "council of models" mechanism.

## Description

AI Council sends your question to five models, receives their answers, then randomly selects a council moderator who defines the evaluation criteria. All models evaluate each answer based on these criteria, and the moderator selects the best answer.

## Installation

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Create a `.env` file in the project root:
    ```bash
    # Ollama API URL
    OLLAMA_URL=http://localhost:11434

    # Model names (comma-separated, exactly 5 models)
    MODELS=llama3.2,llama3.1,mixtral,qwen2.5,phi3

    # Moderator model (optional, if empty - random selection)
    MODERATOR=
    ```

## Usage

### Command-line execution

```bash
python main.py "Your question here"
```

Or interactive mode:
```bash
python main.py
# Then enter your question
```

### Output

-   **stdout**: The best answer selected by the moderator.
-   **stderr**: All process logs (model responses, ratings, criteria, etc.).

### Example

```bash
python main.py "How does quantum mechanics work?"
```

The best answer will be printed to stdout, and the entire council process will be logged to stderr.

## Configuration

All settings are configured via the `.env` file:

-   `OLLAMA_URL`: Your Ollama server URL (default: `http://localhost:11434`)
-   `MODELS`: A comma-separated list of exactly 5 models.
-   `MODERATOR`: The council moderator model (optional, if empty, one is chosen randomly from the list).

## Workflow

1.  Your question is sent to all 5 models.
2.  The answers are displayed in the console (stderr).
3.  A council moderator is selected (from the config or randomly).
4.  The moderator defines 5 evaluation criteria.
5.  Each answer is rated by all models according to the criteria (on a 10-point scale).
6.  Total scores are calculated.
7.  The moderator selects the best answer based on the scores.
8.  The best answer is printed to stdout.

