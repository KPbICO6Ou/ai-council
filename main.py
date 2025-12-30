#!/usr/bin/env python3
"""AI Council - a system for evaluating responses from multiple models."""
import sys
import json
import logging
import random
import os
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from ollama import OllamaClient

load_dotenv()

# Configure logging
logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stderr),
    ],
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s]: (%(name)s.%(funcName)s) - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_ollama_url() -> str:
    """Get Ollama API URL from environment."""
    return os.getenv("OLLAMA_URL", "http://localhost:11434")


def get_models() -> List[str]:
    """Get list of models from environment."""
    models_str = os.getenv("MODELS", "llama3.2,llama3.1,mixtral,qwen2.5,phi3")
    models = [m.strip() for m in models_str.split(",")]
    if len(models) != 5:
        raise ValueError(f"Exactly 5 models required, got {len(models)}: {models}")
    return models


def get_moderator() -> Optional[str]:
    """Get moderator model from environment. Returns None if not set."""
    moderator = os.getenv("MODERATOR", "").strip()
    return moderator if moderator else None


class AICouncil:
    """A council of models for evaluating responses."""

    def __init__(
        self, models: List[str], ollama_url: str, moderator: Optional[str] = None
    ):
        self.models = models
        self.client = OllamaClient(ollama_url)
        self.moderator = moderator

        # Load Jinja2 templates
        template_dir = Path(__file__).parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)
        self.template_prompt = env.get_template("model_question.j2")
        self.template_criteria = env.get_template("moderator_criteria.j2")
        self.template_evaluation = env.get_template("model_evaluation.j2")
        self.template_moderator = env.get_template("moderator_decision.j2")

    def send_to_all_models(self, user_question: str) -> Dict[str, str]:
        """Sends a request to all models and returns the answers."""
        logger.info("Sending request to all models...")

        # Render the prompt from the template
        prompt = self.template_prompt.render(user_question=user_question)

        answers = {}

        for model in self.models:
            logger.info(f"Querying model: {model}")
            answer = self.client.generate(model, prompt)
            answers[model] = answer
            logger.info(
                f"Answer from {model} received (length: {len(answer)} characters)"
            )
            # Output to console (stderr via logging)
            logger.info(f"Answer from {model}:\n{answer}")

        return answers

    def select_moderator(self) -> str:
        """Selects a council moderator (from config or randomly)."""
        if self.moderator is None:
            self.moderator = random.choice(self.models)
            logger.info(f"Council moderator selected randomly: {self.moderator}")
        else:
            # Check if the moderator is in the list of models
            if self.moderator not in self.models:
                logger.warning(
                    f"Moderator {self.moderator} not found in the list of models, selecting randomly"
                )
                self.moderator = random.choice(self.models)
            else:
                logger.info(f"Council moderator set from config: {self.moderator}")
        return self.moderator

    def get_evaluation_criteria(self, user_question: str) -> List[str]:
        """Gets 5 evaluation criteria from the moderator."""
        # Render the prompt from the template
        prompt = self.template_criteria.render(user_question=user_question)

        logger.info(
            f"Requesting evaluation criteria from moderator {self.moderator}..."
        )
        response = self.client.generate(self.moderator, prompt)

        # Parse criteria (one per line)
        criteria = [c.strip() for c in response.strip().split("\n") if c.strip()]
        # Take the first 5
        criteria = criteria[:5]

        # If there are fewer than 5 criteria, add more
        while len(criteria) < 5:
            criteria.append(f"Criterion {len(criteria) + 1}")

        logger.info(f"Received evaluation criteria: {criteria}")
        criteria_log_message = (
            f"\nEvaluation criteria from moderator {self.moderator}:\n"
        )
        for i, criterion in enumerate(criteria, 1):
            criteria_log_message += f"{i}. {criterion}\n"
        logger.info(criteria_log_message)

        return criteria

    def evaluate_answer(
        self, model: str, user_question: str, answer: str, criteria: List[str]
    ) -> Dict[str, int]:
        """Evaluates an answer based on criteria using a single model."""
        # Render the prompt from the template
        prompt = self.template_evaluation.render(
            user_question=user_question, answer=answer, criteria=criteria
        )

        response = self.client.generate(model, prompt)

        # Try to extract JSON from the response
        scores = {}
        try:
            # Find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                scores = json.loads(json_str)
            else:
                # If JSON is not found, try alternative parsing
                logger.warning(
                    f"Could not find JSON in the response from {model}, trying alternative parsing"
                )
                # Parse lines like "criterion: score"
                for line in response.split("\n"):
                    if ":" in line:
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            key = parts[0].strip().strip("\"'{}")
                            try:
                                value = int(parts[1].strip().strip(",}"))
                                if 1 <= value <= 10:
                                    scores[key] = value
                            except ValueError:
                                pass
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from {model}: {e}")
            logger.debug(f"The response was: {response[:200]}")

        # If scores could not be obtained, use default values
        if not scores:
            logger.warning(f"Could not get scores from {model}, using default values")
            scores = {criterion: 5 for criterion in criteria}
        else:
            # Normalize keys (remove extra characters, match to criteria names)
            normalized_scores = {}
            for criterion in criteria:
                # Find a similar key in scores
                found = False
                for key, value in scores.items():
                    if (
                        criterion.lower() in key.lower()
                        or key.lower() in criterion.lower()
                    ):
                        normalized_scores[criterion] = max(1, min(10, int(value)))
                        found = True
                        break
                if not found:
                    normalized_scores[criterion] = 5

            scores = normalized_scores

        # Ensure all criteria are scored
        for criterion in criteria:
            if criterion not in scores:
                scores[criterion] = 5

        return scores

    def evaluate_all_answers(
        self, user_question: str, answers: Dict[str, str], criteria: List[str]
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Evaluates all answers with all models."""
        logger.info("Starting evaluation of all answers by all models...")
        all_evaluations = {}

        for answer_model, answer_text in answers.items():
            logger.info(f"Evaluating answer from {answer_model}...")
            model_evaluations = {}

            for evaluator_model in self.models:
                logger.info(f"  Evaluation from {evaluator_model}...")
                scores = self.evaluate_answer(
                    evaluator_model, user_question, answer_text, criteria
                )
                model_evaluations[evaluator_model] = scores

                # Log the scores
                total = sum(scores.values())
                evaluation_log = f"\nEvaluation of answer from {answer_model} by model {evaluator_model}:\n"
                evaluation_log += (
                    f"\nQuestion:\n{user_question}\n\nAnswer:\n{answer_text}\n"
                )
                for criterion, score in scores.items():
                    evaluation_log += f"  {criterion}: {score}/10\n"
                evaluation_log += f"  Total: {total}/50"
                logger.info(evaluation_log)

            all_evaluations[answer_model] = model_evaluations

        return all_evaluations

    def calculate_total_scores(
        self, evaluations: Dict[str, Dict[str, Dict[str, int]]]
    ) -> Dict[str, int]:
        """Calculates the total scores for each answer."""
        total_scores = {}

        for answer_model, model_evaluations in evaluations.items():
            total = 0
            for evaluator_scores in model_evaluations.values():
                total += sum(evaluator_scores.values())
            total_scores[answer_model] = total

        logger.info(f"Total scores: {total_scores}")
        return total_scores

    def get_final_decision(
        self, user_question: str, answers: Dict[str, str], total_scores: Dict[str, int]
    ) -> str:
        """Gets the final decision from the moderator."""
        # Render the prompt from the template
        prompt = self.template_moderator.render(
            user_question=user_question, answers=answers, total_scores=total_scores
        )

        logger.info(f"Requesting final decision from moderator {self.moderator}...")
        logger.info(f"Total scores for moderator: {total_scores}")

        decision = self.client.generate(self.moderator, prompt)

        # Clean the response from possible introductions
        decision = decision.strip()
        for prefix in ["Best answer:", "Chosen answer:", "Answer:", "Decision:"]:
            if decision.startswith(prefix):
                decision = decision[len(prefix) :].strip()

        # Remove duplication if the text is repeated
        decision_clean = decision.strip()
        if len(decision_clean) > 50:
            # Normalize the text (remove extra spaces and newlines)
            normalized = " ".join(decision_clean.split())
            # Check if the text is duplicated (first half = second half)
            mid_point = len(normalized) // 2
            if mid_point > 25:
                first_half = normalized[:mid_point].strip()
                second_half = normalized[mid_point:].strip()
                # If the second half starts with the first (considering possible spaces), remove duplication
                if first_half and second_half:
                    # Compare the first 100 characters or the entire first half
                    compare_len = min(100, len(first_half))
                    if second_half[:compare_len] == first_half[:compare_len]:
                        # Take the first half of the original text
                        decision = decision_clean[: len(decision_clean) // 2].strip()

        logger.info("Final decision received from the moderator")
        return decision

    def run(self, user_question: str):
        """Runs the full AI Council process."""
        logger.info("=" * 60)
        logger.info("Starting AI Council")
        logger.info(f"User question: {user_question}")
        logger.info("=" * 60)

        # Step 1: Send request to all models
        answers = self.send_to_all_models(user_question)

        # Step 2: Select moderator
        self.select_moderator()

        # Step 3: Get evaluation criteria
        criteria = self.get_evaluation_criteria(user_question)

        # Step 4: Evaluate all answers with all models
        evaluations = self.evaluate_all_answers(user_question, answers, criteria)

        # Step 5: Calculate total scores
        total_scores = self.calculate_total_scores(evaluations)

        # Log total scores
        scores_log = "\nTotal scores for all answers:\n"
        for model, score in sorted(
            total_scores.items(), key=lambda x: x[1], reverse=True
        ):
            scores_log += f"{model}: {score}/250\n"
        logger.info(scores_log)

        # Step 6: Final decision from the moderator
        best_answer = self.get_final_decision(user_question, answers, total_scores)

        # Determine the winning model by the highest score
        best_model = max(total_scores.items(), key=lambda x: x[1])[0]

        # Log final information
        final_log = f"\nModerator: {self.moderator}\n"
        final_log += f"Best answer from model: {best_model}\n"
        final_log += f"\nQuestion:\n{user_question}\n\nAnswer:\n{best_answer}"
        logger.info(final_log)

        # Print the best answer to stdout
        print(best_answer, file=sys.stdout)

        logger.info("=" * 60)
        logger.info("AI Council finished")
        logger.info("=" * 60)


def main():
    """Main function."""
    try:
        # Load configuration
        ollama_url = get_ollama_url()
        models = get_models()
        moderator = get_moderator()

        logger.info(f"Ollama URL: {ollama_url}")
        logger.info(f"Models: {models}")
        if moderator:
            logger.info(f"Moderator specified: {moderator}")
        else:
            logger.info("Moderator will be selected randomly")

        # Get question from user
        if len(sys.argv) > 1:
            # Question from command line arguments
            question = " ".join(sys.argv[1:])
        else:
            # Question from stdin
            logger.info("Enter your question:")
            question = input()

        if not question.strip():
            logger.error("Question cannot be empty")
            sys.exit(1)

        # Run the AI Council
        council = AICouncil(models, ollama_url, moderator)
        council.run(question)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
