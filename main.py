#!/usr/bin/env python3
"""AI Council - система оценки ответов от нескольких моделей."""
import sys
import json
import logging
import random
import os
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, Template
from ollama import OllamaClient

load_dotenv()

# Configure logging
logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stderr),
    ],
    level=logging.WARNING,
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
    """Совет моделей для оценки ответов."""
    
    def __init__(self, models: List[str], ollama_url: str, moderator: Optional[str] = None):
        self.models = models
        self.client = OllamaClient(ollama_url)
        self.moderator = moderator
        
        # Загрузка шаблонов Jinja2
        template_dir = Path(__file__).parent / 'templates'
        env = Environment(loader=FileSystemLoader(template_dir), autoescape=False)
        self.template_prompt = env.get_template('model_question.j2')
        self.template_criteria = env.get_template('moderator_criteria.j2')
        self.template_evaluation = env.get_template('model_evaluation.j2')
        self.template_moderator = env.get_template('moderator_decision.j2')
    
    def send_to_all_models(self, user_question: str) -> Dict[str, str]:
        """Отправляет запрос всем моделям и возвращает ответы."""
        logger.info("Отправка запроса всем моделям...")
        
        # Формируем промпт из шаблона
        prompt = self.template_prompt.render(
            user_question=user_question
        )
        
        answers = {}
        
        for model in self.models:
            logger.info(f"Запрос к модели: {model}")
            answer = self.client.generate(model, prompt)
            answers[model] = answer
            logger.info(f"Ответ от {model} получен (длина: {len(answer)} символов)")
            # Вывод в консоль (stderr через logging)
            print(f"\nОтвет от {model}:", file=sys.stderr)
            print(answer, file=sys.stderr)
        
        return answers
    
    def select_moderator(self) -> str:
        """Выбирает модератора совета (из конфига или случайно)."""
        if self.moderator is None:
            self.moderator = random.choice(self.models)
            logger.info(f"Модератор совета выбран случайно: {self.moderator}")
        else:
            # Проверяем, что модератор в списке моделей
            if self.moderator not in self.models:
                logger.warning(f"Модератор {self.moderator} не найден в списке моделей, выбираю случайно")
                self.moderator = random.choice(self.models)
            else:
                logger.info(f"Модератор совета задан из конфига: {self.moderator}")
        return self.moderator
    
    def get_evaluation_criteria(self, user_question: str) -> List[str]:
        """Получает 5 критериев оценки от модератора."""
        # Формируем промпт из шаблона
        prompt = self.template_criteria.render(
            user_question=user_question
        )
        
        logger.info(f"Запрос критериев оценки у модератора {self.moderator}...")
        response = self.client.generate(self.moderator, prompt)
        
        # Парсим критерии (по одному на строку)
        criteria = [c.strip() for c in response.strip().split('\n') if c.strip()]
        # Берем первые 5
        criteria = criteria[:5]
        
        # Если критериев меньше 5, дополняем
        while len(criteria) < 5:
            criteria.append(f"Критерий {len(criteria) + 1}")
        
        logger.info(f"Получены критерии оценки: {criteria}")
        print(f"\nКритерии оценки от модератора {self.moderator}:", file=sys.stderr)
        for i, criterion in enumerate(criteria, 1):
            print(f"{i}. {criterion}", file=sys.stderr)
        
        return criteria
    
    def evaluate_answer(self, model: str, user_question: str, answer: str, criteria: List[str]) -> Dict[str, int]:
        """Оценивает ответ по критериям одной моделью."""
        # Формируем промпт из шаблона
        prompt = self.template_evaluation.render(
            user_question=user_question,
            answer=answer,
            criteria=criteria
        )
        
        response = self.client.generate(model, prompt)
        
        # Пытаемся извлечь JSON из ответа
        scores = {}
        try:
            # Ищем JSON в ответе
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                scores = json.loads(json_str)
            else:
                # Если JSON не найден, пытаемся парсить по-другому
                logger.warning(f"Не удалось найти JSON в ответе от {model}, пытаемся альтернативный парсинг")
                # Парсим по строкам вида "критерий: оценка"
                for line in response.split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().strip('"\'{}')
                            try:
                                value = int(parts[1].strip().strip(',}'))
                                if 1 <= value <= 10:
                                    scores[key] = value
                            except ValueError:
                                pass
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от {model}: {e}")
            logger.debug(f"Ответ был: {response[:200]}")
        
        # Если не удалось получить оценки, ставим средние значения
        if not scores:
            logger.warning(f"Не удалось получить оценки от {model}, используем значения по умолчанию")
            scores = {criterion: 5 for criterion in criteria}
        else:
            # Нормализуем ключи (убираем лишние символы, приводим к названиям критериев)
            normalized_scores = {}
            for criterion in criteria:
                # Ищем похожий ключ в scores
                found = False
                for key, value in scores.items():
                    if criterion.lower() in key.lower() or key.lower() in criterion.lower():
                        normalized_scores[criterion] = max(1, min(10, int(value)))
                        found = True
                        break
                if not found:
                    normalized_scores[criterion] = 5
            
            scores = normalized_scores
        
        # Убеждаемся, что все критерии оценены
        for criterion in criteria:
            if criterion not in scores:
                scores[criterion] = 5
        
        return scores
    
    def evaluate_all_answers(self, user_question: str, answers: Dict[str, str], criteria: List[str]) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Оценивает все ответы всеми моделями."""
        logger.info("Начало оценки всех ответов всеми моделями...")
        all_evaluations = {}
        
        for answer_model, answer_text in answers.items():
            logger.info(f"Оценка ответа от {answer_model}...")
            model_evaluations = {}
            
            for evaluator_model in self.models:
                logger.info(f"  Оценка от {evaluator_model}...")
                scores = self.evaluate_answer(evaluator_model, user_question, answer_text, criteria)
                model_evaluations[evaluator_model] = scores
                
                # Выводим оценки в лог
                total = sum(scores.values())
                print(f"\nОценка ответа от {answer_model} моделью {evaluator_model}:", file=sys.stderr)
                print(f"\nЗапрос:\n{user_question}\n\nОтвет:\n{answer_text}\n", file=sys.stderr)
                for criterion, score in scores.items():
                    print(f"  {criterion}: {score}/10", file=sys.stderr)
                print(f"  Итого: {total}/50", file=sys.stderr)
            
            all_evaluations[answer_model] = model_evaluations
        
        return all_evaluations
    
    def calculate_total_scores(self, evaluations: Dict[str, Dict[str, Dict[str, int]]]) -> Dict[str, int]:
        """Вычисляет суммарные оценки для каждого ответа."""
        total_scores = {}
        
        for answer_model, model_evaluations in evaluations.items():
            total = 0
            for evaluator_scores in model_evaluations.values():
                total += sum(evaluator_scores.values())
            total_scores[answer_model] = total
        
        logger.info(f"Суммарные оценки: {total_scores}")
        return total_scores
    
    def get_final_decision(self, user_question: str, answers: Dict[str, str], total_scores: Dict[str, int]) -> str:
        """Получает финальное решение от модератора."""
        # Формируем промпт из шаблона
        prompt = self.template_moderator.render(
            user_question=user_question,
            answers=answers,
            total_scores=total_scores
        )
        
        logger.info(f"Запрос финального решения у модератора {self.moderator}...")
        logger.info(f"Суммарные оценки для модератора: {total_scores}")
        
        decision = self.client.generate(self.moderator, prompt)
        
        # Очищаем ответ от возможных предисловий
        decision = decision.strip()
        for prefix in ["Лучший ответ:", "Выбранный ответ:", "Ответ:", "Решение:"]:
            if decision.startswith(prefix):
                decision = decision[len(prefix):].strip()
        
        # Удаляем дублирование, если текст повторяется дважды
        decision_clean = decision.strip()
        if len(decision_clean) > 50:
            # Нормализуем весь текст (убираем лишние пробелы и переносы)
            normalized = ' '.join(decision_clean.split())
            # Проверяем, не является ли текст дублированным (первая половина = второй)
            mid_point = len(normalized) // 2
            if mid_point > 25:
                first_half = normalized[:mid_point].strip()
                second_half = normalized[mid_point:].strip()
                # Если вторая половина начинается с первой (с учетом возможных пробелов), удаляем дублирование
                if first_half and second_half:
                    # Сравниваем первые 100 символов или всю первую половину
                    compare_len = min(100, len(first_half))
                    if second_half[:compare_len] == first_half[:compare_len]:
                        # Берем первую половину оригинального текста
                        decision = decision_clean[:len(decision_clean)//2].strip()
        
        logger.info("Финальное решение получено от модератора")
        return decision
    
    def run(self, user_question: str):
        """Запускает полный процесс совета моделей."""
        logger.info("=" * 60)
        logger.info("Запуск AI Council")
        logger.info(f"Вопрос пользователя: {user_question}")
        logger.info("=" * 60)
        
        # Шаг 1: Отправка запроса всем моделям
        answers = self.send_to_all_models(user_question)
        
        # Шаг 2: Выбор модератора
        moderator = self.select_moderator()
        
        # Шаг 3: Получение критериев оценки
        criteria = self.get_evaluation_criteria(user_question)
        
        # Шаг 4: Оценка всех ответов всеми моделями
        evaluations = self.evaluate_all_answers(user_question, answers, criteria)
        
        # Шаг 5: Вычисление суммарных оценок
        total_scores = self.calculate_total_scores(evaluations)
        
        # Вывод суммарных оценок в лог
        print(f"\nСуммарные оценки всех ответов:", file=sys.stderr)
        for model, score in sorted(total_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"{model}: {score}/250", file=sys.stderr)
        
        # Шаг 6: Финальное решение модератора
        best_answer = self.get_final_decision(user_question, answers, total_scores)
        
        # Определяем модель-победителя по максимальной оценке
        best_model = max(total_scores.items(), key=lambda x: x[1])[0]
        
        # Вывод финальной информации
        print(f"\nМодератор: {self.moderator}", file=sys.stderr)
        print(f"Лучший ответ: {best_model}", file=sys.stderr)
        print(f"\nЗапрос:\n{user_question}\n\nОтвет:\n{best_answer}", file=sys.stderr)
        
        # Вывод лучшего ответа в stdout
        print(best_answer, file=sys.stdout)
        
        logger.info("=" * 60)
        logger.info("AI Council завершил работу")
        logger.info("=" * 60)


def main():
    """Главная функция."""
    try:
        # Загрузка конфигурации
        ollama_url = get_ollama_url()
        models = get_models()
        moderator = get_moderator()
        
        logger.info(f"Ollama URL: {ollama_url}")
        logger.info(f"Модели: {models}")
        if moderator:
            logger.info(f"Модератор задан: {moderator}")
        else:
            logger.info("Модератор будет выбран случайно")
        
        # Получение вопроса от пользователя
        if len(sys.argv) > 1:
            # Вопрос из аргументов командной строки
            question = " ".join(sys.argv[1:])
        else:
            # Вопрос из stdin
            print("Введите ваш вопрос:", file=sys.stderr)
            question = input()
        
        if not question.strip():
            logger.error("Вопрос не может быть пустым")
            sys.exit(1)
        
        # Запуск совета моделей
        council = AICouncil(models, ollama_url, moderator)
        council.run(question)
        
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
