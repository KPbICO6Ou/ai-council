#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ollama client for AI Council project."""
import requests
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = None):
        """Initialize Ollama client.
        
        Args:
            base_url (str, optional): Base URL for Ollama API. 
                                     If None, reads from OLLAMA_URL env variable.
        """
        if base_url is None:
            self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip('/')
        else:
            self.base_url = base_url.rstrip('/')
        logger.info(f"Ollama Client initialized with URL: {self.base_url}")

    def list_models(self):
        """Retrieve list of available models from Ollama server.
        
        Returns:
            list: List of model names available on the server. Returns empty list on error.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate(self, model: str, prompt: str) -> str:
        """Generate text response from Ollama API.
        
        Args:
            model (str): Name of the model to use for generation.
            prompt (str): Text prompt to send to the model.
        
        Returns:
            str: Generated response text, or error message if request fails.
        """
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        logger.debug(f"Sending request to {url} with model {model}")
        logger.debug(f"Prompt: {prompt[:200]}...")
        
        try:
            response = requests.post(url, json=data, timeout=300)
            logger.debug(f"Response Status: {response.status_code} {response.reason}")
            
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "").strip()
            logger.debug(f"Response length: {len(response_text)} characters")
            return response_text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Ollama: {e}")
            return f"Ошибка: {str(e)}"
