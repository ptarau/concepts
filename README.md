## Logic Driven Generative Concept Map Synthesis


We implement a recursive descent into the LLMs parametric knowledge store starting from a single seed question. We use a multi-agent architecture that constrains the LLM to return facts expressed as ground Prolog terms, and facts derived as generalizations of their key concepts. Iteration over prompts engineered via summarization of the generated Prolog facts and salient next question generation enables the synthesis of a focused concept map ready to be extended via logical reasoning.

Install with:

``
pip install -r requirements.txt
``

Then, run it with

``
python3 test.py
``


Make sure to have an *openai key* in your environment!

Alternatively, you can use ollama - see config.py for
changes in CF that redirect to a local LLM by adapting:


USE_OLLAMA = True

OLLAMA_MODEL = "gemma3:12b"

OLLAMA_BASE_URL = "http://localhost:11434/v1"


