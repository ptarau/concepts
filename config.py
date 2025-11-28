import os


class CF:
    USE_OLLAMA = False
    OLLAMA_MODEL = "gemma3:12b"
    OLLAMA_BASE_URL = "http://u.local:11434/v1"
    API_KEY = os.getenv("OPENAI_API_KEY")
    GPT_MODEL = "gpt-5-mini"
    TOPN = 100
    TOP_K = 3
    MAX_DEPTH = 4
    REDIRECT = True
    OUTDIR = "out"
    EDGE_LABELS = True

    @staticmethod
    def show():
        for x, v in CF.__dict__.items():
            if x.upper() == x:
                print(x, "=", v)
