from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ENGLISH_DIR = DATA_DIR / "english"
RUSSIAN_DIR = DATA_DIR / "russian"

MODEL_PATH = MODELS_DIR / "classifier.joblib"
CONFIG_PATH = MODELS_DIR / "training_config.json"

CHUNK_SIZE = 800
MIN_CHUNKS = 3
MAX_CHUNKS_PER_AUTHOR = 100

SPACY_MODELS = {"en": "en_core_web_sm", "ru": "ru_core_news_sm"}

ENGLISH_FUNCTION_WORDS: list[str] = [
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "that", "this", "these", "those",
    "it", "its", "he", "she", "they", "we", "you", "i", "me", "him",
    "her", "us", "them", "my", "his", "their", "our", "your", "who",
    "which", "what", "when", "if", "then", "than", "there", "here",
    "not", "no", "nor", "also", "only", "just", "even", "about", "after",
    "before", "between", "through", "into", "upon", "without", "over",
    "while", "although", "because", "however", "therefore", "yet", "both",
]

RUSSIAN_FUNCTION_WORDS: list[str] = [
    "и", "в", "не", "на", "что", "с", "а", "по", "это", "как", "к",
    "но", "из", "у", "за", "так", "же", "от", "для", "при", "об",
    "до", "или", "со", "через", "без", "да", "чем", "то", "если",
    "ни", "когда", "только", "где", "там", "можно", "нет", "уже",
    "ещё", "тоже", "более", "вот", "был", "была", "было", "были",
    "он", "она", "они", "мы", "я", "ты", "вы", "его", "её", "их",
    "этот", "эта", "эти", "того", "тем", "те", "свой", "весь",
    "всё", "потому", "хотя", "тогда", "здесь", "всегда",
    "очень", "теперь", "всего", "именно", "потом", "затем", "даже",
]
