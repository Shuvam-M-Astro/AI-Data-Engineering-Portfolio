import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-export tests so test discovery can find them here as well
from NLP.Sentiment_Analysis.RPA_Sentiment_Analysis.test_imdb_sentiment import *  # noqa: F401,F403
