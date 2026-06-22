"""Put the tools/ directory on sys.path so tests can import the modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
