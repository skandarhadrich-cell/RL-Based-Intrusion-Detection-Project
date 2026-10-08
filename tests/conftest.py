import sys
from pathlib import Path

# Make `import rl_ids` work without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
