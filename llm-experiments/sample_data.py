import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAFETY_GOLD_PATH = ROOT / "data" / "dices" / "safety_gold" / "diverse_safety_adversarial_dialog_350.csv"
df = pd.read_csv(SAFETY_GOLD_PATH)

df[df.item_id == 193]
print(df[df.item_id == 193])
# df.drop_duplicates(subset="item_id")

# df.drop_duplicates(subset="rater_id")