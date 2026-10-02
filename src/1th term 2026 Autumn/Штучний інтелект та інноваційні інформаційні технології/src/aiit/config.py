"""Спільні гіперпараметри для обох лабораторних робіт.

Обидва фреймворки імпортують саме ці значення, тому будь-яка різниця
в результатах TensorFlow і PyTorch походить від фреймворку, а не від
розбіжності в налаштуваннях експерименту.
"""

from pathlib import Path

SEED = 42

# Поділ вибірки: 60 / 20 / 20
TEST_SIZE = 0.20
VAL_SIZE = 0.25  # 0.25 від залишку (80%) дає 20% від повного набору

# Навчання (завдання 8: 10-20 епох, фіксовані optimizer і learning rate)
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
L2_LAMBDA = 1e-4
DROPOUT_RATE = 0.2

# Архітектури (завдання 6)
HIDDEN_UNITS = (64, 32)

# Активації для порівняння (завдання 7)
ACTIVATIONS = ("relu", "gelu")

# Кількість квантильних груп цільової змінної для аналізу дисбалансу
TARGET_BINS = 5

# src/aiit/config.py -> src/aiit -> src -> тека дисципліни
ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = ROOT.parent.parent.parent  # корінь робочого простору

# Усі згенеровані артефакти (рисунки, метрики) лежать у .cache/ — спільному
# для робочого простору дереві проміжних даних, яке не версіонується. Дерево
# вихідних кодів залишається чистим. Назви семестру й дисципліни беруться
# з фактичного розташування теки, а не дублюються тут рядком.
ARTIFACTS = (
    REPO_ROOT / ".cache" / "reports-artifacts" / ROOT.relative_to(REPO_ROOT / "src")
)

EDA = ARTIFACTS / "eda"
LAB1 = ARTIFACTS / "lab1"
LAB2 = ARTIFACTS / "lab2"
