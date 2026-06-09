import importlib.util
import inspect
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Type

from pydantic import BaseModel


# Post extraction Functions (POST_FUNC)
def word_to_number(num_str: str) -> int:
    """Parse a number written in letters (up to miliardi)"""

    number_system = {
        "zero": 0,
        "uno": 1,
        "due": 2,
        "tre": 3,
        "quattro": 4,
        "cinque": 5,
        "sei": 6,
        "sette": 7,
        "otto": 8,
        "nove": 9,
        "dieci": 10,
        "undici": 11,
        "dodici": 12,
        "tredici": 13,
        "quattordici": 14,
        "quindici": 15,
        "sedici": 16,
        "diciassette": 17,
        "diciotto": 18,
        "diciannove": 19,
        "venti": 20,
        "trenta": 30,
        "quaranta": 40,
        "cinquanta": 50,
        "sessanta": 60,
        "settanta": 70,
        "ottanta": 80,
        "novanta": 90,
    }

    # Billions
    if "miliardo" in num_str or "miliardi" in num_str:
        parts = re.split("miliard[oi]", num_str, maxsplit=1)
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ""

        left_val = 1 if left in ("", "un", "uno") else word_to_number(left)
        right_val = word_to_number(right) if right else 0

        return left_val * 1_000_000_000 + right_val

    # Milions
    if "milione" in num_str or "milioni" in num_str:
        parts = re.split("milion[ei]", num_str, maxsplit=1)
        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ""

        left_val = 1 if left in ("", "un", "uno") else word_to_number(left)
        right_val = word_to_number(right) if right else 0

        return left_val * 1_000_000 + right_val

    # Thousands
    if "mila" in num_str or "mille" in num_str:
        if "mila" in num_str:
            parts = num_str.split("mila", 1)
        else:
            parts = num_str.split("mille", 1)

        left = parts[0].strip()
        right = parts[1].strip() if len(parts) > 1 else ""

        left_val = 1 if left in ("", "un", "uno") else word_to_number(left)
        right_val = word_to_number(right) if right else 0

        return left_val * 1000 + right_val

    # Hundreds
    if "cent" in num_str:
        left, right = num_str.split("cent", 1)
        left_val = 1 if left in ("", "un", "uno") else word_to_number(left)

        if len(right) > 1:
            right_val = (
                word_to_number(right)
                if right[:3] in ("uno", "ott")
                else word_to_number(right[1:])
            )
        else:
            right_val = 0

        return left_val * 100 + right_val

    # Tens and units
    else:
        match = re.match(
            r"^(.*vent|trent|quarant|cinquant|sessant|settant|ottant|novant)(.*)",
            num_str,
        )
        if match:
            tens = (
                match.group(1) + "i"
                if match.group(1) == "vent"
                else match.group(1) + "a"
            )
            units = (
                match.group(2)[1:]
                if match.group(2) and match.group(2)[0] in ["i", "a"]
                else match.group(2)
            )
            units = units if units else "zero"
        else:
            tens = "zero"
            units = num_str

        total = number_system.get(tens, 0) + number_system.get(units, 0)

        return total


def formatted_word_to_number(num_str: str) -> str:
    """Format integer with thousands separators (dot)."""
    # No formatting for termini di legge/come previsto nel legge
    pattern_to_spot = r"\b(?:come previsto|secondo|in base a|ai sensi di)\s+(?:dalla|dai|dal)?\s*(?:legge|termini di legge)\b"
    if re.search(pattern_to_spot, num_str, re.IGNORECASE):
        return num_str
    # return as it is if already digit
    if num_str.isdigit():
        return num_str

    # or try to find closest number
    n = word_to_number(num_str)
    return f"{n:,}".replace(",", ".")


def return_default_json(basemodel_json_schema_properties: dict) -> dict:
    """
    Returns default values of every key in the basemodel.
    Args:
        basemodel_json_schema_properties: .model_json_schema()['properties']
    """
    default_json = {}
    for k, v in basemodel_json_schema_properties.items():
        default_json[k] = v["default"]
    return default_json


def return_keys_description_schema(schema: BaseModel) -> str:
    output_structure = {}
    keys_description = "Descrizione delle chiavi:"

    for k, v in schema.model_json_schema()["properties"].items():
        output_structure[k] = ""
        keys_description += f"\n{k}: {v['description']} (type: {v['type']})"

    output = f"{json.dumps(output_structure, indent=4, ensure_ascii=False)}\n\n{keys_description}"

    return output


# Load all the schemas of info to be extracted from ./schemas
def load_classes_from_path(
    path: str,
    files_to_exclude: List[str] = [],  # "bilanci_e_utili.py", "compenso_degli_amministratori.py", "info_generali.py"
) -> Dict[str, Type]:
    """
    Load all .py files in `path` (non-recursive) and return classes defined in them.
    """
    classes = {}
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Path {path!r} not found or not a directory")

    for pyfile in p.glob("*.py"):
        # Optionally skip __init__.py or files you don't want
        if (pyfile.name == "__init__.py") or (pyfile.name in files_to_exclude):
            continue

        module_name = pyfile.stem  # filename without .py
        spec = importlib.util.spec_from_file_location(module_name, str(pyfile))
        if spec is None:
            continue
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        if loader is None:
            continue
        loader.exec_module(module)  # run the module # this is synchronous

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if getattr(obj, "__module__", None) == module.__name__:
                if not name.startswith("_"):
                    # disambiguate by module: use module.ClassName as key if needed
                    key = f"{module.__name__}.{name}"
                    classes[key] = obj
    return classes


# Group classes in same module in a list
def group_classes_by_module(classes: Dict[str, Type]) -> Dict[str, List[Type]]:
    """
    Given a dict mapping "module.ClassName" -> class object,
    group classes by module (the part before the first dot).
    Returns dict mapping "module" -> [class objects...].
    """
    grouped: DefaultDict[str, List[Type]] = defaultdict(list)
    for key, cls in classes.items():
        if "." in key:
            module_base = key.split(".", 1)[0]
        else:
            # fallback: derive from class __module__
            module_base = getattr(cls, "__module__", "<unknown>")
        grouped[module_base].append(cls)

    # Optionally: convert defaultdict -> dict and sort lists by class name
    result: Dict[str, List[Type]] = {}
    for mod, cls_list in grouped.items():
        cls_list.sort(key=lambda c: c.__name__)
        result[mod.upper()] = cls_list
    return result
