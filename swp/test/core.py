from Levenshtein import editops


def calculate_errors(prediction: list, target: list) -> dict:
    # TODO Daniel docstring
    errors = {
        "inss": 0,
        "dels": 0,
        "subs": 0,
        "total": 0,
        "length": len(target),
        "indices": [
            i + 1 for i, (p, t) in enumerate(zip(prediction, target)) if p != t
        ],
    }

    # Tabulate errors by type
    ops = editops(prediction, target)
    for op, _, _ in ops:
        if op == "insert":
            errors["inss"] += 1
        elif op == "delete":
            errors["dels"] += 1
        elif op == "replace":
            errors["subs"] += 1
    errors["total"] = len(ops)

    return errors
