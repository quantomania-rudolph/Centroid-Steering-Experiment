def sum_amount(csv_text):
    lines = [ln for ln in csv_text.strip().splitlines() if ln.strip()]
    if not lines:
        return 0.0
    headers = [h.strip() for h in lines[0].split(",")]
    idx = headers.index("amount")
    total = 0.0
    for row in lines[1:]:
        parts = [p.strip() for p in row.split(",")]
        total += float(parts[idx])
    return total


def demo():
    csv = "id,amount\n1,10\n2,20\n3,30"
    return sum_amount(csv)
