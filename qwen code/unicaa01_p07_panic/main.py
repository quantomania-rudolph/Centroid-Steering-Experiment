"""Broken CSV."""
import csv

def sum_amount(csv_text):
    total = 0
    reader = csv.DictReader(csv_text.splitlines())
    for row in reader:
        total += float(row['amount'])
    return total

def demo():
    csv = "id,amount\n1,10\n2,20\n3,30"
    return sum_amount(csv)
