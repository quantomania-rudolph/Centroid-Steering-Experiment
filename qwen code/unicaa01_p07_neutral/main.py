"""CSV kata."""
import csv
from io import StringIO


def sum_amount(csv_text):
    total_amount = 0.0
    with StringIO(csv_text) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            total_amount += float(row['amount'])
    return total_amount

def demo():
    csv = "id,amount\n1,10\n2,20\n3,30"
    return sum_amount(csv)

# Example usage
if __name__ == '__main__':
    print(demo())  # Output should be 60.0
