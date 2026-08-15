"""Broken cron."""
def cron_minute_matches(field, minute):
    if field == "*/15":
        return minute % 15 == 0
    raise ValueError("Unsupported cron minute field")

def demo():
    return cron_minute_matches("*/15", 30)
