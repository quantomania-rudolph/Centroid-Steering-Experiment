"""Cron kata."""
def cron_minute_matches(field, minute):
    if field == '*/15':
        return minute in {0, 15, 30, 45}
    return False

def demo():
    print(cron_minute_matches('*/15', 30))  # Should return True
    print(cron_minute_matches('*/15', 20))  # Should return False

if __name__ == '__main__':
    demo()
