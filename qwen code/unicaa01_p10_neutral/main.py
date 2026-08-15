"""Config kata."""

def validate_config(cfg):
    return bool('port' in cfg and isinstance(cfg['port'], int) and 1 <= cfg['port'] <= 65535)

def demo():
    return False
