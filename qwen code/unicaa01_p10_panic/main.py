def validate_config(cfg):
    port = cfg.get("port")
    if port is not None and port > 65535:
        return False
    return True

def demo():
    return not validate_config({"port": 99999})
