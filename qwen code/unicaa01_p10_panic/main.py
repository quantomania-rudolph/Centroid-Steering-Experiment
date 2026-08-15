def validate_config(cfg):
    port = cfg.get("port")
    if port is None:
        return False
    return not (not isinstance(port, int) or port < 0 or port > 65535)

def demo():
    return not validate_config({"port": 99999})
