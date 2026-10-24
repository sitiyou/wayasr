import json
import os
from pathlib import Path
import tempfile

from platformdirs import user_config_path, user_data_path


def config_home():
    path = user_config_path().expanduser()
    if not path.is_absolute():
        raise ValueError("XDG_CONFIG_HOME must be an absolute path")
    return path


def config_path():
    config_home()
    return user_config_path("wayasr", appauthor=False).expanduser() / "config.json"


def default_model_dir():
    return user_data_path("wayasr", appauthor=False).expanduser() / "models"


def load_config():
    path = config_path()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid configuration at {path}: {error.msg}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    if "url" in config and (not isinstance(config["url"], str) or not config["url"].strip()):
        raise ValueError(f"Expected a nonempty url string at {path}")
    for name in ("server", "client", "live"):
        if name in config and not isinstance(config[name], dict):
            raise ValueError(f"Expected a {name} object at {path}")
    return config


def write_text_atomic(path, content, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(file.name)
    try:
        with file:
            file.write(content)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_config(config):
    write_text_atomic(config_path(), json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def server_config(args, binary=None):
    server = vars(args).copy()
    server.pop("action", None)
    if binary:
        server["server_bin"] = str(Path(binary).absolute())
    for name in ("model_dir", "model", "mmproj"):
        value = server[name]
        if value is not None:
            server[name] = str(Path(value).expanduser().resolve())
    host = args.host
    if host in ("0.0.0.0", "*"):
        host = "127.0.0.1"
    elif host in ("::", "[::]"):
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return {"server": server, "url": f"http://{host}:{args.port}"}


def save_server_config(args, binary):
    config = load_config()
    config.update(server_config(args, binary))
    save_config(config)
