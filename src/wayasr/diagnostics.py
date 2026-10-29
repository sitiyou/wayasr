import logging
from logging.handlers import RotatingFileHandler

from platformdirs import user_state_path


class FileOnlyHandler(RotatingFileHandler):
    def handleError(self, record):
        pass


logger = logging.getLogger("wayasr.alignment")
logger.propagate = False
logger.addHandler(logging.NullHandler())


def log_alignment_failure(error):
    try:
        if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
            path = user_state_path("wayasr", appauthor=False) / "alignment.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = FileOnlyHandler(path, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        logger.error("Forced alignment failed: %s", error, exc_info=True)
    except OSError:
        pass
