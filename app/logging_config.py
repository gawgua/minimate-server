import logging


def configure_logging(level: str):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level=level)
    logging.getLogger("app").setLevel(level)
