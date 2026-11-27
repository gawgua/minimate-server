from app.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import faulthandler
    import logging

    faulthandler.enable()

    import uvicorn

    from app.config import Settings

    settings = Settings.load()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=logging.getLevelNamesMapping()[settings.log_level],
    )
