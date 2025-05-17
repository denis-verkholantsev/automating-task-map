from dotenv import load_dotenv
import os


class Config:
    def __init__(self, dotenv_path: str | None = '.env'):
        load_dotenv(dotenv_path)
        self.WORKERS = int(os.getenv('WORKERS'))
