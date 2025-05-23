from dotenv import load_dotenv
import os

MAX_SIZE_DEFAULT_KMEANS=5000000
WORKERS=1

class Config:
    def __init__(self, dotenv_path: str | None = '.env'):
        load_dotenv(dotenv_path)
        self.WORKERS = int(os.getenv('WORKERS')) or WORKERS
        self.MAX_SIZE_DEFAULT_KMEANS =  int(os.getenv('MAX_SIZE_DEFAULT_KMEANS')) or MAX_SIZE_DEFAULT_KMEANS
