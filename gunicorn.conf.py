import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers = 1
threads = 8
worker_class = "gthread"
timeout = 180
