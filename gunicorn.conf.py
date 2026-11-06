# Gunicorn reads this file automatically when it starts in this folder, so Render's start
# command (gunicorn Rev14:app) picks it up without any change in the dashboard.
#
# Before this file, one worker served one request at a time: a single slow page (waiting
# on a news site or the AI) made every other visitor wait in line. Threads let one
# process serve many readers at once while staying inside the 512 MB of the free plan.
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
worker_class = 'gthread'
workers = int(os.environ.get('WEB_CONCURRENCY', '1'))      # each process uses about 250 MB
threads = int(os.environ.get('GUNICORN_THREADS', '8'))
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '60'))
graceful_timeout = 30
keepalive = 5
# Recycle the process now and then so memory can't creep up over weeks.
max_requests = int(os.environ.get('GUNICORN_MAX_REQUESTS', '3000'))
max_requests_jitter = 300
accesslog = None
errorlog = '-'
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')
