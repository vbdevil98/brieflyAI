#Report, comment edit and delete

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import hashlib
import time
import logging
import urllib.parse
import secrets
import re
import threading
from collections import defaultdict, deque, OrderedDict
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Response, abort, make_response

# Third-party imports
import nltk
import requests
from flask import (Flask, render_template, url_for, redirect, request, jsonify, session, flash)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, case
from sqlalchemy.orm import joinedload
from jinja2 import DictLoader
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from newspaper import Article, Config
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import LangChainException
import pytz

# --- Load Environment Variables ---
load_dotenv()

# ==============================================================================
# --- 1. NLTK 'punkt' Tokenizer Setup ---
# ==============================================================================
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_nltk_data_path = os.path.join(project_root, 'nltk_data')
    if local_nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_data_path)
    nltk.data.find('tokenizers/punkt', paths=[local_nltk_data_path])
    print("NLTK 'punkt' tokenizer found.", file=sys.stderr)
except LookupError:
    print("WARNING: NLTK 'punkt' tokenizer not found. Attempting to download...", file=sys.stderr)
    try:
        nltk.download('punkt')
        print("NLTK 'punkt' downloaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"FATAL: Failed to download 'punkt'. Error: {e}", file=sys.stderr)
        sys.exit("Exiting: Missing critical NLTK data.")

# ==============================================================================
# --- 2. Flask Application Initialization & Configuration ---
# ==============================================================================
app = Flask(__name__)

template_storage = {}
app.jinja_loader = DictLoader(template_storage)

# SECURITY: Flask decides autoescaping from the template *filename* extension
# (select_jinja_autoescape -> endswith(".html", ".htm", ".xml", ...)). These templates are
# keyed as "ARTICLE_HTML_TEMPLATE" etc., which match none of those, so autoescaping would be
# OFF and every user-supplied value (comment text, article titles, display names) would render
# as live HTML -> stored XSS. Force it on for all templates.
app.jinja_env.autoescape = True

_FALLBACK_SECRET_SENTINEL = 'YOUR_FALLBACK_FLASK_SECRET_KEY_HERE_32_CHARS'
_secret_key = os.environ.get('FLASK_SECRET_KEY')
# APP_ENV=production is the signal that this is a real deployment.
IS_PRODUCTION = os.environ.get('APP_ENV', '').lower() in ('production', 'prod')

if not _secret_key or _secret_key == _FALLBACK_SECRET_SENTINEL:
    if IS_PRODUCTION:
        # A known/placeholder key means anyone can forge a session cookie and log in as
        # any user, including the admin. Refuse to boot rather than run insecurely.
        raise RuntimeError(
            "FLASK_SECRET_KEY is missing or still set to the placeholder value. "
            "Set a strong random value (e.g. `python -c \"import secrets; print(secrets.token_hex(32))\"`) "
            "before starting in production."
        )
    # Dev/local: a per-process random key. Sessions won't survive a restart, which is
    # the correct trade-off versus shipping a publicly-known key.
    _secret_key = secrets.token_hex(32)
    logging.warning("FLASK_SECRET_KEY not set - using an ephemeral random key. Sessions reset on restart.")

app.secret_key = _secret_key

# Session cookie hardening.
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,      # JS cannot read the session cookie (XSS containment)
    SESSION_COOKIE_SAMESITE='Lax',     # blocks the cookie on cross-site POSTs
    SESSION_COOKIE_SECURE=IS_PRODUCTION,  # HTTPS-only in production; off locally so dev still works
)

# The admin account is configurable rather than hardcoded.
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'vbdevil').strip().lower()

# --- Google AdSense -------------------------------------------------------------------
# Accepts "ca-pub-123..." or "pub-123...". Ads only load on reading pages (ADS_ENDPOINTS).
def _adsense_client(raw):
    match = re.fullmatch(r'(?:ca-)?(pub-\d{10,20})', (raw or '').strip())
    return 'ca-' + match.group(1) if match else ''


# The site's own AdSense account. Set ADSENSE_CLIENT_ID to override it, or to "off" to remove all ads.
ADSENSE_DEFAULT_CLIENT = 'pub-6975904325280886'
ADSENSE_CLIENT_ID = _adsense_client(os.environ.get('ADSENSE_CLIENT_ID') or ADSENSE_DEFAULT_CLIENT)
# Optional fixed ad units (numeric slot ids from AdSense > Ads > By ad unit). Auto ads
# work without them; these only reserve tidy, labelled spaces in the layout.
ADSENSE_SLOTS = {
    name: slot for name, slot in (
        (name, (os.environ.get(f'ADSENSE_SLOT_{name.upper()}') or '').strip()) for name in ('feed', 'article', 'list'))
    if re.fullmatch(r'\d{6,20}', slot)
}
ADSENSE_ON_LANDING = (os.environ.get('ADSENSE_ON_LANDING') or '').strip().lower() in ('1', 'true', 'yes', 'on')
# Pages with real reading content. AdSense policy keeps ads off sign in, account, error and
# thank you pages, so those are never listed here.
ADS_ENDPOINTS = {'index', 'article_detail', 'search_results'} | ({'landing'} if ADSENSE_ON_LANDING else set())
# Extra ads.txt lines for other ad networks, separated by ";" or new lines.
# Google Analytics 4 measurement id; set GA_MEASUREMENT_ID to "off" to remove analytics.
_ga = (os.environ.get('GA_MEASUREMENT_ID') or 'G-CV5LWJ7NQ7').strip()
GA_MEASUREMENT_ID = _ga if re.fullmatch(r'G-[A-Z0-9]{4,16}', _ga) else ''
ADS_TXT_EXTRA = [line.strip() for line in re.split(r'[;\n]', os.environ.get('ADS_TXT_EXTRA') or '') if line.strip()]
app.config['PER_PAGE'] = 9
app.config['CATEGORIES'] = ['All Articles', 'Popular Stories', "Yesterday's Headlines", 'Community Hub']

app.config['NEWS_API_QUERY'] = 'India OR "Indian politics" OR "Indian economy" OR "Bollywood"'
app.config['NEWS_API_DOMAINS'] = 'timesofindia.indiatimes.com,thehindu.com,ndtv.com,indianexpress.com,hindustantimes.com'
app.config['NEWS_API_DAYS_AGO'] = 7 # Fetch news from the last 7 days
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'publishedAt' #relevance, popularity, publishedAt
app.config['CACHE_EXPIRY_SECONDS'] = 1800 
app.permanent_session_lifetime = timedelta(days=30)
# Reject oversized uploads/bodies before they are buffered into memory.
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH_BYTES', 2 * 1024 * 1024))

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# Data Persistence
using_postgres_flag = False

# --- MODIFIED Data Persistence ---
app.logger.info(f"--- Database Configuration ---")
app.logger.info(f"Attempting to read DATABASE_URL environment variable...")
SUPABASE_PROJECT_REF = (os.environ.get('SUPABASE_PROJECT_REF') or 'szxesgdqoenhcnawwscm').strip()
SUPABASE_REGION = (os.environ.get('SUPABASE_REGION') or 'ap-southeast-1').strip()


def _supabase_url_from_password(password):
    """
    Build the Supabase connection string from just the database password, so setting up
    needs one value instead of a hand-edited URL. Supabase runs its connection pooler on
    two clusters per region; the one that accepts this project's login is used.
    Session pooler (port 5432) works from Render, which cannot reach Supabase over IPv6.
    """
    import psycopg2
    user = (os.environ.get('SUPABASE_DB_USER') or f'postgres.{SUPABASE_PROJECT_REF}').strip()
    port = int(os.environ.get('SUPABASE_POOLER_PORT') or 5432)
    name = (os.environ.get('SUPABASE_DB_NAME') or 'postgres').strip()
    pinned = (os.environ.get('SUPABASE_POOLER_HOST') or '').strip()
    hosts = [pinned] if pinned else [f'aws-0-{SUPABASE_REGION}.pooler.supabase.com',
                                     f'aws-1-{SUPABASE_REGION}.pooler.supabase.com']
    sslmode = (os.environ.get('SUPABASE_SSLMODE') or 'require').strip()
    errors = []
    for host in hosts:
        url = (f"postgresql://{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(password, safe='')}"
               f"@{host}:{port}/{name}?sslmode={sslmode}")
        try:
            psycopg2.connect(url, connect_timeout=10).close()
            app.logger.info(f"Supabase: connected through {host} as {user}.")
            return url
        except psycopg2.OperationalError as e:
            # The message never contains the password; keep only its first line.
            errors.append(f"{host}: {str(e).strip().splitlines()[0] if str(e).strip() else 'connection failed'}")
    raise RuntimeError("Could not sign in to the Supabase database with SUPABASE_DB_PASSWORD. "
                       "Check the password was copied in full. Tried: " + " | ".join(errors))


# PRIMARY_DATABASE_URL (or just SUPABASE_DB_PASSWORD) is the permanent database. DATABASE_URL
# is Render's free database, which Render deletes 30 days after it is created. When both are
# set the app runs on the primary one and, on its first start, copies every table across
# from DATABASE_URL (see import_legacy_database), so the move never loses an account.
primary_database_url = (os.environ.get('PRIMARY_DATABASE_URL') or '').strip()
if not primary_database_url and (os.environ.get('SUPABASE_DB_PASSWORD') or '').strip():
    primary_database_url = _supabase_url_from_password(os.environ['SUPABASE_DB_PASSWORD'].strip())
legacy_database_url = (os.environ.get('DATABASE_URL') or '').strip()
database_url = primary_database_url or legacy_database_url

if database_url: 
    app.logger.info(f"DATABASE_URL found. Raw value (prefix): '{database_url[:30]}...'") # Log prefix for security
else:
    app.logger.info("DATABASE_URL environment variable NOT FOUND or is empty.")

configured_db_uri = None
# using_postgres_flag is already initialized to False above

if database_url and (database_url.startswith("postgres://") or database_url.startswith("postgresql://")):
    app.logger.info(f"DATABASE_URL indicates a PostgreSQL connection.")
    if database_url.startswith("postgres://"):
        configured_db_uri = database_url.replace("postgres://", "postgresql://", 1)
        app.logger.info(f"Converted DATABASE_URL from 'postgres://' to 'postgresql://'.")
    else: # Already starts with "postgresql://"
        configured_db_uri = database_url
        app.logger.info(f"DATABASE_URL already uses 'postgresql://' scheme.")

    # Name the driver explicitly. SQLAlchemy 2.1 made psycopg (v3) the default for a bare
    # 'postgresql://' URL, but this app installs psycopg2-binary, so an unpinned fresh build
    # crashed at startup with "No module named 'psycopg'".
    configured_db_uri = configured_db_uri.replace("postgresql://", "postgresql+psycopg2://", 1)
    # Supabase only accepts encrypted connections from outside its network.
    if 'supabase.' in configured_db_uri and 'sslmode=' not in configured_db_uri:
        configured_db_uri += ('&' if '?' in configured_db_uri else '?') + 'sslmode=require'

    app.config['SQLALCHEMY_DATABASE_URI'] = configured_db_uri
    uri_to_log = configured_db_uri
    try:
        parsed_uri = urllib.parse.urlparse(configured_db_uri)
        if parsed_uri.username or parsed_uri.password: # Mask credentials
            # Ensure port is handled correctly if present
            host_port = parsed_uri.hostname
            if parsed_uri.port:
                host_port += f":{parsed_uri.port}"
            uri_to_log = f"{parsed_uri.scheme}://********:********@{host_port}{parsed_uri.path}"
    except Exception:
        pass # Keep original if parsing fails
    app.logger.info(f"SQLAlchemy URI configured for PostgreSQL: {uri_to_log}")
    using_postgres_flag = True # Set flag to True
else:
    # This block is entered if database_url (from env) is None, empty, or not a valid Postgres URL format
    if database_url: # It existed but wasn't a postgres URL
        app.logger.warning(f"DATABASE_URL found ('{database_url[:30]}...') but it does not seem to be a PostgreSQL URL (expected 'postgres://' or 'postgresql://').")
    
    app.logger.info("Falling back to local SQLite database.")
    
    db_file_name = 'app_data.db'
    # Ensure project_root_for_db is correctly determined
    project_root_for_db = ""
    try:
        project_root_for_db = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(project_root_for_db, db_file_name)
    except NameError: # __file__ might not be defined in some execution contexts (e.g. interactive)
        app.logger.warning("__file__ not defined when setting SQLite path, using relative path for DB.")
        db_path = db_file_name # Fallback to relative path if absolute path fails
        
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.logger.info(f"SQLAlchemy URI configured for SQLite: {app.config['SQLALCHEMY_DATABASE_URI']}")
    # using_postgres_flag remains False (as initialized)

def _postgres_uri(url):
    """Normalise a postgres:// URL for the psycopg2 driver, or None if it is not Postgres."""
    if not url or not url.startswith(("postgres://", "postgresql://")):
        return None
    url = url.replace("postgres://", "postgresql://", 1).replace("postgresql://", "postgresql+psycopg2://", 1)
    if 'supabase.' in url and 'sslmode=' not in url:
        url += ('&' if '?' in url else '?') + 'sslmode=require'
    return url


# The database to copy from once, when the app has just been pointed at a new primary.
LEGACY_DATABASE_URI = None
if primary_database_url and legacy_database_url:
    _legacy = _postgres_uri(legacy_database_url)
    if _legacy and _legacy != _postgres_uri(primary_database_url):
        LEGACY_DATABASE_URI = _legacy

_active_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
if 'supabase.' in _active_uri:
    DATABASE_PROVIDER = 'Supabase'
elif 'render.com' in _active_uri or '@dpg-' in _active_uri:
    DATABASE_PROVIDER = 'Render (free databases are deleted 30 days after creation)'
elif _active_uri.startswith('postgresql'):
    DATABASE_PROVIDER = 'PostgreSQL'
else:
    DATABASE_PROVIDER = 'SQLite (local file)'
try:
    DATABASE_LIMIT_MB = int(os.environ.get('DATABASE_LIMIT_MB') or 0) or (
        500 if DATABASE_PROVIDER == 'Supabase' else 1024 if DATABASE_PROVIDER.startswith('Render') else 0)
except ValueError:
    DATABASE_LIMIT_MB = 0

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    # Managed Postgres drops idle connections; without these, the first query after an
    # idle period fails with "server closed the connection unexpectedly".
    'pool_pre_ping': True,
    'pool_recycle': 280,
}
if using_postgres_flag:
    # Stay well inside the free connection limits of hosted Postgres poolers.
    app.config['SQLALCHEMY_ENGINE_OPTIONS'].update({
        'pool_size': 5, 'max_overflow': 5, 'connect_args': {'connect_timeout': 10},
    })
db = SQLAlchemy(app) 
app.logger.info(f"SQLAlchemy instance created.")
app.logger.info(f"--- End of Database Configuration ---")

# ==============================================================================
# --- 2b. Security layer (CSRF, rate limiting, headers, input validation) ---
#
# Implemented with the standard library only, so deployment needs no new packages.
# ==============================================================================

# --- CSRF -------------------------------------------------------------------
# Without this, any other website can silently make a logged-in visitor's browser
# POST here (post articles, delete their comments, change bookmarks), because the
# session cookie rides along automatically.
CSRF_FIELD_NAME = 'csrf_token'
CSRF_HEADER_NAME = 'X-CSRFToken'
CSRF_EXEMPT_ENDPOINTS = set()  # add endpoint names here if a webhook ever needs to bypass


def generate_csrf_token():
    """Return this session's CSRF token, creating one on first use."""
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


def csrf_exempt(view):
    """Decorator to opt a view out of CSRF validation."""
    CSRF_EXEMPT_ENDPOINTS.add(view.__name__)
    return view


def _request_csrf_token():
    token = request.form.get(CSRF_FIELD_NAME)
    if token:
        return token
    token = request.headers.get(CSRF_HEADER_NAME)
    if token:
        return token
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if isinstance(payload, dict):
            return payload.get(CSRF_FIELD_NAME)
    return None


@app.before_request
def csrf_protect():
    if request.method not in ('POST', 'PUT', 'PATCH', 'DELETE'):
        return None
    if request.endpoint in CSRF_EXEMPT_ENDPOINTS:
        return None

    sent = _request_csrf_token()
    expected = session.get('_csrf_token')
    # compare_digest avoids leaking token contents through timing differences.
    if not expected or not sent or not secrets.compare_digest(str(sent), str(expected)):
        app.logger.warning(
            "CSRF validation failed for %s %s (endpoint=%s)",
            request.method, request.path, request.endpoint
        )
        wants_json = request.is_json or request.headers.get('Accept', '').startswith('application/json')
        if wants_json:
            return jsonify({
                "success": False,
                "error": "Your session expired or the request could not be verified. Please refresh the page and try again."
            }), 400
        flash("Your session expired or the request could not be verified. Please try again.", "danger")
        return redirect(safe_redirect_target(request.referrer))
    return None


# --- Rate limiting ----------------------------------------------------------
# In-memory sliding window. Note: this is per-process, so with multiple gunicorn
# workers each worker enforces its own budget. It stops casual brute force and
# spam; a shared store (Redis) would be needed for strict global limits.
_rate_buckets = defaultdict(deque)
_rate_lock = threading.Lock()
_RATE_SWEEP_EVERY = 500
_rate_calls_since_sweep = 0


def _client_identity():
    """Best-effort caller identity: logged-in user, else client IP."""
    if session.get('user_id'):
        return f"user:{session['user_id']}"
    forwarded = request.headers.get('X-Forwarded-For', '')
    ip = forwarded.split(',')[0].strip() if forwarded else (request.remote_addr or 'unknown')
    return f"ip:{ip}"


def _sweep_rate_buckets(now, max_window):
    """Drop buckets that have gone quiet so memory doesn't grow without bound."""
    stale = [k for k, dq in _rate_buckets.items() if not dq or (now - dq[-1]) > max_window]
    for k in stale:
        _rate_buckets.pop(k, None)


SAFE_METHODS = frozenset({'GET', 'HEAD', 'OPTIONS'})


def rate_limit(limit, per_seconds, scope=None, message=None, methods=None):
    """
    Allow at most `limit` requests per `per_seconds` per caller.

    By default only state-changing methods are counted. Several of these views serve
    both GET and POST (login, register), and counting page views would lock a
    legitimate user out simply for reloading the form.
    """
    def decorator(view):
        bucket_scope = scope or view.__name__
        counted = frozenset(m.upper() for m in methods) if methods else None

        @wraps(view)
        def wrapper(*args, **kwargs):
            global _rate_calls_since_sweep
            if counted is None:
                if request.method in SAFE_METHODS:
                    return view(*args, **kwargs)
            elif request.method not in counted:
                return view(*args, **kwargs)
            key = f"{bucket_scope}:{_client_identity()}"
            now = time.time()
            with _rate_lock:
                _rate_calls_since_sweep += 1
                if _rate_calls_since_sweep >= _RATE_SWEEP_EVERY:
                    _rate_calls_since_sweep = 0
                    _sweep_rate_buckets(now, max(per_seconds * 4, 3600))
                dq = _rate_buckets[key]
                while dq and (now - dq[0]) > per_seconds:
                    dq.popleft()
                if len(dq) >= limit:
                    retry_after = int(per_seconds - (now - dq[0])) + 1
                    app.logger.warning("Rate limit hit on %s by %s", bucket_scope, key)
                    text = message or "Too many requests. Please slow down and try again shortly."
                    wants_json = request.is_json or request.headers.get('Accept', '').startswith('application/json')
                    if wants_json:
                        resp = jsonify({"success": False, "error": text})
                        resp.status_code = 429
                        resp.headers['Retry-After'] = str(retry_after)
                        return resp
                    flash(text, "warning")
                    resp = make_response(redirect(safe_redirect_target(request.referrer)))
                    resp.headers['Retry-After'] = str(retry_after)
                    return resp
                dq.append(now)
            return view(*args, **kwargs)
        return wrapper
    return decorator


# --- Safe redirects ---------------------------------------------------------
def is_safe_redirect_url(target):
    """True only for same-host relative/absolute URLs, so ?next= can't send users off-site."""
    if not target:
        return False
    if target.startswith('//') or target.startswith('\\\\'):
        return False
    parsed = urllib.parse.urlparse(urllib.parse.urljoin(request.host_url, target))
    host_parsed = urllib.parse.urlparse(request.host_url)
    return parsed.scheme in ('http', 'https') and parsed.netloc == host_parsed.netloc


def safe_redirect_target(target, fallback_endpoint='index'):
    return target if is_safe_redirect_url(target) else url_for(fallback_endpoint)


# --- Input validation -------------------------------------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]{2,}$")
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,80}$")

LIMITS = {
    'comment': 5000,
    'article_title': 250,
    'article_description': 1000,
    'article_content': 50000,
    'source_name': 100,
    'image_url': 500,
    'name': 120,
    'email': 120,
}


def clean_text(value, max_length, allow_empty=False):
    """Trim, collapse NULs, and enforce a maximum length. Returns (value, error)."""
    value = (value or '').replace('\x00', '').strip()
    if not value and not allow_empty:
        return None, "This field is required."
    if len(value) > max_length:
        return None, f"Too long - please keep this under {max_length} characters."
    return value, None


# --- Security headers -------------------------------------------------------
@app.after_request
def set_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    response.headers.setdefault('Permissions-Policy', 'geolocation=(), microphone=(), camera=(), interest-cohort=()')
    if IS_PRODUCTION:
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response


@app.context_processor
def inject_csrf_token():
    return {'csrf_token': generate_csrf_token}


# --- Request correlation + teardown ------------------------------------------
@app.before_request
def attach_request_id():
    """Tag each request so its log lines can be traced together."""
    from flask import g
    g.request_id = request.headers.get('X-Request-ID') or secrets.token_hex(6)


@app.after_request
def echo_request_id(response):
    from flask import g
    rid = getattr(g, 'request_id', None)
    if rid:
        response.headers.setdefault('X-Request-ID', rid)
    return response


@app.teardown_appcontext
def release_db_session(exception=None):
    """Always return the connection to the pool, even when a view raised."""
    try:
        if exception:
            db.session.rollback()
        db.session.remove()
    except Exception:
        pass


def _wants_json():
    return request.is_json or request.headers.get('Accept', '').startswith('application/json')


@app.errorhandler(400)
def bad_request(e):
    if _wants_json():
        return jsonify({"success": False, "error": "The request could not be understood."}), 400
    return render_template("404_TEMPLATE"), 400


@app.errorhandler(403)
def forbidden(e):
    if _wants_json():
        return jsonify({"success": False, "error": "You don't have permission to do that."}), 403
    flash("You don't have permission to do that.", "danger")
    return redirect(url_for('index'))


@app.errorhandler(413)
def payload_too_large(e):
    msg = "That submission is too large. Please shorten it and try again."
    if _wants_json():
        return jsonify({"success": False, "error": msg}), 413
    flash(msg, "warning")
    return redirect(url_for('index'))


@app.errorhandler(429)
def too_many_requests(e):
    msg = "Too many requests. Please slow down and try again shortly."
    if _wants_json():
        return jsonify({"success": False, "error": msg}), 429
    flash(msg, "warning")
    return redirect(url_for('index'))


# ==============================================================================
# --- 3. API Client Initialization ---
# ==============================================================================
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
if not newsapi: app.logger.error("NEWSAPI_KEY missing. News fetching will fail.")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=GROQ_API_KEY, temperature=0.1)
        app.logger.info("Groq client initialized.")
    except Exception as e:
        app.logger.error(f"Failed to initialize Groq client: {e}")
else:
    app.logger.warning("GROQ_API_KEY missing. AI analysis disabled.")

SCRAPER_API_KEY = os.environ.get('SCRAPER_API_KEY')
if not SCRAPER_API_KEY:
    app.logger.warning("SCRAPER_API_KEY missing. Article content fetching may fail.")

# ==============================================================================
# --- 4. Database Models ---
# ==============================================================================

# In Rev14.py, add this new model

class ReportedArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Foreign key to the community article that was reported
    article_id = db.Column(db.Integer, db.ForeignKey('community_article.id', ondelete="CASCADE"), nullable=False)
    # Foreign key to the user who filed the report
    reporter_user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    # The reason for the report (optional but recommended)
    reason = db.Column(db.String(250), nullable=True)
    # The status of the report
    status = db.Column(db.String(20), nullable=False, default='pending') # e.g., 'pending', 'resolved'
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # Ensures a user can only report a specific article once
    __table_args__ = (db.UniqueConstraint('article_id', 'reporter_user_id', name='_article_reporter_uc'),)

    # --- THIS IS THE CORRECTED PART ---
    # The backref now tells SQLAlchemy to delete any associated reports when an article is deleted.
    article = db.relationship('CommunityArticle', backref=db.backref('reports', lazy='dynamic', cascade="all, delete-orphan"))
    
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    articles = db.relationship('CommunityArticle', backref='author', lazy='dynamic', cascade="all, delete-orphan")
    comments = db.relationship('Comment', backref=db.backref('author', lazy='joined'), lazy='dynamic', cascade="all, delete-orphan")
    comment_votes = db.relationship('CommentVote', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    bookmarks = db.relationship('BookmarkedArticle', backref='user', lazy='dynamic', cascade="all, delete-orphan")

class CommunityArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_hash_id = db.Column(db.String(32), unique=True, nullable=False, index=True)
    title = db.Column(db.String(250), nullable=False)
    description = db.Column(db.Text, nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    source_name = db.Column(db.String(100), nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    published_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    groq_summary = db.Column(db.Text, nullable=True)
    groq_takeaways = db.Column(db.Text, nullable=True) # Stored as JSON string
    comments = db.relationship('Comment', backref=db.backref('community_article', lazy='joined'), lazy='dynamic', foreign_keys='Comment.community_article_id', cascade="all, delete-orphan")

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    community_article_id = db.Column(db.Integer, db.ForeignKey('community_article.id'), nullable=True, index=True)
    api_article_hash_id = db.Column(db.String(32), nullable=True, index=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('comment.id'), nullable=True, index=True)
    replies = db.relationship('Comment', backref=db.backref('parent', remote_side=[id]), lazy='selectin', cascade="all, delete-orphan")
    votes = db.relationship('CommentVote', backref='comment', lazy='dynamic', cascade="all, delete-orphan")


class CommentVote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    comment_id = db.Column(db.Integer, db.ForeignKey('comment.id', ondelete="CASCADE"), nullable=False, index=True)
    # MODIFIED: Changed to store the specific emoji character for the reaction.
    vote_emoji = db.Column(db.String(10), nullable=False)
    # The unique constraint ensures a user can only have one reaction per comment.
    __table_args__ = (db.UniqueConstraint('user_id', 'comment_id', name='_user_comment_uc'),)


class Subscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    subscribed_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

class BookmarkedArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False, index=True)
    article_hash_id = db.Column(db.String(32), nullable=False, index=True)
    is_community_article = db.Column(db.Boolean, default=False, nullable=False)
    title_cache = db.Column(db.String(250), nullable=True)
    source_name_cache = db.Column(db.String(100), nullable=True)
    image_url_cache = db.Column(db.String(500), nullable=True)
    description_cache = db.Column(db.Text, nullable=True)
    published_at_cache = db.Column(db.DateTime, nullable=True) # Store as datetime for API articles
    bookmarked_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    __table_args__ = (db.UniqueConstraint('user_id', 'article_hash_id', name='_user_article_bookmark_uc'),)

class ArticleAnalysis(db.Model):
    """
    Durable store for AI analysis of API articles.

    These previously lived only in the in-memory MASTER_ARTICLE_STORE. Render's free
    web services spin down after ~15 minutes idle, so that cache was wiped constantly
    and every article got re-summarised on the next visit -- which is what burns the
    Groq request quota. Persisting here means each article is summarised once, ever.

    Note this stores only the generated summary/takeaways; nothing about HOW they are
    generated changes.
    """
    id = db.Column(db.Integer, primary_key=True)
    article_hash_id = db.Column(db.String(32), unique=True, nullable=False, index=True)
    groq_summary = db.Column(db.Text, nullable=True)
    groq_takeaways = db.Column(db.Text, nullable=True)  # JSON string
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)


class ArticleStat(db.Model):
    """
    View counts, keyed by article hash so it works for both community and API
    articles. Deliberately a NEW table: adding a column to an existing model would
    need a migration, and db.create_all() only creates missing tables.
    """
    id = db.Column(db.Integer, primary_key=True)
    article_hash_id = db.Column(db.String(32), unique=True, nullable=False, index=True)
    view_count = db.Column(db.Integer, nullable=False, default=0)
    last_viewed_at = db.Column(db.DateTime, nullable=True, index=True)


class Testimonial(db.Model):
    """
    Reader quotes for the landing page. A quote is either sent by a signed in reader who
    ticked the consent box, or entered by the admin with the reader's permission. Nothing
    is shown publicly until the admin approves it. New table, so db.create_all() adds it.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=True, index=True)
    name = db.Column(db.String(120), nullable=False)
    attribution = db.Column(db.String(80), nullable=True)   # e.g. "Student, Pune"
    quote = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='pending', index=True)  # pending / approved / rejected
    source = db.Column(db.String(20), nullable=False, default='reader')  # reader / admin
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    # Deleting an account removes that reader's testimonials too.
    user = db.relationship('User', backref=db.backref('testimonials', lazy='dynamic', cascade="all, delete-orphan"))


class AppMeta(db.Model):
    """Small key/value store for housekeeping state: the one-time data import and clean ups."""
    __tablename__ = 'app_meta'
    key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))


def _meta_get(key):
    row = db.session.get(AppMeta, key)
    if not row or not row.value:
        return None
    try:
        return json.loads(row.value)
    except ValueError:
        return None


def _meta_set(key, value):
    row = db.session.get(AppMeta, key) or AppMeta(key=key)
    row.value = json.dumps(value, default=str)
    db.session.add(row)
    db.session.commit()


LEGACY_IMPORT_KEY = 'legacy_import'
LEGACY_IMPORT_LOCK = 724001  # any constant; serialises the import across workers


def import_legacy_database():
    """
    Copy every table from the old database (DATABASE_URL) into the new primary database,
    exactly once. Runs in a single transaction: either everything arrives or nothing does.
    If the copy fails the app refuses to start, so the previous deploy (still on the old
    database) keeps serving instead of a new, empty site.
    """
    from sqlalchemy import create_engine, inspect as sa_inspect, insert, select, text
    if db.engine.dialect.name != 'postgresql':
        app.logger.warning("Data import skipped: the primary database is not PostgreSQL.")
        return
    meta = AppMeta.__table__
    started = time.time()
    with db.engine.connect() as conn:
        # Transaction scoped lock: works through Supabase's pooler in both session and
        # transaction mode, and is released automatically on commit or rollback.
        conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {'k': LEGACY_IMPORT_LOCK})
        if conn.execute(select(meta.c.value).where(meta.c.key == LEGACY_IMPORT_KEY)).first():
            conn.rollback()
            app.logger.info("Data import: already done on an earlier start, nothing to copy.")
            return
        if conn.execute(select(func.count()).select_from(User.__table__)).scalar():
            conn.execute(insert(meta).values(
                key=LEGACY_IMPORT_KEY, updated_at=datetime.now(timezone.utc),
                value=json.dumps({'status': 'skipped', 'reason': 'the new database already had accounts',
                                  'at': datetime.now(timezone.utc).isoformat()})))
            conn.commit()
            app.logger.warning("Data import skipped: the new database already has accounts.")
            return

        legacy = create_engine(LEGACY_DATABASE_URI, pool_pre_ping=True, connect_args={'connect_timeout': 15})
        try:
            inspector = sa_inspect(legacy)
            legacy_tables = set(inspector.get_table_names())
            counts = {}
            with legacy.connect() as source:
                for table in db.metadata.sorted_tables:  # parents before children
                    if table.name == meta.name or table.name not in legacy_tables:
                        continue
                    present = {c['name'] for c in inspector.get_columns(table.name)}
                    columns = [c for c in table.columns if c.name in present]
                    query = select(*columns)
                    if 'id' in present:
                        query = query.order_by(table.c.id)  # replies after the comments they answer
                    result = source.execution_options(stream_results=True, yield_per=500).execute(query).mappings()
                    copied = 0
                    for batch in result.partitions(500):
                        conn.execute(insert(table), [dict(row) for row in batch])
                        copied += len(batch)
                    counts[table.name] = copied
                    if 'id' in table.c and isinstance(table.c.id.type, db.Integer):
                        # Continue numbering after the copied rows, or the next insert collides.
                        conn.execute(text(
                            f"SELECT setval(pg_get_serial_sequence('\"{table.name}\"', 'id'), "
                            f"COALESCE((SELECT MAX(id) FROM \"{table.name}\"), 0) + 1, false)"))
            conn.execute(insert(meta).values(
                key=LEGACY_IMPORT_KEY, updated_at=datetime.now(timezone.utc),
                value=json.dumps({'status': 'done', 'rows': counts, 'seconds': round(time.time() - started, 1),
                                  'at': datetime.now(timezone.utc).isoformat()})))
            conn.commit()
            app.logger.info(f"Data import finished in {time.time() - started:.1f}s: {counts}")
        except Exception:
            conn.rollback()
            app.logger.error("Data import FAILED. Nothing was written to the new database; "
                             "the app will not start on an empty database.", exc_info=True)
            raise
        finally:
            legacy.dispose()


# --- Automatic clean up ---------------------------------------------------------------
# Summaries and view counts are the only tables that grow with every news cycle. Old ones
# are dropped unless a reader bookmarked or discussed the story. A summary that is dropped
# is simply generated again if anyone opens that story later.
ANALYSIS_RETENTION_DAYS = int(os.environ.get('ANALYSIS_RETENTION_DAYS') or os.environ.get('RETAIN_ANALYSIS_DAYS') or 120)
STATS_RETENTION_DAYS = int(os.environ.get('STATS_RETENTION_DAYS') or os.environ.get('RETAIN_STATS_DAYS') or 90)
CLEANUP_KEY = 'last_cleanup'
CLEANUP_EVERY_SECONDS = 24 * 3600
_cleanup_state = {'checked_at': 0.0, 'running': False}
_cleanup_lock = threading.Lock()


def prune_old_data():
    """Delete stale summaries and view counts. Returns how many rows went."""
    from sqlalchemy import select
    now = datetime.now(timezone.utc)
    bookmarked = select(BookmarkedArticle.article_hash_id)
    discussed = select(Comment.api_article_hash_id).where(Comment.api_article_hash_id.isnot(None))
    analyses = (ArticleAnalysis.query
                .filter(ArticleAnalysis.created_at < now - timedelta(days=ANALYSIS_RETENTION_DAYS),
                        ArticleAnalysis.article_hash_id.notin_(bookmarked),
                        ArticleAnalysis.article_hash_id.notin_(discussed))
                .delete(synchronize_session=False))
    stats = (ArticleStat.query
             .filter(db.or_(ArticleStat.last_viewed_at.is_(None),
                            ArticleStat.last_viewed_at < now - timedelta(days=STATS_RETENTION_DAYS)))
             .delete(synchronize_session=False))
    db.session.commit()
    result = {'summaries': analyses, 'view_counts': stats, 'at': now.isoformat()}
    _meta_set(CLEANUP_KEY, result)
    app.logger.info(f"Clean up removed {analyses} old summaries and {stats} old view counts.")
    return result


def _cleanup_worker():
    try:
        with app.app_context():
            last = _meta_get(CLEANUP_KEY) or {}
            try:
                last_at = datetime.fromisoformat(last.get('at', ''))
            except ValueError:
                last_at = None
            if last_at and (datetime.now(timezone.utc) - last_at).total_seconds() < CLEANUP_EVERY_SECONDS:
                return
            prune_old_data()
    except Exception as e:
        app.logger.error(f"Automatic clean up failed: {e}", exc_info=True)
    finally:
        _cleanup_state['running'] = False
        try:
            with app.app_context():
                db.session.remove()
        except Exception:
            pass


@app.before_request
def schedule_cleanup():
    """At most once an hour, check in the background whether a daily clean up is due."""
    if time.time() - _cleanup_state['checked_at'] < 3600 or app.config.get('TESTING'):
        return
    with _cleanup_lock:
        if _cleanup_state['running'] or time.time() - _cleanup_state['checked_at'] < 3600:
            return
        _cleanup_state['checked_at'] = time.time()
        _cleanup_state['running'] = True
    threading.Thread(target=_cleanup_worker, name='cleanup', daemon=True).start()


def database_storage_report():
    """Size of the database and each table, for the admin storage page."""
    from sqlalchemy import select, text
    report = {'provider': DATABASE_PROVIDER, 'limit_mb': DATABASE_LIMIT_MB, 'size_bytes': None, 'tables': [],
              'data_copy': _meta_get(LEGACY_IMPORT_KEY), 'cleanup': _meta_get(CLEANUP_KEY),
              'retention': {'summaries': ANALYSIS_RETENTION_DAYS, 'view_counts': STATS_RETENTION_DAYS}}
    is_pg = db.engine.dialect.name == 'postgresql'
    if is_pg:
        report['size_bytes'] = db.session.execute(text("SELECT pg_database_size(current_database())")).scalar()
    else:
        try:
            report['size_bytes'] = os.path.getsize(db.engine.url.database)
        except (OSError, TypeError):
            pass
    for table in db.metadata.sorted_tables:
        rows = db.session.execute(select(func.count()).select_from(table)).scalar()
        size = None
        if is_pg:
            size = db.session.execute(text("SELECT pg_total_relation_size(CAST(:t AS regclass))"),
                                      {'t': f'"{table.name}"'}).scalar()
        report['tables'].append({'name': table.name, 'rows': rows, 'bytes': size})
    report['tables'].sort(key=lambda t: (t['bytes'] or 0, t['rows']), reverse=True)
    if report['size_bytes'] and DATABASE_LIMIT_MB:
        report['percent'] = min(100.0, round(report['size_bytes'] / (DATABASE_LIMIT_MB * 1024 * 1024) * 100, 1))
    return report


def lock_down_public_tables():
    """
    Supabase publishes the public schema through its REST API with a key anyone can read
    from a web page. Row level security with no policies closes that door completely. The
    app's own connection owns these tables, so it is not affected.
    """
    from sqlalchemy import text
    try:
        with db.engine.begin() as conn:
            for table in db.metadata.sorted_tables:
                conn.execute(text(f'ALTER TABLE "{table.name}" ENABLE ROW LEVEL SECURITY'))
        app.logger.info("Row level security is on for every table (no public API access).")
    except Exception as e:
        app.logger.error(f"Could not enable row level security: {e}", exc_info=True)


def init_db():
    # To access the global 'using_postgres_flag' set earlier
    global using_postgres_flag
    with app.app_context():
        app.logger.info("--- Database Initialization (init_db) ---")
        try:
            engine = db.get_engine()
            engine_url_str = str(engine.url) 
            dialect_name = engine.dialect.name
            app.logger.info(f"SQLAlchemy engine is configured with URL (actual): {engine_url_str}")
            app.logger.info(f"SQLAlchemy dialect in use: {dialect_name}")

            # Check the flag set during initial config
            if using_postgres_flag: 
                if dialect_name == "postgresql":
                    app.logger.info("CONFIRMED: SQLAlchemy is using PostgreSQL dialect as intended.")
                else:
                    app.logger.warning(f"WARNING: Intended to use PostgreSQL (using_postgres_flag=True), but SQLAlchemy dialect is '{dialect_name}'. Check DATABASE_URL and configuration.")
            else: # Intended to use SQLite (fallback)
                if dialect_name == "sqlite":
                    app.logger.info("CONFIRMED: SQLAlchemy is using SQLite dialect as intended (fallback or no valid DATABASE_URL).")
                else:
                    app.logger.warning(f"WARNING: Intended to use SQLite (using_postgres_flag=False), but SQLAlchemy dialect is '{dialect_name}'.")
        except Exception as e:
            app.logger.error(f"Error during SQLAlchemy engine/dialect logging: {e}", exc_info=True)

        app.logger.info("Attempting to create database tables (db.create_all()). This is non-destructive to existing tables.")
        try:
            db.create_all()
            app.logger.info("db.create_all() executed successfully. Tables should be ready or already exist.")
        except Exception as e:
            app.logger.error(f"Error during db.create_all(): {e}", exc_info=True)
        if 'supabase.' in str(db.engine.url):
            lock_down_public_tables()
        if LEGACY_DATABASE_URI:
            import_legacy_database()  # raises on failure so we never serve an empty database
        app.logger.info("--- End of Database Initialization (init_db) ---")

# ==============================================================================
# --- 5. Helper Functions ---
# ==============================================================================
class BoundedCache:
    """
    Thread-safe cache with LRU eviction and optional TTL.

    Replaces the plain dicts previously used here, which were never evicted from and
    so grew for the lifetime of the process. Reads refresh recency, so entries that
    are still actively viewed (including cached AI analysis) survive eviction.
    """

    def __init__(self, max_entries=5000, ttl_seconds=None):
        self._data = OrderedDict()
        self._lock = threading.RLock()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _expired(self, stamped_at):
        return self.ttl_seconds is not None and (time.time() - stamped_at) > self.ttl_seconds

    def get(self, key, default=None):
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return default
            value, stamped_at = entry
            if self._expired(stamped_at):
                self._data.pop(key, None)
                self.misses += 1
                return default
            self._data.move_to_end(key)  # mark as recently used
            self.hits += 1
            return value

    def set(self, key, value):
        with self._lock:
            self._data[key] = (value, time.time())
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # drop least-recently-used
                self.evictions += 1

    # dict-style access, so existing call sites keep working unchanged.
    def __getitem__(self, key):
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        self.set(key, value)

    def __contains__(self, key):
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    def __len__(self):
        with self._lock:
            return len(self._data)

    def stats(self):
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._data),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": round(self.hits / total, 3) if total else None,
            }


# MASTER_ARTICLE_STORE holds fetched API articles (including any cached AI analysis).
# It is capped rather than unbounded; the app already handles a missing entry as
# "article not found", which is the same behaviour as after a restart.
MASTER_ARTICLE_STORE = BoundedCache(max_entries=int(os.environ.get('ARTICLE_STORE_MAX', '4000')))
API_CACHE = BoundedCache(max_entries=1000, ttl_seconds=None)
INDIAN_TIMEZONE = pytz.timezone('Asia/Kolkata')

def generate_article_id(url_or_title): return hashlib.md5(url_or_title.encode('utf-8')).hexdigest()

def jinja_truncate_filter(s, length=120, killwords=False, end='...'):
    if not s: return ''
    if len(s) <= length: return s
    if killwords: return s[:length - len(end)] + end
    words = s.split()
    result_words = []
    current_length = 0
    for word in words:
        if current_length + len(word) + (1 if result_words else 0) > length - len(end): break
        result_words.append(word)
        current_length += len(word) + (1 if len(result_words) > 1 else 0)
    if not result_words: return s[:length - len(end)] + end
    return ' '.join(result_words) + end
app.jinja_env.filters['truncate'] = jinja_truncate_filter

def to_ist_filter(utc_dt):
    if not utc_dt: return "N/A"
    if isinstance(utc_dt, str):
        try:
            if utc_dt.endswith('Z'):
                 utc_dt = datetime.fromisoformat(utc_dt[:-1] + '+00:00')
            else:
                 utc_dt = datetime.fromisoformat(utc_dt)
        except ValueError: return "Invalid date string"
    
    if not isinstance(utc_dt, datetime): return "Invalid date object"

    if utc_dt.tzinfo is None: utc_dt = pytz.utc.localize(utc_dt)
    else: utc_dt = utc_dt.astimezone(pytz.utc)
    ist_dt = utc_dt.astimezone(INDIAN_TIMEZONE)
    return ist_dt.strftime('%b %d, %Y at %I:%M %p %Z')
app.jinja_env.filters['to_ist'] = to_ist_filter


def simple_cache(expiry_seconds_default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            expiry = expiry_seconds_default or app.config['CACHE_EXPIRY_SECONDS']
            key_parts = [func.__name__] + list(map(str, args)) + sorted(kwargs.items())
            cache_key = hashlib.md5(str(key_parts).encode('utf-8')).hexdigest()
            cached_entry = API_CACHE.get(cache_key)
            if cached_entry and (time.time() - cached_entry[1] < expiry):
                app.logger.debug(f"Cache HIT for {func.__name__}")
                return cached_entry[0]
            app.logger.debug(f"Cache MISS for {func.__name__}. Calling function.")
            result = func(*args, **kwargs)
            # Per-call expiry is kept in the tuple; the cache itself only bounds size.
            API_CACHE.set(cache_key, (result, time.time()))
            return result
        return wrapper
    return decorator

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # Check if the request is an API/JSON request
            if request.headers.get('Accept') == 'application/json':
                return jsonify({"success": False, "error": "Authentication required. Please log in."}), 401
            # Otherwise, it's a normal page load, so redirect to login page
            else:
                flash("You must be logged in to access this page.", "warning")
                return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@simple_cache(expiry_seconds_default=3600 * 12)
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client: return {"error": "AI analysis service not available."}
    if not article_text or not article_text.strip(): return {"error": "No text provided for AI analysis."}
    app.logger.info(f"Requesting Groq analysis for: {article_title[:50]}...")
    system_prompt = ("You are an expert news analyst. Analyze the following article. "
        "1. Provide a tight, neutral summary of 2-3 sentences (under 60 words total). Lead with the single most important fact. No preamble, no repetition of the headline. "
        "2. List exactly 3-4 key takeaways as short bullet points. Each takeaway must be one complete sentence under 20 words. "
        "Format your entire response as a single JSON object with keys 'summary' (string) and 'takeaways' (a list of strings).")
    human_prompt = f"Article Title: {article_title}\n\nArticle Text:\n{article_text[:20000]}"
    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content) # Groq should return JSON string in ai_response.content
        
        # Ensure keys exist, default to null or empty if not explicitly present
        groq_summary = analysis.get("summary") 
        groq_takeaways = analysis.get("takeaways")

        # Even if keys exist, they might be null from Groq if it couldn't generate them
        # No specific error from Groq if it returns nulls in valid JSON format
        return {"groq_summary": groq_summary, "groq_takeaways": groq_takeaways, "error": None}

    except json.JSONDecodeError as e:
        app.logger.error(f"Groq analysis - JSONDecodeError for '{article_title[:50]}': {e}. Response content: {ai_response.content if 'ai_response' in locals() else 'N/A'}")
        return {"error": f"AI analysis failed to decode JSON: {str(e)}"}
    except LangChainException as e: # More specific Langchain errors
        app.logger.error(f"Groq analysis - LangChainException for '{article_title[:50]}': {e}")
        return {"error": f"AI analysis failed (LangChain): {str(e)}"}
    except Exception as e:
        app.logger.error(f"Unexpected error during Groq analysis for '{article_title[:50]}': {e}", exc_info=True)
        return {"error": "An unexpected error occurred during AI analysis."}

# In Rev14.py, add this new function

@simple_cache(expiry_seconds_default=14400) # Cache the synthesis for 4 hours
def get_daily_synthesis():
    """
    Fetches top stories and uses an LLM to create a high-level synthesis
    of the day's main themes and extract keywords.
    """
    app.logger.info("Generating AI Daily Synthesis...")
    if not groq_client:
        return {"synthesis_text": None, "keywords": []}

    # Get the articles to be synthesized
    articles_to_synthesize = fetch_popular_news()
    if not articles_to_synthesize:
        return {"synthesis_text": None, "keywords": []}

    # Prepare the content for the AI
    # We'll combine titles and descriptions for a rich context
    content_for_ai = ""
    for art in articles_to_synthesize[:15]: # Use up to 15 articles for the context
        content_for_ai += f"Title: {art.get('title', '')}\\nDescription: {art.get('description', '')}\\n\\n"

    system_prompt = (
        "You are a top-tier news editor for an Indian audience. Your task is to provide a 'big picture' summary of the day's news based on a collection of article titles and descriptions. "
        "Analyze the provided text and generate a JSON object with two keys: 'synthesis_text' and 'keywords'. "
        "1. For 'synthesis_text': Write a single, insightful paragraph of exactly 2 sentences (under 45 words total) that connects the day's most important themes. Do not list the news; connect the ideas. Be punchy and specific. "
        "2. For 'keywords': Extract the 4-5 most significant and distinct keywords or key phrases from the articles. These should represent the main topics of the day."
    )
    
    human_prompt = f"Here is the collection of today's news articles:\n\n{content_for_ai}"

    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content)
        
        # Validate the response from the AI
        synthesis = analysis.get("synthesis_text")
        keywords = analysis.get("keywords")
        if not isinstance(synthesis, str) or not isinstance(keywords, list):
            raise ValueError("AI response did not have the correct format.")

        app.logger.info(f"Successfully generated AI Daily Synthesis. Keywords: {keywords}")
        return {"synthesis_text": synthesis, "keywords": keywords}

    except Exception as e:
        app.logger.error(f"Error during AI Daily Synthesis generation: {e}", exc_info=True)
        return {"synthesis_text": "The AI summary for the day could not be generated at this time.", "keywords": []}

# ==============================================================================
# --- NEWS FETCHING ---
# ==============================================================================
@simple_cache()
def fetch_news_from_api(target_date_str=None):
    # RSS first: it is real-time, unlimited and carries no commercial-use restriction,
    # whereas NewsAPI's free tier is ~24h delayed, capped at 100 requests/day and is
    # licensed for development only. NewsAPI remains the fallback.
    if os.environ.get('USE_RSS', 'true').lower() in ('1', 'true', 'yes'):
        try:
            rss_articles = fetch_news_from_rss()
            if rss_articles:
                if target_date_str:
                    # Keep the same date-filtering contract the callers expect.
                    rss_articles = [a for a in rss_articles
                                    if (a.get('publishedAt') or '')[:10] == target_date_str]
                if rss_articles:
                    return rss_articles
            app.logger.warning("RSS returned nothing usable; falling back to NewsAPI.")
        except Exception as e:
            app.logger.error(f"RSS ingestion failed, falling back to NewsAPI: {e}", exc_info=True)

    if not newsapi:
        app.logger.error("NewsAPI client not initialized. Cannot fetch news.")
        return []

    api_call_from_date_str = None
    api_call_to_date_str = None
    is_specific_date_fetch = False

    # INDIAN_TIMEZONE should be globally defined in your script, e.g.,
    # INDIAN_TIMEZONE = pytz.timezone('Asia/Kolkata')

    if target_date_str:
        try:
            # Primary: Interpret target_date_str as user's local day in INDIAN_TIMEZONE
            local_day_start_naive = datetime.strptime(target_date_str, '%Y-%m-%d')
            local_day_start_aware_ist = INDIAN_TIMEZONE.localize(local_day_start_naive) # Assign IST timezone
            # Define end of the local day in IST
            local_day_end_aware_ist = local_day_start_aware_ist.replace(hour=23, minute=59, second=59, microsecond=999999)

            # Convert these IST times to UTC for the NewsAPI query
            api_call_from_utc_dt = local_day_start_aware_ist.astimezone(timezone.utc)
            api_call_to_utc_dt = local_day_end_aware_ist.astimezone(timezone.utc)

            api_call_from_date_str = api_call_from_utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
            api_call_to_date_str = api_call_to_utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            app.logger.info(f"Date filter active for '{target_date_str}' (interpreted as IST).")
            app.logger.info(f"Querying NewsAPI with UTC range: FROM {api_call_from_date_str} TO {api_call_to_date_str}")
            is_specific_date_fetch = True
        except Exception as e:
            app.logger.error(f"Error processing target_date_str '{target_date_str}' with IST conversion: {e}. Reverting to simpler UTC day or default fetch.", exc_info=True)
            # Fallback 1: Try interpreting target_date_str as a simple UTC date (original behavior for specific date)
            try:
                utc_target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
                api_call_from_date_str = utc_target_dt.strftime('%Y-%m-%dT00:00:00')
                api_call_to_date_str = utc_target_dt.strftime('%Y-%m-%dT23:59:59')
                app.logger.info(f"Fallback: Fetching news specifically for UTC date: {target_date_str} (from {api_call_from_date_str} to {api_call_to_date_str})")
                is_specific_date_fetch = True
            except ValueError: # If target_date_str is malformed even for simple parsing
                 app.logger.warning(f"Invalid target_date_str '{target_date_str}' for both IST and UTC interpretation. Clearing date filter.")
                 target_date_str = None # This will ensure it uses the default N-day range logic in the next block
                 is_specific_date_fetch = False # Ensure default logic runs if date string is unusable

    if not is_specific_date_fetch: 
        # Default fetch logic (no specific date selected, or date was invalid)
        # This fetches news from the last N days up to the end of the current UTC day.
        from_date_utc_default = datetime.now(timezone.utc) - timedelta(days=app.config['NEWS_API_DAYS_AGO'])
        api_call_from_date_str = from_date_utc_default.strftime('%Y-%m-%dT%H:%M:%S')
        
        current_day_utc_end_default = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0)
        api_call_to_date_str = current_day_utc_end_default.strftime('%Y-%m-%dT%H:%M:%S')
        app.logger.info(f"Fetching news with default date range (last {app.config['NEWS_API_DAYS_AGO']} days up to current UTC day end): from {api_call_from_date_str} to {api_call_to_date_str}")

    # --- API Call Attempts (The rest of the function remains the same as my previous detailed response) ---
    all_raw_articles = []

    # Attempt 1: Top Headlines (only if not fetching for a specific historical date, or for default range)
    # Note: is_specific_date_fetch is true if a date was successfully parsed (either IST-based or fallback UTC-based)
    if not is_specific_date_fetch: # Only run top_headlines if no specific date was successfully set for the fetch
        try:
            app.logger.info("Attempt 1: Fetching top headlines from country: 'in' (default range).")
            top_headlines_response = newsapi.get_top_headlines(
                country='in',
                language='en',
                page_size=app.config['NEWS_API_PAGE_SIZE']
            )
            status = top_headlines_response.get('status')
            total_results = top_headlines_response.get('totalResults', 0)
            app.logger.info(f"Top-Headlines API Response -> Status: {status}, TotalResults: {total_results}")
            if status == 'ok' and total_results > 0:
                all_raw_articles.extend(top_headlines_response['articles'])
            elif status == 'error':
                app.logger.error(f"NewsAPI Error (Top-Headlines): Code: {top_headlines_response.get('code')}, Message: {top_headlines_response.get('message')}. Full response: {top_headlines_response}")
            elif total_results == 0:
                app.logger.info(f"NewsAPI (Top-Headlines) returned 0 results. Response: {top_headlines_response}")
        except NewsAPIException as e:
            app.logger.error(f"NewsAPIException (Top-Headlines): {e.get_message() if hasattr(e, 'get_message') else str(e)}", exc_info=False)
        except Exception as e:
            app.logger.error(f"Generic Exception (Top-Headlines): {e}", exc_info=True)

    # Attempt 2: Everything query (This will always run, using the determined date range)
    try:
        current_query = app.config['NEWS_API_QUERY']
        current_sort_by = app.config['NEWS_API_SORT_BY']
        app.logger.info(f"Attempt 2: Fetching 'everything' with query: \"{current_query}\" for period: {api_call_from_date_str} to {api_call_to_date_str}, sort_by: {current_sort_by}")
        everything_response = newsapi.get_everything(
            q=current_query,
            from_param=api_call_from_date_str,
            to=api_call_to_date_str,
            language='en',
            sort_by=current_sort_by,
            page_size=app.config['NEWS_API_PAGE_SIZE']
        )
        status = everything_response.get('status')
        total_results = everything_response.get('totalResults', 0)
        app.logger.info(f"Everything API Response -> Status: {status}, TotalResults: {total_results}")
        if status == 'ok' and total_results > 0:
            all_raw_articles.extend(everything_response['articles'])
        elif status == 'error':
            app.logger.error(f"NewsAPI Error (Everything Query): Code: {everything_response.get('code')}, Message: {everything_response.get('message')}. Parameters used: q='{current_query}', from='{api_call_from_date_str}', to='{api_call_to_date_str}', sort_by='{current_sort_by}'. Full response: {everything_response}")
        elif total_results == 0:
            app.logger.info(f"NewsAPI (Everything Query) returned 0 results for q='{current_query}' from '{api_call_from_date_str}' to '{api_call_to_date_str}', sort_by='{current_sort_by}'. Response: {everything_response}")
    except NewsAPIException as e:
        app.logger.error(f"NewsAPIException (Everything Query): {e.get_message() if hasattr(e, 'get_message') else str(e)}. Parameters used: q='{current_query}', from='{api_call_from_date_str}', to='{api_call_to_date_str}', sort_by='{current_sort_by}'.", exc_info=False)
    except Exception as e:
        app.logger.error(f"Generic Exception (Everything Query): {e}", exc_info=True)

    # Attempt 3: Fallback with domains
    # Condition: Run if no articles yet, OR if it was a specific date fetch (to maximize chances for that day)
    # The `is_specific_date_fetch` flag is true if the target_date_str was successfully parsed into a date range (either IST-based or UTC-based fallback).
    if not all_raw_articles or is_specific_date_fetch:
        log_prefix_attempt3 = "Fallback/Augment"
        if not all_raw_articles and not is_specific_date_fetch:
             app.logger.warning("No articles from primary calls. Trying Fallback with domains for default range.")
        elif not all_raw_articles and is_specific_date_fetch:
            app.logger.warning(f"No articles from query for specific date '{target_date_str}'. Trying with domains.")
        elif all_raw_articles and is_specific_date_fetch:
            app.logger.info(f"Augmenting results for specific date '{target_date_str}' with domain-specific search.")


        try:
            domains_to_check = app.config['NEWS_API_DOMAINS']
            current_sort_by_fallback = app.config['NEWS_API_SORT_BY']
            app.logger.info(f"Attempt 3 ({log_prefix_attempt3}): Fetching from domains: {domains_to_check} for period: {api_call_from_date_str} to {api_call_to_date_str}, sort_by: {current_sort_by_fallback}")
            fallback_response = newsapi.get_everything(
                domains=domains_to_check,
                from_param=api_call_from_date_str,
                to=api_call_to_date_str,
                language='en',
                sort_by=current_sort_by_fallback,
                page_size=app.config['NEWS_API_PAGE_SIZE']
            )
            status = fallback_response.get('status')
            total_results = fallback_response.get('totalResults', 0)
            app.logger.info(f"{log_prefix_attempt3} API Response -> Status: {status}, TotalResults: {total_results}")
            if status == 'ok' and total_results > 0:
                all_raw_articles.extend(fallback_response['articles'])
            elif status == 'error':
                app.logger.error(f"NewsAPI Error ({log_prefix_attempt3}): Code: {fallback_response.get('code')}, Message: {fallback_response.get('message')}. Parameters used: domains='{domains_to_check}', from='{api_call_from_date_str}', to='{api_call_to_date_str}', sort_by='{current_sort_by_fallback}'. Full response: {fallback_response}")
            elif total_results == 0:
                app.logger.info(f"NewsAPI ({log_prefix_attempt3}) returned 0 results for domains='{domains_to_check}' from '{api_call_from_date_str}' to '{api_call_to_date_str}', sort_by='{current_sort_by_fallback}'. Response: {fallback_response}")
        except NewsAPIException as e:
            app.logger.error(f"NewsAPIException ({log_prefix_attempt3}): {e.get_message() if hasattr(e, 'get_message') else str(e)}. Parameters used: domains='{domains_to_check}', from='{api_call_from_date_str}', to='{api_call_to_date_str}', sort_by='{current_sort_by_fallback}'.", exc_info=False)
        except Exception as e:
            app.logger.error(f"Generic Exception ({log_prefix_attempt3}): {e}", exc_info=True)

    processed_articles, unique_urls = [], set()
    app.logger.info(f"Total raw articles fetched before deduplication: {len(all_raw_articles)}")
    for art_data in all_raw_articles:
        url = art_data.get('url')
        if not url or url in unique_urls: continue
        title = art_data.get('title')
        description = art_data.get('description')

        if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip() or not description.strip():
            continue
        unique_urls.add(url)
        article_id = generate_article_id(url)
        source_name = art_data['source'].get('name', 'Unknown Source')
        published_at_dt = None
        if art_data.get('publishedAt'):
            try:
                published_at_dt = datetime.fromisoformat(art_data.get('publishedAt').replace('Z', '+00:00'))
            except ValueError:
                app.logger.warning(f"Could not parse publishedAt date for article: {title} - Date: {art_data.get('publishedAt')}")
                published_at_dt = datetime.now(timezone.utc) # Fallback if parsing fails
        else:
            # If publishedAt is missing, which can happen, use current time as a fallback.
            # Or decide if such articles should be skipped. For now, using current time.
            app.logger.warning(f"Missing 'publishedAt' for article: {title}. Using current UTC time.")
            published_at_dt = datetime.now(timezone.utc)


        standardized_article = {
            'id': article_id, 'title': title, 'description': description,
            'url': url, 'urlToImage': art_data.get('urlToImage') or None,
            'publishedAt': published_at_dt.isoformat(), # Store as ISO string
            'source': {'name': source_name}, 'is_community_article': False,
            'groq_summary': None, 'groq_takeaways': None
        }
        MASTER_ARTICLE_STORE[article_id] = standardized_article
        processed_articles.append(standardized_article)
    
    processed_articles.sort(key=lambda x: x.get('publishedAt', datetime.min.replace(tzinfo=timezone.utc).isoformat()), reverse=True)
    app.logger.info(f"Total unique articles processed and returned by fetch_news_from_api: {len(processed_articles)} for period from {api_call_from_date_str} to {api_call_to_date_str}.")
    return processed_articles

@simple_cache(expiry_seconds_default=21600) # Cache popular news for 6 hours
def fetch_popular_news():
    """
    Fetches articles from the last few days and sorts them by popularity
    to find the most impactful stories.
    """
    app.logger.info("Fetching POPULAR news from API.")
    if not newsapi:
        return []

    # Query a window of the last 5 days to get a good measure of popularity
    to_date = datetime.now(timezone.utc) - timedelta(days=1)
    from_date = to_date - timedelta(days=5)

    try:
        response = newsapi.get_everything(
            q=app.config['NEWS_API_QUERY'],
            language='en',
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            sort_by='popularity', # This is the key for this section
            page_size=30
        )
        
        if response.get('status') == 'ok':
            raw_articles = response.get('articles', [])
            processed_articles, unique_urls = [], set()
            for art_data in raw_articles:
                url = art_data.get('url')
                if not url or url in unique_urls: continue
                title = art_data.get('title')
                description = art_data.get('description')

                if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip() or not description.strip():
                    continue
                
                unique_urls.add(url)
                article_id = generate_article_id(url)
                source_name = art_data['source'].get('name', 'Unknown Source')
                
                published_at_dt = None
                if art_data.get('publishedAt'):
                    try:
                        published_at_dt = datetime.fromisoformat(art_data.get('publishedAt').replace('Z', '+00:00'))
                    except ValueError:
                        published_at_dt = datetime.now(timezone.utc)
                else:
                    published_at_dt = datetime.now(timezone.utc)

                standardized_article = {
                    'id': article_id, 'title': title, 'description': description, 'url': url,
                    'urlToImage': art_data.get('urlToImage') or None,
                    'publishedAt': published_at_dt.isoformat(), 'source': {'name': source_name}, 
                    'is_community_article': False, 'groq_summary': None, 'groq_takeaways': None
                }
                MASTER_ARTICLE_STORE[article_id] = standardized_article
                processed_articles.append(standardized_article)
            
            app.logger.info(f"Returning {len(processed_articles)} unique POPULAR articles.")
            return processed_articles
        return []
    except Exception as e:
        app.logger.error(f"Exception in fetch_popular_news: {e}", exc_info=True)
        return []

@simple_cache(expiry_seconds_default=14400) # Cache yesterday's news for 4 hours
def fetch_yesterdays_latest_news():
    """
    Fetches all articles specifically from yesterday, calculated correctly
    using the Indian timezone, and sorts them by time.
    """
    app.logger.info("Fetching LATEST news from YESTERDAY from API.")
    if not newsapi:
        return []
    
    # 1. Get the current time in the Indian Timezone
    now_in_ist = datetime.now(INDIAN_TIMEZONE)
    
    # 2. Calculate the start and end of "yesterday" in IST
    yesterday_in_ist = now_in_ist - timedelta(days=1)
    start_of_yesterday_ist = yesterday_in_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_yesterday_ist = yesterday_in_ist.replace(hour=23, minute=59, second=59, microsecond=999999)

    # 3. Convert the IST start/end times to UTC for the API call
    start_utc = start_of_yesterday_ist.astimezone(pytz.utc)
    end_utc = end_of_yesterday_ist.astimezone(pytz.utc)

    # 4. Format the UTC dates into the string format the API requires (THE FIX IS HERE)
    from_param_str = start_utc.strftime('%Y-%m-%dT%H:%M:%S')
    to_param_str = end_utc.strftime('%Y-%m-%dT%H:%M:%S')

    app.logger.info(f"Querying for yesterday's news in UTC range: {from_param_str} to {to_param_str}")

    try:
        response = newsapi.get_everything(
            q=app.config['NEWS_API_QUERY'],
            language='en',
            from_param=from_param_str,
            to=to_param_str,
            sort_by='publishedAt',
            page_size=100
        )
        
        if response.get('status') == 'ok':
            raw_articles = response.get('articles', [])
            processed_articles, unique_urls = [], set()
            for art_data in raw_articles:
                url = art_data.get('url')
                if not url or url in unique_urls: continue
                title = art_data.get('title')
                description = art_data.get('description')
                if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip() or not description.strip():
                    continue
                unique_urls.add(url)
                article_id = generate_article_id(url)
                source_name = art_data['source'].get('name', 'Unknown Source')
                published_at_dt = None
                if art_data.get('publishedAt'):
                    try:
                        published_at_dt = datetime.fromisoformat(art_data.get('publishedAt').replace('Z', '+00:00'))
                    except ValueError:
                        published_at_dt = datetime.now(timezone.utc)
                else:
                    published_at_dt = datetime.now(timezone.utc)
                standardized_article = {
                    'id': article_id, 'title': title, 'description': description, 'url': url,
                    'urlToImage': art_data.get('urlToImage') or None,
                    'publishedAt': published_at_dt.isoformat(), 'source': {'name': source_name}, 
                    'is_community_article': False, 'groq_summary': None, 'groq_takeaways': None
                }
                MASTER_ARTICLE_STORE[article_id] = standardized_article
                processed_articles.append(standardized_article)
            app.logger.info(f"Returning {len(processed_articles)} unique LATEST articles from YESTERDAY.")
            return processed_articles
        return []
    except Exception as e:
        app.logger.error(f"Exception in fetch_yesterdays_latest_news: {e}", exc_info=True)
        return []
        
@simple_cache(expiry_seconds_default=3600 * 6)
def fetch_and_parse_article_content(article_hash_id, url):
    app.logger.info(f"Fetching content for API article ID: {article_hash_id}, URL: {url}")
    if not SCRAPER_API_KEY:
        return {"full_text": None, "groq_analysis": None, "error": "Content fetching service unavailable."}
    
    params = {'api_key': SCRAPER_API_KEY, 'url': url}
    try:
        response = requests.get('http://api.scraperapi.com', params=params,
                                timeout=int(os.environ.get('SCRAPER_TIMEOUT_SECONDS', '25')))
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        config = Config()
        config.fetch_images = False # To speed up, images are not needed for text analysis
        config.memoize_articles = False # Avoid disk caching by newspaper itself
        article_scraper = Article(url, config=config)
        article_scraper.download(input_html=response.text)
        article_scraper.parse()

        if not article_scraper.text:
            app.logger.warning(f"Could not extract text from article URL: {url}")
            return {"full_text": None, "groq_analysis": None, "error": "Could not extract text from the article."}
        
        article_title_for_groq = article_scraper.title or MASTER_ARTICLE_STORE.get(article_hash_id, {}).get('title', 'Unknown Title')
        
        groq_analysis_result = None # Initialize

        # Check if Groq analysis already exists in MASTER_ARTICLE_STORE for this API article
        # This is a redundancy check; the route get_article_content_json already does this.
        # However, keeping it ensures consistency if this function were called from elsewhere.
        if article_hash_id in MASTER_ARTICLE_STORE and \
           MASTER_ARTICLE_STORE[article_hash_id].get('groq_summary') is not None and \
           MASTER_ARTICLE_STORE[article_hash_id].get('groq_takeaways') is not None:
            app.logger.info(f"Re-confirming pre-cached Groq analysis from MASTER_ARTICLE_STORE for {article_hash_id} within fetch_and_parse.")
            groq_analysis_result = {
                "groq_summary": MASTER_ARTICLE_STORE[article_hash_id]['groq_summary'],
                "groq_takeaways": MASTER_ARTICLE_STORE[article_hash_id]['groq_takeaways'],
                "error": None 
            }
        else:
            # Second chance before spending quota: the durable store survives restarts,
            # unlike the in-memory one above.
            persisted = load_persisted_analysis(article_hash_id)
            if persisted:
                app.logger.info(f"Reusing persisted Groq analysis for {article_hash_id} (no API call).")
                groq_analysis_result = persisted
                if article_hash_id in MASTER_ARTICLE_STORE:
                    MASTER_ARTICLE_STORE[article_hash_id]['groq_summary'] = persisted.get("groq_summary")
                    MASTER_ARTICLE_STORE[article_hash_id]['groq_takeaways'] = persisted.get("groq_takeaways")
            else:
                # If not cached anywhere, generate it
                groq_analysis_result = get_article_analysis_with_groq(article_scraper.text, article_title_for_groq)
                # Persist so this article is never summarised twice, even across restarts.
                if groq_analysis_result and not groq_analysis_result.get("error"):
                    save_persisted_analysis(article_hash_id, groq_analysis_result)
            # And cache it in MASTER_ARTICLE_STORE if successfully generated
            if article_hash_id in MASTER_ARTICLE_STORE and groq_analysis_result and not groq_analysis_result.get("error"):
                MASTER_ARTICLE_STORE[article_hash_id]['groq_summary'] = groq_analysis_result.get("groq_summary")
                MASTER_ARTICLE_STORE[article_hash_id]['groq_takeaways'] = groq_analysis_result.get("groq_takeaways")
                app.logger.info(f"Groq analysis generated and cached in MASTER_ARTICLE_STORE for API article ID: {article_hash_id}")
            elif groq_analysis_result and groq_analysis_result.get("error"):
                 app.logger.warning(f"Groq analysis for {article_hash_id} resulted in error: {groq_analysis_result.get('error')}")


        return {
            "full_text": article_scraper.text,
            "groq_analysis": groq_analysis_result, 
            "error": None # Overall error for this function; specific errors are in groq_analysis_result if they occurred there
        }
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Failed to fetch article content via proxy for {url}: {e}")
        return {"full_text": None, "groq_analysis": None, "error": f"Failed to fetch article content: {str(e)}"}
    except Exception as e: # Catches ArticleException from newspaper, and any other general errors during parsing/processing
        app.logger.error(f"Failed to parse or process article content for {url}: {e}", exc_info=True)
        return {"full_text": None, "groq_analysis": None, "error": f"Failed to parse or process article content: {str(e)}"}
    
# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
@app.context_processor
def inject_global_vars():
    return {'categories': app.config['CATEGORIES'],
            'current_year': datetime.utcnow().year,
            'session': session,
            'request': request,
            'groq_client': groq_client is not None}

MAX_PAGE = 10000  # guards against ?page=999999999 style requests

def get_paginated_articles(articles, page, per_page):
    # Clamp first: a zero/negative page produces a negative slice start, which in Python
    # silently wraps to the END of the list instead of erroring.
    try:
        page = int(page)
    except (TypeError, ValueError):
        page = 1
    page = max(1, min(page, MAX_PAGE))
    total = len(articles)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = articles[start:end]
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0
    return paginated_items, total_pages

def get_sort_key(article):
    date_val = None
    if isinstance(article, dict): date_val = article.get('publishedAt')
    elif hasattr(article, 'published_at'): date_val = article.published_at
    if not date_val: return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(date_val, str):
        try:
            if date_val.endswith('Z'): date_val_dt = datetime.fromisoformat(date_val[:-1] + '+00:00')
            elif '+' in date_val or '-' in date_val[10:]: date_val_dt = datetime.fromisoformat(date_val)
            else: date_val_dt = datetime.fromisoformat(date_val).replace(tzinfo=timezone.utc)
            return date_val_dt
        except ValueError:
            app.logger.warning(f"Could not parse date string: {date_val}")
            return datetime.min.replace(tzinfo=timezone.utc)
    elif isinstance(date_val, datetime): return date_val if date_val.tzinfo else pytz.utc.localize(date_val)
    return datetime.min.replace(tzinfo=timezone.utc)

# In Rev14.py, find your existing index function and REPLACE IT with this entire block.

# In Rev14.py, add this new route

@app.route('/report_article/<article_hash_id>', methods=['POST'])
@login_required
def report_article(article_hash_id):
    # This feature is only for community articles
    article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first_or_404()

    # Check if the user has already reported this article
    existing_report = ReportedArticle.query.filter_by(
        article_id=article.id, 
        reporter_user_id=session['user_id']
    ).first()

    if existing_report:
        return jsonify({
            "success": False, 
            "error": "You have already reported this article."
        }), 409 # 409 Conflict

    # Create a new report record
    new_report = ReportedArticle(
        article_id=article.id,
        reporter_user_id=session['user_id']
        # You could expand this to include a reason from the request body
    )
    
    try:
        db.session.add(new_report)
        db.session.commit()
        app.logger.info(f"User {session['user_id']} reported article {article.id} ({article_hash_id})")
        return jsonify({
            "success": True, 
            "message": "Article has been reported for review. Thank you."
        })
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error reporting article {article.id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A database error occurred."}), 500

@app.route('/news')
@app.route('/news/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    session['previous_list_page'] = request.full_path
    per_page = app.config['PER_PAGE']
    query_str = request.args.get('query')
    filter_date_str = request.args.get('filter_date')

    # This block handles the main homepage view
    if page == 1 and category_name == 'All Articles' and not query_str and not filter_date_str:
        app.logger.info("Rendering main homepage with AI Synthesis and other sections.")
        
        # --- NEW: Call the synthesis function ---
        synthesis_data = get_daily_synthesis()
        
        # Fetch articles for the other sections
        all_popular_articles = fetch_popular_news()
        featured_article = all_popular_articles[0] if all_popular_articles else None
        popular_articles = all_popular_articles[1:] if all_popular_articles else [] 
        
        latest_yesterday_articles = fetch_yesterdays_latest_news()

        POPULAR_NEWS_COUNT = 6
        LATEST_NEWS_COUNT = 6
        
        user_bookmarks_hashes = set()
        if 'user_id' in session:
            bookmarks = BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()
            user_bookmarks_hashes = {b.article_hash_id for b in bookmarks}

        for art in ([featured_article] + popular_articles + latest_yesterday_articles):
            if art:
                art['is_bookmarked'] = art.get('id') in user_bookmarks_hashes

        return render_template("INDEX_HTML_TEMPLATE",
                               # --- NEW: Pass synthesis data to the template ---
                               synthesis=synthesis_data.get('synthesis_text'),
                               keywords=synthesis_data.get('keywords', []),
                               # --- Existing variables ---
                               featured_article=featured_article,
                               popular_articles=popular_articles[:POPULAR_NEWS_COUNT],
                               latest_yesterday_articles=latest_yesterday_articles[:LATEST_NEWS_COUNT],
                               selected_category=category_name,
                               is_main_homepage=True,
                               current_page=1, total_pages=1, query=None, current_filter_date=None)

    # This 'else' block handles all other paginated views and remains unchanged
    else:
        app.logger.info(f"Rendering standard list view for: category='{category_name}', page='{page}'")
        all_display_articles_raw = []
        if category_name == 'Popular Stories':
            all_display_articles_raw = fetch_popular_news()
        elif category_name == "Yesterday's Headlines":
            all_display_articles_raw = fetch_yesterdays_latest_news()
        elif category_name == 'Community Hub':
            db_articles = CommunityArticle.query.options(joinedload(CommunityArticle.author)).order_by(CommunityArticle.published_at.desc()).all()
            for art in db_articles:
                art.is_community_article = True
            all_display_articles_raw.extend(db_articles)
        else:
            if filter_date_str:
                try: datetime.strptime(filter_date_str, '%Y-%m-%d')
                except ValueError: flash("Invalid date format.", "warning"); filter_date_str = None
            api_articles = fetch_news_from_api(target_date_str=filter_date_str)
            all_display_articles_raw.extend(api_articles)
            
        all_display_articles_raw.sort(key=get_sort_key, reverse=True)
        paginated_display_articles_raw, total_pages = get_paginated_articles(all_display_articles_raw, page, per_page)
        paginated_display_articles_with_bookmark_status = []
        user_bookmarks_hashes = set()
        if 'user_id' in session:
            bookmarks = BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()
            user_bookmarks_hashes = {b.article_hash_id for b in bookmarks}
        for art_item in paginated_display_articles_raw:
            if hasattr(art_item, 'is_community_article') and art_item.is_community_article:
                art_item.is_bookmarked = art_item.article_hash_id in user_bookmarks_hashes
                paginated_display_articles_with_bookmark_status.append(art_item)
            elif isinstance(art_item, dict):
                art_item_copy = art_item.copy()
                art_item_copy['is_bookmarked'] = art_item_copy.get('id') in user_bookmarks_hashes
                paginated_display_articles_with_bookmark_status.append(art_item_copy)
        return render_template("INDEX_HTML_TEMPLATE",
                               articles=paginated_display_articles_with_bookmark_status,
                               selected_category=category_name,
                               is_main_homepage=False,
                               current_page=page, total_pages=total_pages,
                               featured_article_on_this_page=False,
                               current_filter_date=filter_date_str, query=query_str)

@app.route('/user/<username>')
def public_profile(username):
    user = User.query.filter_by(username=username).first_or_404()
    
    # Fetch all articles posted by this user
    posted_articles = CommunityArticle.query.filter_by(user_id=user.id)\
        .order_by(CommunityArticle.published_at.desc())\
        .all()
        
    return render_template("PUBLIC_PROFILE_HTML_TEMPLATE", user=user, posted_articles=posted_articles)
    
@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    # --- STEP 1: VERIFICATION ---
    # This message will appear on your webpage if this function is running correctly.
    # If you don't see it, your server has not reloaded the new code.

    session['previous_list_page'] = request.full_path
    query_str = request.args.get('query', '').strip()
    per_page = app.config['PER_PAGE']

    if not query_str:
        return redirect(url_for('index'))

    # This log will appear in your server console (terminal).
    app.logger.info(f"Performing LIVE API search for query: '{query_str}'")
    
    # This list will hold the fresh results from our new API call.
    api_articles = []
    if newsapi:
        try:
            # --- STEP 2: LIVE API CALL ---
            # We are making a new, targeted request to the NewsAPI here.
            
            # Log the exact parameters to be sent to the API for debugging.
            app.logger.info(f"NEWSAPI CALL PARAMS: q='{query_str}', language='en', sort_by='relevancy'")

            search_response = newsapi.get_everything(
                q=query_str,          # Use the user's keyword for the query.
                language='en',
                sort_by='relevancy',  # Get the most relevant articles first.
                page_size=100         # Fetch a good number of articles for pagination.
            )

            # Log the raw response from the API to see exactly what it returned.
            app.logger.info(f"RAW API RESPONSE (first 500 chars): {str(search_response)[:500]}")

            # --- STEP 3: PROCESS THE RELEVANT RESULTS ---
            # The following code processes the fresh articles returned by the API.
            if search_response.get('status') == 'ok':
                raw_articles = search_response.get('articles', [])
                unique_urls = set()
                for art_data in raw_articles:
                    url = art_data.get('url')
                    if not url or url in unique_urls: 
                        continue
                        
                    title = art_data.get('title')
                    description = art_data.get('description')

                    # Skip articles that are malformed or removed.
                    if not all([title, description, art_data.get('source')]) or title == '[Removed]':
                        continue
                    
                    unique_urls.add(url)
                    article_id = generate_article_id(url)
                    source_name = art_data['source'].get('name', 'Unknown Source')
                    
                    published_at_dt = None
                    if art_data.get('publishedAt'):
                        try:
                            published_at_dt = datetime.fromisoformat(art_data.get('publishedAt').replace('Z', '+00:00'))
                        except ValueError:
                            published_at_dt = datetime.now(timezone.utc)
                    else:
                        published_at_dt = datetime.now(timezone.utc)
                    
                    standardized_article = {
                        'id': article_id,
                        'title': title,
                        'description': description,
                        'url': url,
                        'urlToImage': art_data.get('urlToImage') or None,
                        'publishedAt': published_at_dt.isoformat(),
                        'source': {'name': source_name},
                        'is_community_article': False
                    }
                    # Add to the master store so the article detail page can find it.
                    MASTER_ARTICLE_STORE[article_id] = standardized_article 
                    api_articles.append(standardized_article)
            else:
                # Handle cases where the API returns an error.
                api_error_message = search_response.get('message', 'Unknown API error')
                app.logger.error(f"NewsAPI error on search: {api_error_message}")
                flash(f"Could not perform search at this time. Error: {api_error_message}", "danger")

        except Exception as e:
            app.logger.error(f"Exception during API search for '{query_str}': {e}", exc_info=True)
            flash("An unexpected error occurred during the search.", "danger")
    
    # --- STEP 4: COMBINE WITH LOCAL RESULTS ---
    # Also search your app's own community-posted articles for the same keyword.
    # This matches each word independently across title, description AND body, so a
    # multi-word query like "modi economy" finds "Modi discusses the economy".
    # (The previous single ilike required the whole phrase to appear contiguously.)
    community_db_articles = []
    for art in search_community_articles(query_str):
        art.is_community_article = True
        community_db_articles.append(art)

    # --- STEP 5: FINALIZE AND RENDER ---
    # Combine API results and community results, then sort them by date.
    all_search_results_raw = api_articles + community_db_articles
    all_search_results_raw.sort(key=get_sort_key, reverse=True)

    # Paginate the final combined list.
    paginated_search_articles_raw, total_pages = get_paginated_articles(all_search_results_raw, page, per_page)

    # Add user-specific data like bookmark status before rendering.
    paginated_search_articles_with_bookmark_status = []
    user_bookmarks_hashes = set()
    if 'user_id' in session:
        bookmarks = BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()
        user_bookmarks_hashes = {b.article_hash_id for b in bookmarks}

    for art_item in paginated_search_articles_raw:
        if hasattr(art_item, 'is_community_article') and art_item.is_community_article:
            art_item.is_bookmarked = art_item.article_hash_id in user_bookmarks_hashes
            paginated_search_articles_with_bookmark_status.append(art_item)
        elif isinstance(art_item, dict):
            art_item_copy = art_item.copy()
            art_item_copy['is_bookmarked'] = art_item_copy.get('id') in user_bookmarks_hashes
            paginated_search_articles_with_bookmark_status.append(art_item_copy)
            
    # Render the results page.
    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_search_articles_with_bookmark_status,
                           selected_category=f"Search: {query_str}",
                           current_page=page,
                           total_pages=total_pages,
                           is_main_homepage=False,
                           featured_article_on_this_page=False,
                           query=query_str,
                           current_filter_date=None)


@app.route('/article/<article_hash_id>')
def article_detail(article_hash_id):
    article_data, is_community_article, is_bookmarked = None, False, False
    previous_list_page = session.get('previous_list_page', url_for('index'))

    _record_article_view(article_hash_id)

    article_db = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=article_hash_id).first()
    if article_db:
        article_data, is_community_article = article_db, True
        if article_data.groq_takeaways:
            try: article_data.parsed_takeaways = json.loads(article_data.groq_takeaways)
            except json.JSONDecodeError: article_data.parsed_takeaways = []
    else:
        if not MASTER_ARTICLE_STORE: fetch_news_from_api()
        article_api_dict = MASTER_ARTICLE_STORE.get(article_hash_id)
        if article_api_dict:
            article_data, is_community_article = article_api_dict.copy(), False
        else:
            flash("Article not found.", "danger"); return redirect(previous_list_page)

    if 'user_id' in session:
        is_bookmarked = bool(BookmarkedArticle.query.filter_by(user_id=session['user_id'], article_hash_id=article_hash_id).first())

    # --- SIMPLIFIED & STABLE COMMENT HANDLING ---
    comment_data, total_comment_count = {}, 0
    
    # Define a base query for all comments on this article, loading authors efficiently.
    base_comments_query = None
    if is_community_article:
        base_comments_query = Comment.query.options(joinedload(Comment.author)).filter_by(community_article_id=article_data.id)
    else:
        base_comments_query = Comment.query.options(joinedload(Comment.author)).filter_by(api_article_hash_id=article_hash_id)

    # 1. Get ALL comments (including replies) to process reactions.
    all_comments_in_thread = base_comments_query.all()
    total_comment_count = len(all_comments_in_thread)
    all_comment_ids = {c.id for c in all_comments_in_thread}

    # 2. Fetch reaction data for all comments in a single batch.
    if all_comment_ids:
        for c_id in all_comment_ids: comment_data[c_id] = {'reactions': {}, 'user_reaction': None}
        
        reaction_counts = db.session.query(
            CommentVote.comment_id, CommentVote.vote_emoji, func.count(CommentVote.vote_emoji)
        ).filter(CommentVote.comment_id.in_(all_comment_ids)).group_by(CommentVote.comment_id, CommentVote.vote_emoji).all()
        for c_id, emoji, count in reaction_counts:
            if c_id in comment_data: comment_data[c_id]['reactions'][emoji] = count

        if 'user_id' in session:
            user_reactions = CommentVote.query.filter(CommentVote.comment_id.in_(all_comment_ids), CommentVote.user_id==session['user_id']).all()
            for vote in user_reactions:
                if vote.comment_id in comment_data: comment_data[vote.comment_id]['user_reaction'] = vote.vote_emoji

    # 3. Fetch only the TOP-LEVEL comments to start the template loop.
    # The 'replies' relationship in the model will handle fetching children.
    comments_for_template = base_comments_query.filter(Comment.parent_id.is_(None)).order_by(Comment.timestamp.asc()).all()

    if isinstance(article_data, dict): article_data['is_community_article'] = False
    elif article_data: article_data.is_community_article = True
            
    return render_template("ARTICLE_HTML_TEMPLATE", 
                           article=article_data, 
                           is_community_article=is_community_article, 
                           comments=comments_for_template, 
                           comment_data=comment_data,
                           total_comment_count=total_comment_count,
                           previous_list_page=previous_list_page, 
                           is_bookmarked=is_bookmarked)

@app.route('/get_article_content/<article_hash_id>')
def get_article_content_json(article_hash_id):
    if not MASTER_ARTICLE_STORE and not CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first(): fetch_news_from_api()
    article_data = MASTER_ARTICLE_STORE.get(article_hash_id)
    if not article_data or 'url' not in article_data: return jsonify({"error": "Article data or URL not found in API cache"}), 404
    if article_data.get('groq_summary') is not None and article_data.get('groq_takeaways') is not None:
        app.logger.info(f"Returning cached Groq analysis from MASTER_ARTICLE_STORE for API article ID: {article_hash_id}")
        return jsonify({"groq_analysis": {"groq_summary": article_data['groq_summary'], "groq_takeaways": article_data['groq_takeaways'], "error": None}, "error": None})
    processed_content = fetch_and_parse_article_content(article_hash_id, article_data['url'])
    return jsonify(processed_content)

# In Rev14.py, replace the entire add_comment function with this definitive version.

@app.route('/add_comment/<article_hash_id>', methods=['POST'])
@login_required
@rate_limit(12, 300, scope='add_comment', message="You're commenting very quickly. Please wait a moment.")
def add_comment(article_hash_id):
    payload = request.get_json(silent=True) or {}
    content, content_err = clean_text(payload.get('content'), LIMITS['comment'])
    if content_err:
        return jsonify({"success": False, "error": content_err if content else "Comment cannot be empty."}), 400

    parent_id = payload.get('parent_id')
    
    user = User.query.get(session['user_id'])
    if not user:
        # This is a fallback, the decorator should handle it.
        return jsonify({"success": False, "error": "User not found."}), 401

    # --- ROBUST LOGIC ---
    # The new logic is simpler and more reliable.
    # It trusts the article_hash_id from the page the user is on.
    
    new_comment = Comment(content=content, user_id=user.id)
    
    # First, check if it's a permanent community article from our database.
    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    
    if community_article:
        # If it is, link the comment to it.
        new_comment.community_article_id = community_article.id
    else:
        # If not, assume it's an API article and save the hash ID.
        # We no longer check against the volatile MASTER_ARTICLE_STORE cache.
        new_comment.api_article_hash_id = article_hash_id

    # parent_id comes from the client, so confirm it exists AND belongs to this same
    # article - otherwise a reply could be grafted into another article's thread.
    if parent_id not in (None, '', 0):
        try:
            parent_id = int(parent_id)
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "Invalid reply target."}), 400
        parent = Comment.query.get(parent_id)
        if not parent:
            return jsonify({"success": False, "error": "The comment you replied to no longer exists."}), 404
        same_thread = (
            (community_article and parent.community_article_id == community_article.id)
            or (not community_article and parent.api_article_hash_id == article_hash_id)
        )
        if not same_thread:
            app.logger.warning("Rejected cross-article reply: comment %s -> article %s", parent_id, article_hash_id)
            return jsonify({"success": False, "error": "Invalid reply target."}), 400
        new_comment.parent_id = parent.id

    try:
        db.session.add(new_comment)
        db.session.commit()
        db.session.refresh(new_comment)
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error saving comment to database: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A database error occurred. Could not save comment."}), 500

    # Render the new comment's HTML on the server.
    comment_html = render_template("_COMMENT_TEMPLATE", comment=new_comment, session=session)

    return jsonify({
        "success": True, 
        "html": comment_html,
        "parent_id": new_comment.parent_id
    }), 201
    
@app.route('/vote_comment/<int:comment_id>', methods=['POST'])
@login_required
def vote_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    emoji = (request.get_json(silent=True) or {}).get('emoji')
    
    # Define the set of allowed emojis for reactions.
    allowed_emojis = ['👍', '❤️', '😂', '😮', '😢', '😠']
    if not emoji or emoji not in allowed_emojis:
        return jsonify({"error": "Invalid reaction."}), 400

    existing_vote = CommentVote.query.filter_by(user_id=session['user_id'], comment_id=comment_id).first()
    
    user_reaction_after_vote = None
    
    if existing_vote:
        # If the user clicks the same emoji again, it's an "un-react", so we delete the vote.
        if existing_vote.vote_emoji == emoji:
            db.session.delete(existing_vote)
            user_reaction_after_vote = None
        # If they click a different emoji, we update their existing vote.
        else:
            existing_vote.vote_emoji = emoji
            user_reaction_after_vote = emoji
    # If no vote exists from this user, create a new one.
    else:
        new_vote = CommentVote(user_id=session['user_id'], comment_id=comment_id, vote_emoji=emoji)
        db.session.add(new_vote)
        user_reaction_after_vote = emoji
        
    db.session.commit()

    # After any change, recalculate all reaction counts for this specific comment.
    reaction_counts_query = db.session.query(
        CommentVote.vote_emoji,
        func.count(CommentVote.vote_emoji)
    ).filter(CommentVote.comment_id == comment_id).group_by(CommentVote.vote_emoji).all()
    
    # Format the counts into a dictionary for the frontend.
    reactions = {emo: count for emo, count in reaction_counts_query}

    return jsonify({
        "success": True, 
        "reactions": reactions, 
        "user_reaction": user_reaction_after_vote
    }), 200

# In Rev14.py, add these two new functions

@app.route('/delete_comment/<int:comment_id>', methods=['POST'])
@login_required
def delete_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)

    # Security Check: Ensure the logged-in user is the owner of the comment.
    if comment.user_id != session['user_id']:
        return jsonify({"success": False, "error": "You are not authorized to delete this comment."}), 403

    try:
        # Thanks to `cascade="all, delete-orphan"` in our Comment model's 'replies' relationship,
        # deleting the parent comment will automatically delete all its replies from the database.
        db.session.delete(comment)
        db.session.commit()
        app.logger.info(f"User {session['user_id']} deleted comment {comment_id} and its replies.")
        return jsonify({"success": True})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting comment {comment_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A database error occurred."}), 500


@app.route('/edit_comment/<int:comment_id>', methods=['POST'])
@login_required
def edit_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    
    # Security Check: Ensure the logged-in user is the owner of the comment.
    if comment.user_id != session['user_id']:
        return jsonify({"success": False, "error": "You are not authorized to edit this comment."}), 403

    new_content, _content_err = clean_text((request.get_json(silent=True) or {}).get('content'), LIMITS['comment'])
    if _content_err:
        return jsonify({"success": False, "error": _content_err}), 400
    if not new_content:
        return jsonify({"success": False, "error": "Comment content cannot be empty."}), 400

    try:
        comment.content = new_content
        db.session.commit()
        app.logger.info(f"User {session['user_id']} edited comment {comment_id}.")
        # Return the new content so the frontend can display it instantly.
        return jsonify({"success": True, "new_content": comment.content})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error editing comment {comment_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A database error occurred."}), 500

@app.route('/post_article', methods=['POST'])
@login_required
@rate_limit(8, 3600, scope='post_article', message="You've posted several articles recently. Please try again later.")
def post_article():
    title, description, content, source_name, image_url = map(lambda x: request.form.get(x, '').strip(), ['title', 'description', 'content', 'sourceName', 'imageUrl'])
    source_name = source_name or 'Community Post'

    # Validate against the column limits so oversized input is a friendly message
    # rather than a database error and a 500 page.
    for value, key, label in (
        (title, 'article_title', 'Title'),
        (description, 'article_description', 'Description'),
        (content, 'article_content', 'Full content'),
        (source_name, 'source_name', 'Source name'),
    ):
        cleaned, err = clean_text(value, LIMITS[key])
        if err:
            flash(f"{label}: {err}", "danger")
            return redirect(safe_redirect_target(request.referrer))

    if image_url:
        parsed_img = urllib.parse.urlparse(image_url)
        if parsed_img.scheme not in ('http', 'https') or len(image_url) > LIMITS['image_url']:
            flash("Image URL must be a valid http(s) link under 500 characters.", "warning")
            return redirect(safe_redirect_target(request.referrer))

    article_hash_id = generate_article_id(title + str(session['user_id']) + str(time.time()))
    groq_analysis_result = get_article_analysis_with_groq(content, title)
    groq_summary_text, groq_takeaways_json_str = None, None
    if groq_analysis_result and not groq_analysis_result.get("error"):
        groq_summary_text = groq_analysis_result.get('groq_summary')
        takeaways_list = groq_analysis_result.get('groq_takeaways')
        if takeaways_list and isinstance(takeaways_list, list): groq_takeaways_json_str = json.dumps(takeaways_list)
    # No image is fine: the templates render a styled fallback tile. (Previously this
    # pointed at via.placeholder.com, a third-party service that often fails to load.)
    new_article = CommunityArticle(article_hash_id=article_hash_id, title=title, description=description, full_text=content, source_name=source_name, image_url=image_url or None, user_id=session['user_id'], published_at=datetime.now(timezone.utc), groq_summary=groq_summary_text, groq_takeaways=groq_takeaways_json_str)
    try:
        db.session.add(new_article); db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error saving community article: {e}", exc_info=True)
        flash("Could not post your article right now. Please try again.", "danger")
        return redirect(safe_redirect_target(request.referrer))
    flash("Your article has been posted!", "success")
    return redirect(url_for('article_detail', article_hash_id=new_article.article_hash_id))

@app.route('/register', methods=['GET', 'POST'])
@rate_limit(5, 3600, scope='register', message="Too many accounts created from here recently. Please try again later.")
def register():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name, username, password = request.form.get('name', '').strip(), request.form.get('username', '').strip().lower(), request.form.get('password', '')
        name, name_err = clean_text(name, LIMITS['name'])
        if name_err: flash(f'Name: {name_err}', 'danger')
        elif not all([name, username, password]): flash('All fields are required.', 'danger')
        elif not USERNAME_RE.match(username):
            flash('Username must be 3-80 characters, using only letters, numbers, dots, underscores or hyphens.', 'warning')
        elif len(password) < 6: flash('Password must be at least 6 characters.', 'warning')
        elif len(password) > 200: flash('Password must be under 200 characters.', 'warning')
        elif User.query.filter_by(username=username).first(): flash('Username already exists. Please choose another.', 'warning')
        else:
            new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
            try:
                db.session.add(new_user); db.session.commit()
            except Exception as e:
                # Two simultaneous signups for the same username: the unique constraint
                # catches what the check above can't.
                db.session.rollback()
                app.logger.warning("Registration failed for username=%r: %s", username, e)
                flash('That username was just taken. Please try another.', 'warning')
                return redirect(url_for('register'))
            flash(f'Registration successful, {name}! Please log in.', 'success')
            return redirect(url_for('login'))
        return redirect(url_for('register'))
    return render_template("REGISTER_HTML_TEMPLATE")

@app.route('/delete_community_article/<article_hash_id>', methods=['POST'])
@login_required
def delete_community_article(article_hash_id):
    # Security Check: Ensure the user has the admin flag from the session.
    if not session.get('is_admin') or session.get('username') != ADMIN_USERNAME:
        return jsonify({"success": False, "error": "Administrator access required."}), 403

    article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    
    if not article:
        return jsonify({"success": False, "error": "Article not found."}), 404

    try:
        # The 'cascade' option in the models will handle related deletions.
        db.session.delete(article)
        db.session.commit()
        app.logger.info(f"Admin user 'vbdevil' deleted community article {article.id} ({article_hash_id})")
        flash("Community article has been successfully deleted by the administrator.", "success")
        return jsonify({"success": True, "redirect_url": url_for('index', category_name='Community Hub')})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting community article {article.id} by admin: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A database error occurred during deletion."}), 500

@app.route('/login', methods=['GET', 'POST'])
@rate_limit(10, 300, scope='login', message="Too many login attempts. Please wait a few minutes and try again.")
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username, password = request.form.get('username', '').strip().lower(), request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            # Drop any pre-login session state (incl. the old CSRF token) so a token
            # fixed by an attacker before login can't be reused afterwards.
            session.clear()
            session.permanent = True
            session['user_id'] = user.id
            session['user_name'] = user.name
            # Store username for easy access in templates/routes
            session['username'] = user.username
            session['is_admin'] = (user.username == ADMIN_USERNAME)

            flash(f"Welcome back, {user.name}!", "success")
            # `next` is attacker-controllable, so only follow same-host targets.
            next_url = safe_redirect_target(request.args.get('next'))
            session.pop('previous_list_page', None)
            return redirect(next_url)
        else:
            app.logger.info("Failed login attempt for username=%r from %s", username, _client_identity())
            flash('Invalid username or password.', 'danger')
    return render_template("LOGIN_HTML_TEMPLATE")

@app.route('/logout')
def logout(): session.clear(); flash("You have been successfully logged out.", "info"); return redirect(url_for('index'))
@app.route('/about')
def about(): return render_template("ABOUT_US_HTML_TEMPLATE")
@app.route('/contact')
def contact(): return render_template("CONTACT_HTML_TEMPLATE")
@app.route('/privacy')
def privacy(): return render_template("PRIVACY_POLICY_HTML_TEMPLATE")


# ==============================================================================
# --- Terms of service ---
# ==============================================================================
TERMS_UPDATED = "November 6, 2026"
CONTACT_EMAIL = os.environ.get('CONTACT_EMAIL', 'vbansal639@gmail.com')


@app.route('/terms')
def terms():
    return render_template("TERMS_HTML_TEMPLATE", terms_updated=TERMS_UPDATED, contact_email=CONTACT_EMAIL)


# ==============================================================================
# --- Testimonials ---
# Readers send a note (with consent), the admin approves it, and approved notes show on
# the landing page. Nothing is ever published without review.
# ==============================================================================
TESTIMONIAL_LIMITS = {'name': 120, 'attribution': 80, 'quote': 300}
TESTIMONIAL_MIN_QUOTE = 20


def _is_admin():
    return bool(session.get('is_admin')) and session.get('username') == ADMIN_USERNAME


def _read_testimonial_form():
    """Validate the shared name / attribution / quote fields. Returns (data, error)."""
    name, err = clean_text(request.form.get('name', ''), TESTIMONIAL_LIMITS['name'])
    if err:
        return None, f"Name: {err}"
    attribution, err = clean_text(request.form.get('attribution', ''), TESTIMONIAL_LIMITS['attribution'], allow_empty=True)
    if err:
        return None, f"About you: {err}"
    quote, err = clean_text(request.form.get('quote', ''), TESTIMONIAL_LIMITS['quote'])
    if err:
        return None, f"Note: {err}"
    if len(quote) < TESTIMONIAL_MIN_QUOTE:
        return None, f"Please write at least {TESTIMONIAL_MIN_QUOTE} characters."
    return {'name': name, 'attribution': attribution or None, 'quote': quote}, None


@app.route('/share-your-experience', methods=['GET', 'POST'])
@login_required
@rate_limit(5, 3600, scope='testimonial', message="You've sent several notes recently. Please try again later.")
def share_testimonial():
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('login'))
    mine = user.testimonials.order_by(Testimonial.created_at.desc()).all()
    has_pending = any(t.status == 'pending' for t in mine)
    form = {'name': user.name, 'attribution': '', 'quote': ''}

    if request.method == 'POST':
        data, error = _read_testimonial_form()
        if has_pending:
            flash("Your last note is still waiting for review.", "info")
            return redirect(url_for('share_testimonial'))
        if not error and request.form.get('consent') != 'yes':
            error = "Please agree that we may publish your note."
        if error:
            flash(error, "warning")
            form.update({k: request.form.get(k, '') for k in ('name', 'attribution', 'quote')})
        else:
            try:
                db.session.add(Testimonial(user_id=user.id, source='reader', status='pending', **data))
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Saving testimonial failed for user {user.id}: {e}", exc_info=True)
                flash("We couldn't save your note. Please try again.", "danger")
                return redirect(url_for('share_testimonial'))
            flash("Thanks. Your note is waiting for review and will appear on the website once approved.", "success")
            return redirect(url_for('share_testimonial'))

    return render_template("SHARE_TESTIMONIAL_HTML_TEMPLATE", mine=mine, has_pending=has_pending,
                           form=form, limits=TESTIMONIAL_LIMITS)


@app.route('/admin/testimonials')
@login_required
def admin_testimonials():
    if not _is_admin():
        abort(403)
    rows = (Testimonial.query.options(joinedload(Testimonial.user))
            .order_by(Testimonial.created_at.desc()).all())
    groups = {'pending': [], 'approved': [], 'rejected': []}
    for t in rows:
        groups.setdefault(t.status, []).append(t)
    return render_template("ADMIN_TESTIMONIALS_HTML_TEMPLATE", groups=groups, limits=TESTIMONIAL_LIMITS)


@app.route('/admin/testimonials/add', methods=['POST'])
@login_required
def admin_add_testimonial():
    if not _is_admin():
        abort(403)
    data, error = _read_testimonial_form()
    if not error and request.form.get('permission') != 'yes':
        error = "Confirm that the reader gave permission to publish."
    if error:
        flash(error, "warning")
        return redirect(url_for('admin_testimonials'))
    try:
        db.session.add(Testimonial(source='admin', status='approved',
                                   reviewed_at=datetime.now(timezone.utc), **data))
        db.session.commit()
        flash("Quote added and published.", "success")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Admin testimonial add failed: {e}", exc_info=True)
        flash("Could not add the quote. Please try again.", "danger")
    return redirect(url_for('admin_testimonials'))


@app.route('/admin/testimonials/<int:testimonial_id>/<action>', methods=['POST'])
@login_required
def admin_review_testimonial(testimonial_id, action):
    if not _is_admin():
        abort(403)
    if action not in ('approve', 'reject', 'delete'):
        abort(404)
    t = Testimonial.query.get_or_404(testimonial_id)
    try:
        if action == 'delete':
            db.session.delete(t)
            message = "Testimonial deleted."
        else:
            t.status = 'approved' if action == 'approve' else 'rejected'
            t.reviewed_at = datetime.now(timezone.utc)
            message = "Testimonial approved. It is now on the landing page." if action == 'approve' else "Testimonial hidden from the website."
        db.session.commit()
        flash(message, "success")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Testimonial review failed ({action} {testimonial_id}): {e}", exc_info=True)
        flash("Could not update the testimonial. Please try again.", "danger")
    return redirect(url_for('admin_testimonials'))


@app.route('/welcome')
def landing_old_address():
    """The landing page used to live here; keep old links working."""
    return redirect(url_for('landing'), code=301)


@app.route('/page/<int:page>')
def index_old_pagination(page):
    """The feed moved from / to /news; keep old paginated links working."""
    return redirect(url_for('index', page=page), code=301)


def _landing_testimonials(limit=6):
    """Approved testimonials for the landing page, newest approval first."""
    try:
        rows = (Testimonial.query.options(joinedload(Testimonial.user))
                .filter_by(status='approved')
                .order_by(Testimonial.reviewed_at.desc(), Testimonial.created_at.desc())
                .limit(limit).all())
    except Exception as e:
        app.logger.warning(f"Landing page testimonials lookup failed: {e}")
        return []
    items = []
    for t in rows:
        details = [t.attribution] if t.attribution else []
        if t.user and t.user.created_at:
            details.append('Reader since ' + t.user.created_at.strftime('%b %Y'))
        initials = ''.join(part[0] for part in t.name.split()[:2]).upper() or '?'
        items.append({'quote': t.quote, 'name': t.name, 'detail': ' · '.join(details), 'initials': initials})
    return items


# ==============================================================================
# --- Landing page (/welcome) ---
# The news feed stays at '/'. This page explains the product to new visitors and
# sends them to the feed. Everything it states is read from the app's own config and
# data, so it cannot drift out of date or claim something the app does not do.
# ==============================================================================
_LANDING_SOURCE_LABELS = (
    ('thehindu', 'The Hindu'),
    ('ndtv', 'NDTV'),
    ('indianexpress', 'The Indian Express'),
    ('hindustantimes', 'Hindustan Times'),
    ('timesofindia', 'The Times of India'),
)
_NUMBER_WORDS = ('', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')
LANDING_TAGLINE = ("You don't need every word of every story. "
                   "You need what happened, in plain words, and a link to the rest.")
# A small weekly count reads as a weakness rather than proof, so it stays hidden below this.
LANDING_STAT_MIN = int(os.environ.get('LANDING_STAT_MIN', '20'))
LANDING_OG_IMAGE = 'og-welcome.jpg'


def _join_names(names):
    if len(names) <= 1:
        return ''.join(names)
    return ', '.join(names[:-1]) + ' and ' + names[-1]


def _landing_source_names():
    """Publisher names behind the main feed, taken from the feeds actually configured."""
    if os.environ.get('USE_RSS', 'true').lower() in ('1', 'true', 'yes'):
        locations = _rss_feed_urls()
    else:
        locations = app.config['NEWS_API_DOMAINS'].split(',')
    names = []
    for loc in locations:
        loc = loc.strip()
        if not loc:
            continue
        lowered = loc.lower()
        name = next((label for needle, label in _LANDING_SOURCE_LABELS if needle in lowered), None)
        if not name:
            host = urllib.parse.urlparse(loc if '://' in loc else f'https://{loc}').hostname or ''
            name = host[4:] if host.startswith('www.') else host
        if name and name not in names:
            names.append(name)
    return names


def _landing_sample_story():
    """
    The most recent real story that already has a saved AI summary, for the hero card.
    Reads only the database and the in-memory article store, so it never triggers a news
    fetch or an AI call. Returns None when nothing suitable is cached (e.g. right after a
    restart); the template then shows what every summary contains instead.
    """
    try:
        rows = (ArticleAnalysis.query
                .filter(ArticleAnalysis.groq_summary.isnot(None))
                .order_by(ArticleAnalysis.created_at.desc())
                .limit(60).all())
    except Exception as e:
        app.logger.warning(f"Landing page sample story lookup failed: {e}")
        return None
    for row in rows:
        article = MASTER_ARTICLE_STORE.get(row.article_hash_id)
        if not article or not article.get('title'):
            continue
        try:
            takeaways = json.loads(row.groq_takeaways) if row.groq_takeaways else []
        except (ValueError, TypeError):
            takeaways = []
        if not isinstance(takeaways, list):
            continue
        takeaways = [t.strip() for t in takeaways if isinstance(t, str) and t.strip()][:4]
        summary = (row.groq_summary or '').strip()
        if not summary or len(takeaways) < 2:
            continue
        published = to_ist_filter(article.get('publishedAt')) if article.get('publishedAt') else None
        if published and (published.startswith('Invalid') or published == 'N/A'):
            published = None
        return {
            'title': article['title'],
            'source': (article.get('source') or {}).get('name') or 'Publisher',
            'published': published,
            'summary': summary,
            'takeaways': takeaways,
            'url': url_for('article_detail', article_hash_id=row.article_hash_id),
        }
    return None


def _landing_weekly_summary_count():
    try:
        since = datetime.now(timezone.utc) - timedelta(days=7)
        return ArticleAnalysis.query.filter(ArticleAnalysis.created_at >= since).count()
    except Exception as e:
        app.logger.warning(f"Landing page weekly count failed: {e}")
        return 0


def _landing_faqs(sources_sentence):
    feed_line = (f"The main feed comes from the public news feeds of {sources_sentence}. "
                 if sources_sentence else "The main feed comes from the public news feeds of major Indian newsrooms. ")
    return [
        {"q": "Is BrieflyAI free?",
         "a": "Yes. Every story, summary and takeaway on BrieflyAI is free to read, and you don't need an account or a card."},
        {"q": "Do I need an account?",
         "a": "Not to read. A free account needs only a name, a username and a password, with no email address. "
              "It lets you bookmark stories, react and reply in the comments, and post articles to the Community Hub."},
        {"q": "Where does the news come from?",
         "a": feed_line + "The popular stories and yesterday's headlines sections can include other outlets too. "
              "Community Hub posts are written by registered readers and are marked as community posts."},
        {"q": "Who writes the summaries?",
         "a": "A large language model writes them from the text of each article. It is told to stay neutral, "
              "lead with the most important fact, keep the summary under 60 words and list three or four takeaways."},
        {"q": "Can a summary get something wrong?",
         "a": "It can. AI can miss context or nuance, which is why every summary links to the original article. "
              "When a story matters to you, read the full report at the source."},
        {"q": "How fresh is the news?",
         "a": "New stories are pulled in through the day, and the big picture briefing at the top of the feed "
              "is rewritten every few hours."},
        {"q": "Does it work on my phone?",
         "a": "Yes. It works in any mobile browser, you can add it to your home screen like an app, "
              "and pages you have already opened stay readable when you lose signal."},
        {"q": "Can I download or delete my data?",
         "a": "Yes. From your profile you can export everything tied to your account, "
              "or delete the account permanently."},
    ]


@app.route('/')
def landing():
    """Homepage: the landing page. The news feed lives at /news."""
    sources = _landing_source_names()
    sources_sentence = _join_names(sources)
    faqs = _landing_faqs(sources_sentence)

    weekly = _landing_weekly_summary_count()

    # Render terminates TLS in front of the app, so trust its forwarded scheme for absolute URLs.
    scheme = request.headers.get('X-Forwarded-Proto', request.scheme).split(',')[0].strip() or 'https'
    origin = f"{scheme}://{request.host}"
    og_image_url = None
    if app.static_folder and os.path.exists(os.path.join(app.static_folder, LANDING_OG_IMAGE)):
        og_image_url = origin + url_for('static', filename=LANDING_OG_IMAGE)

    pref = request.cookies.get('darkMode')
    theme = 'dark' if pref == 'enabled' else ('light' if pref == 'disabled' else 'auto')

    return render_template(
        "LANDING_HTML_TEMPLATE",
        theme=theme,
        canonical_url=origin + url_for('landing'),
        og_image_url=og_image_url,
        sources=sources,
        sources_sentence=sources_sentence,
        sources_short=', '.join(sources[:3]) if sources else 'major Indian newsrooms',
        source_count_word=_NUMBER_WORDS[len(sources)] if len(sources) < len(_NUMBER_WORDS) else str(len(sources)),
        sample_story=_landing_sample_story(),
        testimonials=_landing_testimonials(),
        weekly_summaries=weekly if weekly >= LANDING_STAT_MIN else None,
        tagline=LANDING_TAGLINE,
        faqs=faqs,
        faq_schema={
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [{
                "@type": "Question",
                "name": item["q"],
                "acceptedAnswer": {"@type": "Answer", "text": item["a"]},
            } for item in faqs],
        },
    )

@app.route('/subscribe', methods=['POST'])
@rate_limit(5, 3600, scope='subscribe', message="Too many subscription attempts. Please try again later.")
def subscribe():
    email = request.form.get('email', '').strip().lower()
    if not email:
        flash('Email is required to subscribe.', 'warning')
    elif len(email) > LIMITS['email'] or not EMAIL_RE.match(email):
        flash('Please enter a valid email address.', 'warning')
    elif Subscriber.query.filter_by(email=email).first():
        flash('You are already subscribed to our newsletter.', 'info')
    else:
        try:
            db.session.add(Subscriber(email=email)); db.session.commit(); flash('Thank you for subscribing!', 'success')
        except Exception as e:
            db.session.rollback(); app.logger.error(f"Error subscribing email: {e}"); flash('Could not subscribe at this time. Please try again later.', 'danger')
    return redirect(safe_redirect_target(request.referrer))

@app.route('/toggle_bookmark/<article_hash_id>', methods=['POST'])
@login_required
def toggle_bookmark(article_hash_id):
    user_id = session['user_id']
    is_community_str = request.json.get('is_community_article', 'false').lower()
    is_community = True if is_community_str == 'true' else False
    article_title_cache = request.json.get('title', 'Bookmarked Article')
    article_source_cache = request.json.get('source_name', 'Unknown Source')
    article_image_cache = request.json.get('image_url', None)
    article_desc_cache = request.json.get('description', None)
    article_published_at_cache_str = request.json.get('published_at', None)
    article_published_at_dt = None
    if article_published_at_cache_str:
        try:
            if article_published_at_cache_str.endswith('Z'): article_published_at_dt = datetime.fromisoformat(article_published_at_cache_str[:-1] + '+00:00')
            else: article_published_at_dt = datetime.fromisoformat(article_published_at_cache_str)
            if article_published_at_dt.tzinfo is None: article_published_at_dt = pytz.utc.localize(article_published_at_dt)
        except ValueError: app.logger.warning(f"Could not parse published_at_cache_str for bookmark: {article_published_at_cache_str}"); article_published_at_dt = None
    existing_bookmark = BookmarkedArticle.query.filter_by(user_id=user_id, article_hash_id=article_hash_id).first()
    if existing_bookmark:
        db.session.delete(existing_bookmark); db.session.commit()
        return jsonify({"success": True, "status": "removed", "message": "Bookmark removed."})
    else:
        if is_community:
            if not CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first(): return jsonify({"success": False, "error": "Community article not found."}), 404
        else:
            if article_hash_id not in MASTER_ARTICLE_STORE:
                fetch_news_from_api() 
                if article_hash_id not in MASTER_ARTICLE_STORE: return jsonify({"success": False, "error": "API article not found."}), 404
        new_bookmark = BookmarkedArticle(user_id=user_id, article_hash_id=article_hash_id, is_community_article=is_community, title_cache=article_title_cache, source_name_cache=article_source_cache, image_url_cache=article_image_cache, description_cache=article_desc_cache, published_at_cache=article_published_at_dt)
        db.session.add(new_bookmark); db.session.commit()
        return jsonify({"success": True, "status": "added", "message": "Article bookmarked!"})

@app.route('/profile')
@login_required
def profile():
    user = User.query.get_or_404(session['user_id'])
    page = request.args.get('page', 1, type=int) or 1
    page = max(1, min(page, MAX_PAGE))
    per_page = app.config['PER_PAGE']
    user_posted_articles = CommunityArticle.query.filter_by(user_id=user.id).order_by(CommunityArticle.published_at.desc()).all()
    bookmarks_query = BookmarkedArticle.query.filter_by(user_id=user.id).order_by(BookmarkedArticle.bookmarked_at.desc())
    user_bookmarks_paginated_query = bookmarks_query.paginate(page=page, per_page=per_page, error_out=False)
    user_bookmarked_articles_data = []
    for bookmark in user_bookmarks_paginated_query.items:
        article_detail_data = None
        if bookmark.is_community_article:
            comm_art = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=bookmark.article_hash_id).first()
            if comm_art: article_detail_data = {'id': comm_art.article_hash_id, 'title': comm_art.title, 'description': comm_art.description, 'urlToImage': comm_art.image_url, 'publishedAt': comm_art.published_at.isoformat() if comm_art.published_at else None, 'source': {'name': comm_art.author.name if comm_art.author else comm_art.source_name}, 'is_community_article': True, 'article_url': url_for('article_detail', article_hash_id=comm_art.article_hash_id)}
        else:
            api_art = MASTER_ARTICLE_STORE.get(bookmark.article_hash_id)
            if api_art: article_detail_data = {'id': api_art['id'], 'title': api_art['title'], 'description': api_art['description'], 'urlToImage': api_art['urlToImage'], 'publishedAt': api_art['publishedAt'], 'source': {'name': api_art['source']['name']}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=api_art['id'])}
            else: article_detail_data = {'id': bookmark.article_hash_id, 'title': bookmark.title_cache or "Bookmarked Article (Details N/A)", 'description': bookmark.description_cache or "Description not available.", 'urlToImage': bookmark.image_url_cache or None, 'publishedAt': bookmark.published_at_cache.isoformat() if bookmark.published_at_cache else None, 'source': {'name': bookmark.source_name_cache or "Unknown Source"}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=bookmark.article_hash_id), 'is_stale_bookmark': True}
        if article_detail_data: user_bookmarked_articles_data.append(article_detail_data)
    return render_template("PROFILE_HTML_TEMPLATE", user=user, posted_articles=user_posted_articles, bookmarked_articles=user_bookmarked_articles_data, bookmarks_pagination=user_bookmarks_paginated_query, current_page=page)

@app.errorhandler(404)
def page_not_found(e): return render_template("404_TEMPLATE"), 404
@app.errorhandler(500)
def internal_server_error(e): db.session.rollback(); app.logger.error(f"500 error at {request.url}: {e}", exc_info=True); return render_template("500_TEMPLATE"), 500

# ==============================================================================
# --- 6b. Discovery, syndication, PWA and operations endpoints ---
# ==============================================================================

def _xml_escape(text):
    return (str(text or '')
            .replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            .replace('"', '&quot;').replace("'", '&apos;'))


@app.route('/feed.xml')
@app.route('/rss')
def rss_feed():
    """RSS 2.0 feed of the newest community articles."""
    try:
        articles = (CommunityArticle.query
                    .options(joinedload(CommunityArticle.author))
                    .order_by(CommunityArticle.published_at.desc())
                    .limit(40).all())
    except Exception as e:
        app.logger.error(f"RSS feed query failed: {e}", exc_info=True)
        articles = []

    items = []
    for art in articles:
        link = url_for('article_detail', article_hash_id=art.article_hash_id, _external=True)
        published = art.published_at
        if published and published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        pub_date = published.strftime('%a, %d %b %Y %H:%M:%S %z') if published else ''
        description = art.groq_summary or art.description or ''
        items.append(
            "<item>"
            f"<title>{_xml_escape(art.title)}</title>"
            f"<link>{_xml_escape(link)}</link>"
            f"<guid isPermaLink=\"true\">{_xml_escape(link)}</guid>"
            f"<description>{_xml_escape(description)}</description>"
            f"<author>{_xml_escape(art.author.name if art.author else 'BrieflyAI')}</author>"
            f"<pubDate>{_xml_escape(pub_date)}</pubDate>"
            "</item>"
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel>'
        '<title>BrieflyAI - Community Stories</title>'
        f'<link>{_xml_escape(url_for("index", _external=True))}</link>'
        '<description>AI-summarized, India-centric news and community perspectives.</description>'
        '<language>en-in</language>'
        f'<atom:link href="{_xml_escape(url_for("rss_feed", _external=True))}" rel="self" type="application/rss+xml" />'
        + ''.join(items) +
        '</channel></rss>'
    )
    return Response(xml, mimetype='application/rss+xml')


@app.route('/sitemap.xml')
def sitemap():
    """Sitemap covering static pages, category listings and community articles."""
    urls = []

    def add(loc, changefreq, priority, lastmod=None):
        entry = f"<url><loc>{_xml_escape(loc)}</loc>"
        if lastmod:
            entry += f"<lastmod>{lastmod.strftime('%Y-%m-%d')}</lastmod>"
        entry += f"<changefreq>{changefreq}</changefreq><priority>{priority}</priority></url>"
        urls.append(entry)

    add(url_for('landing', _external=True), 'weekly', '1.0')
    add(url_for('index', _external=True), 'hourly', '0.9')
    for endpoint in ('about', 'contact', 'privacy', 'terms'):
        add(url_for(endpoint, _external=True), 'monthly', '0.4')
    for category in app.config['CATEGORIES']:
        add(url_for('index', category_name=category, page=1, _external=True), 'hourly', '0.8')

    try:
        for art in (CommunityArticle.query
                    .order_by(CommunityArticle.published_at.desc())
                    .limit(2000).all()):
            add(url_for('article_detail', article_hash_id=art.article_hash_id, _external=True),
                'weekly', '0.7', art.published_at)
    except Exception as e:
        app.logger.error(f"Sitemap query failed: {e}", exc_info=True)

    xml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           + ''.join(urls) + '</urlset>')
    return Response(xml, mimetype='application/xml')


@app.context_processor
def inject_ads():
    on_page = bool(ADSENSE_CLIENT_ID) and request.endpoint in ADS_ENDPOINTS
    return {'adsense_client': ADSENSE_CLIENT_ID, 'ads_on_page': on_page,
            'ad_slots': ADSENSE_SLOTS if on_page else {}, 'ga_measurement_id': GA_MEASUREMENT_ID}


@app.route('/ads.txt')
def ads_txt():
    """Authorised sellers file. Without it AdSense rejects every ad request from the site."""
    lines = []
    if ADSENSE_CLIENT_ID:
        lines.append(f"google.com, {ADSENSE_CLIENT_ID[3:]}, DIRECT, f08c47fec0942fa0")
    lines.extend(ADS_TXT_EXTRA)
    if not lines:
        abort(404)
    return Response("\n".join(lines) + "\n", mimetype='text/plain',
                    headers={'Cache-Control': 'public, max-age=3600'})


@app.route('/admin/storage')
def admin_storage():
    if not _is_admin():
        abort(403)
    return render_template("ADMIN_STORAGE_HTML_TEMPLATE", report=database_storage_report())


@app.route('/admin/storage/cleanup', methods=['POST'])
def admin_storage_cleanup():
    if not _is_admin():
        abort(403)
    try:
        result = prune_old_data()
        flash(f"Clean up done: removed {result['summaries']} old summaries and {result['view_counts']} old view counts.", 'success')
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Manual clean up failed: {e}", exc_info=True)
        flash("The clean up could not run. Check the logs.", 'danger')
    return redirect(url_for('admin_storage'))


@app.route('/admin/storage/backup.json')
def admin_storage_backup():
    """Full copy of every table as JSON. Free databases have no automatic backups."""
    from sqlalchemy import select
    if not _is_admin():
        abort(403)
    data = {'generated_at': datetime.now(timezone.utc).isoformat(), 'provider': DATABASE_PROVIDER, 'tables': {}}
    for table in db.metadata.sorted_tables:
        data['tables'][table.name] = [dict(row) for row in db.session.execute(select(table)).mappings()]
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return Response(json.dumps(data, default=str, ensure_ascii=False), mimetype='application/json',
                    headers={'Content-Disposition': f'attachment; filename=brieflyai-backup-{stamp}.json',
                             'Cache-Control': 'no-store'})


@app.route('/robots.txt')
def robots_txt():
    lines = [
        "User-agent: *",
        "Allow: /",
        # Keep crawlers out of personal and action endpoints.
        "Disallow: /profile",
        "Disallow: /login",
        "Disallow: /register",
        "Disallow: /logout",
        "Disallow: /toggle_bookmark/",
        "Disallow: /add_comment/",
        "Disallow: /vote_comment/",
        "Disallow: /admin/",
        "Disallow: /healthz",
        "",
        f"Sitemap: {url_for('sitemap', _external=True)}",
    ]
    return Response("\n".join(lines), mimetype='text/plain')


@app.route('/related/<article_hash_id>')
def related_articles(article_hash_id):
    """Related community articles, scored by title/description word overlap."""
    STOPWORDS = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'has', 'have', 'was', 'are',
        'not', 'but', 'its', 'his', 'her', 'they', 'their', 'you', 'who', 'what', 'why',
        'how', 'will', 'can', 'been', 'over', 'after', 'says', 'said', 'new', 'more',
    }

    def keywords(text):
        words = re.findall(r"[a-z0-9]{3,}", (text or '').lower())
        return {w for w in words if w not in STOPWORDS}

    try:
        current = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
        if not current:
            return jsonify({"success": True, "articles": []})

        target = keywords(current.title) | keywords(current.description)
        candidates = (CommunityArticle.query
                      .options(joinedload(CommunityArticle.author))
                      .filter(CommunityArticle.id != current.id)
                      .order_by(CommunityArticle.published_at.desc())
                      .limit(200).all())

        scored = []
        for art in candidates:
            overlap = len(target & (keywords(art.title) | keywords(art.description)))
            if overlap:
                scored.append((overlap, art))
        scored.sort(key=lambda pair: (pair[0], pair[1].published_at or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)

        results = [{
            "title": art.title,
            "url": url_for('article_detail', article_hash_id=art.article_hash_id),
            "source": art.author.name if art.author else art.source_name,
            "image_url": art.image_url,
        } for _, art in scored[:4]]
        return jsonify({"success": True, "articles": results})
    except Exception as e:
        app.logger.error(f"Related articles lookup failed for {article_hash_id}: {e}", exc_info=True)
        return jsonify({"success": True, "articles": []})


@app.route('/manifest.webmanifest')
def web_manifest():
    manifest = {
        "name": "BrieflyAI",
        "short_name": "BrieflyAI",
        "description": "Short AI summaries and key takeaways for India's top stories.",
        "start_url": "/news",
        "scope": "/",
        "display": "standalone",
        "background_color": "#F7F5EF",
        "theme_color": "#F7F5EF",
        "icons": [{
            "src": url_for('app_icon', size=size, v=2),
            "sizes": f"{size}x{size}",
            "type": "image/svg+xml",
            "purpose": "any",
        } for size in (192, 512)],
    }
    return Response(json.dumps(manifest), mimetype='application/manifest+json')


@app.route('/icon-<int:size>.svg')
def app_icon(size):
    """Self-contained SVG app icon, so the PWA needs no binary asset files."""
    size = 512 if size not in (192, 512) else size
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="{size}" height="{size}">'
        '<rect width="512" height="512" rx="112" fill="#16140F"/>'
        '<rect x="128" y="136" width="256" height="48" rx="24" fill="#F7F5EF"/>'
        '<rect x="128" y="232" width="176" height="48" rx="24" fill="#F7F5EF"/>'
        '<rect x="128" y="328" width="96" height="48" rx="24" fill="#6CC9BD"/>'
        '</svg>'
    )
    return Response(svg, mimetype='image/svg+xml',
                    headers={'Cache-Control': 'public, max-age=604800'})


@app.route('/offline')
def offline_page():
    return render_template("OFFLINE_TEMPLATE")


@app.route('/sw.js')
def service_worker():
    """
    Service worker: network-first for pages (news must stay fresh), cache-first for
    static CDN assets, and an offline fallback page. Served from the app root so its
    scope covers the whole site.
    """
    sw = """
const VERSION = 'brieflyai-v3';
// Only these hosts are cached. Ads, ad measurement and news images always go to the network.
const CACHEABLE_HOSTS = ['cdn.jsdelivr.net', 'fonts.googleapis.com', 'fonts.gstatic.com'];
const OFFLINE_URL = '/offline';
const PRECACHE = [OFFLINE_URL, '/manifest.webmanifest'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(VERSION).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  // Never cache authenticated or mutating endpoints.
  if (/^\\/(admin|healthz|ads\\.txt|login|logout|register|profile|add_comment|vote_comment|delete_comment|edit_comment|toggle_bookmark|post_article|report_article)/.test(url.pathname)) {
    return;
  }

  // Static assets from our allowed CDNs: cache-first, they are versioned URLs.
  if (url.origin !== self.location.origin) {
    if (!CACHEABLE_HOSTS.includes(url.hostname)) return;
    event.respondWith(
      caches.match(req).then((hit) => hit || fetch(req).then((res) => {
        if (res && res.status === 200) {
          const copy = res.clone();
          caches.open(VERSION).then((c) => c.put(req, copy));
        }
        return res;
      }).catch(() => hit))
    );
    return;
  }

  // Pages: network-first so news is always current, cache as a fallback.
  event.respondWith(
    fetch(req).then((res) => {
      if (res && res.status === 200 && res.type === 'basic') {
        const copy = res.clone();
        caches.open(VERSION).then((c) => c.put(req, copy));
      }
      return res;
    }).catch(() =>
      caches.match(req).then((hit) => hit || (req.mode === 'navigate' ? caches.match(OFFLINE_URL) : undefined))
    )
  );
});
"""
    return Response(sw, mimetype='application/javascript',
                    headers={'Cache-Control': 'no-cache'})


def _record_article_view(article_hash_id):
    """
    Increment an article's view counter. Analytics must never break page rendering,
    so every failure here is swallowed after logging.
    """
    if not article_hash_id:
        return
    try:
        stat = ArticleStat.query.filter_by(article_hash_id=article_hash_id).first()
        if stat:
            stat.view_count = (stat.view_count or 0) + 1
            stat.last_viewed_at = datetime.now(timezone.utc)
        else:
            db.session.add(ArticleStat(
                article_hash_id=article_hash_id, view_count=1,
                last_viewed_at=datetime.now(timezone.utc)))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.warning(f"Could not record view for {article_hash_id}: {e}")


def search_community_articles(query_str, limit=60):
    """
    Search community posts by title, description and body.

    Previously /search only queried NewsAPI, so community posts were effectively
    invisible -- you could not find a story through search right after posting it.
    """
    if not query_str:
        return []
    try:
        terms = [t for t in re.split(r"\s+", query_str.strip()) if len(t) >= 2][:6]
        if not terms:
            return []
        conditions = []
        for term in terms:
            like = f"%{term.lower()}%"
            conditions.append(db.or_(
                func.lower(CommunityArticle.title).like(like),
                func.lower(CommunityArticle.description).like(like),
                func.lower(CommunityArticle.full_text).like(like),
            ))
        return (CommunityArticle.query
                .options(joinedload(CommunityArticle.author))
                .filter(db.and_(*conditions))
                .order_by(CommunityArticle.published_at.desc())
                .limit(limit).all())
    except Exception as e:
        app.logger.error(f"Community search failed for {query_str!r}: {e}", exc_info=True)
        return []


@app.route('/trending')
def trending():
    """Most-viewed articles over the last week, as JSON for the homepage strip."""
    try:
        since = datetime.now(timezone.utc) - timedelta(days=7)
        stats = (ArticleStat.query
                 .filter(ArticleStat.last_viewed_at >= since)
                 .order_by(ArticleStat.view_count.desc())
                 .limit(20).all())
        results = []
        for stat in stats:
            art = CommunityArticle.query.filter_by(article_hash_id=stat.article_hash_id).first()
            if art:
                title, source = art.title, (art.author.name if art.author else art.source_name)
            else:
                cached = MASTER_ARTICLE_STORE.get(stat.article_hash_id)
                if not cached:
                    continue  # article aged out of the cache; skip rather than show a dead row
                title = cached.get('title')
                source = (cached.get('source') or {}).get('name', 'Unknown')
            results.append({
                "title": title,
                "source": source,
                "views": stat.view_count,
                "url": url_for('article_detail', article_hash_id=stat.article_hash_id),
            })
            if len(results) >= 5:
                break
        return jsonify({"success": True, "articles": results})
    except Exception as e:
        app.logger.error(f"Trending lookup failed: {e}", exc_info=True)
        return jsonify({"success": True, "articles": []})


@app.route('/account/export')
@login_required
def export_my_data():
    """Download everything this account holds, as JSON."""
    try:
        user = User.query.get(session['user_id'])
        if not user:
            abort(404)
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "account": {
                "name": user.name,
                "username": user.username,
                "created_at": user.created_at.isoformat() if user.created_at else None,
            },
            "articles": [{
                "title": a.title,
                "description": a.description,
                "full_text": a.full_text,
                "published_at": a.published_at.isoformat() if a.published_at else None,
                "url": url_for('article_detail', article_hash_id=a.article_hash_id, _external=True),
            } for a in user.articles],
            "comments": [{
                "content": c.content,
                "timestamp": c.timestamp.isoformat() if c.timestamp else None,
            } for c in user.comments],
            "testimonials": [{
                "name": t.name,
                "attribution": t.attribution,
                "quote": t.quote,
                "status": t.status,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            } for t in user.testimonials],
            "bookmarks": [{
                "title": b.title_cache,
                "source": b.source_name_cache,
                "bookmarked_at": b.bookmarked_at.isoformat() if b.bookmarked_at else None,
                "url": url_for('article_detail', article_hash_id=b.article_hash_id, _external=True),
            } for b in user.bookmarks],
        }
        filename = f"brieflyai-export-{user.username}-{datetime.now(timezone.utc).date()}.json"
        return Response(
            json.dumps(payload, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'})
    except Exception as e:
        app.logger.error(f"Data export failed: {e}", exc_info=True)
        flash("Could not build your export right now. Please try again.", "danger")
        return redirect(url_for('profile'))


@app.route('/account/delete', methods=['POST'])
@login_required
@rate_limit(3, 3600, scope='delete_account')
def delete_account():
    """
    Permanently delete the account. Requires the password again, so a stolen session
    alone can't destroy someone's data.
    """
    user = User.query.get(session['user_id'])
    if not user:
        session.clear()
        return redirect(url_for('index'))

    password = request.form.get('password', '')
    if not password or not check_password_hash(user.password_hash, password):
        flash("Password incorrect. Your account has not been deleted.", "danger")
        return redirect(url_for('profile'))

    username = user.username
    try:
        # Comments/articles/bookmarks/votes cascade via the relationships.
        db.session.delete(user)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Account deletion failed for {username}: {e}", exc_info=True)
        flash("Could not delete your account right now. Please try again.", "danger")
        return redirect(url_for('profile'))

    session.clear()
    app.logger.info(f"Account deleted: {username}")
    flash("Your account and all associated data have been permanently deleted.", "info")
    return redirect(url_for('index'))


def load_persisted_analysis(article_hash_id):
    """Return previously generated AI analysis for an article, or None."""
    try:
        row = ArticleAnalysis.query.filter_by(article_hash_id=article_hash_id).first()
        if not row or not row.groq_summary:
            return None
        takeaways = None
        if row.groq_takeaways:
            try:
                takeaways = json.loads(row.groq_takeaways)
            except (ValueError, TypeError):
                takeaways = None
        return {"groq_summary": row.groq_summary, "groq_takeaways": takeaways, "error": None}
    except Exception as e:
        app.logger.warning(f"Could not read persisted analysis for {article_hash_id}: {e}")
        return None


def save_persisted_analysis(article_hash_id, analysis):
    """Store AI analysis so the same article is never summarised twice."""
    if not analysis or analysis.get("error") or not analysis.get("groq_summary"):
        return
    try:
        takeaways = analysis.get("groq_takeaways")
        takeaways_json = json.dumps(takeaways) if isinstance(takeaways, list) else None
        row = ArticleAnalysis.query.filter_by(article_hash_id=article_hash_id).first()
        if row:
            row.groq_summary = analysis.get("groq_summary")
            row.groq_takeaways = takeaways_json
        else:
            db.session.add(ArticleAnalysis(
                article_hash_id=article_hash_id,
                groq_summary=analysis.get("groq_summary"),
                groq_takeaways=takeaways_json))
        db.session.commit()
        app.logger.info(f"Persisted AI analysis for {article_hash_id} (saves a future API call).")
    except Exception as e:
        db.session.rollback()
        app.logger.warning(f"Could not persist analysis for {article_hash_id}: {e}")


# --- RSS ingestion -----------------------------------------------------------
# NewsAPI's free tier delays articles ~24h and forbids commercial use. Publisher RSS
# feeds are free, real-time, unlimited and carry no such restriction, so they are used
# as the primary source with NewsAPI kept as a fallback.
DEFAULT_RSS_FEEDS = [
    "https://www.thehindu.com/news/national/feeder/default.rss",
    "https://feeds.feedburner.com/ndtvnews-top-stories",
    "https://indianexpress.com/section/india/feed/",
    "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
]


def _rss_feed_urls():
    configured = os.environ.get('RSS_FEEDS', '').strip()
    if configured:
        return [u.strip() for u in configured.split(',') if u.strip()]
    return DEFAULT_RSS_FEEDS


def _rss_text(element, *tag_names):
    for tag in tag_names:
        found = element.find(tag)
        if found is not None and found.text:
            return found.text.strip()
    return None


def _parse_rss_datetime(value):
    if not value:
        return datetime.now(timezone.utc)
    for fmt in ('%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S%z'):
        try:
            parsed = datetime.strptime(value.strip(), fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    try:
        parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _extract_rss_image(item):
    """RSS has no single image convention; try the common ones in order."""
    for tag, attr in (('{http://search.yahoo.com/mrss/}content', 'url'),
                      ('{http://search.yahoo.com/mrss/}thumbnail', 'url'),
                      ('enclosure', 'url')):
        node = item.find(tag)
        if node is not None:
            url = node.get(attr)
            if url and url.startswith('http'):
                return url
    description = _rss_text(item, 'description') or ''
    match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', description)
    return match.group(1) if match else None


@simple_cache(expiry_seconds_default=900)
def fetch_news_from_rss(limit_per_feed=25):
    """
    Pull articles straight from publisher RSS feeds.

    Free, real-time (no 24h delay) and not rate limited, which is why this is the
    primary source. Parsed with the standard library, so no new dependency.
    """
    import xml.etree.ElementTree as ET

    articles, seen_urls = [], set()
    for feed_url in _rss_feed_urls():
        try:
            resp = requests.get(feed_url, timeout=int(os.environ.get('RSS_TIMEOUT_SECONDS', '10')),
                                headers={'User-Agent': 'BrieflyAI/1.0 (+https://brieflyai.example)'})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            # One bad feed must never take down the homepage.
            app.logger.warning(f"RSS feed failed ({feed_url}): {e}")
            continue

        channel_title = None
        channel = root.find('channel')
        if channel is not None:
            channel_title = _rss_text(channel, 'title')
        source_name = (channel_title or urllib.parse.urlparse(feed_url).netloc or 'RSS').strip()

        items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
        for item in items[:limit_per_feed]:
            link = _rss_text(item, 'link', '{http://www.w3.org/2005/Atom}id')
            title = _rss_text(item, 'title', '{http://www.w3.org/2005/Atom}title')
            if not link or not title or link in seen_urls:
                continue
            raw_description = _rss_text(item, 'description', '{http://www.w3.org/2005/Atom}summary') or ''
            # Feed descriptions often contain markup; strip it to plain text.
            description = re.sub(r'<[^>]+>', '', raw_description).strip()
            description = re.sub(r'\s+', ' ', description)
            if not description:
                description = title
            seen_urls.add(link)
            published_dt = _parse_rss_datetime(_rss_text(item, 'pubDate', 'published', 'updated'))
            article_id = generate_article_id(link)
            standardized = {
                'id': article_id, 'title': title, 'description': description[:500],
                'url': link, 'urlToImage': _extract_rss_image(item),
                'publishedAt': published_dt.isoformat(),
                'source': {'name': source_name}, 'is_community_article': False,
                'groq_summary': None, 'groq_takeaways': None,
            }
            MASTER_ARTICLE_STORE[article_id] = standardized
            articles.append(standardized)

    articles.sort(key=lambda a: a.get('publishedAt', ''), reverse=True)
    app.logger.info(f"RSS ingestion returned {len(articles)} articles from {len(_rss_feed_urls())} feeds.")
    return articles


@app.route('/healthz')
def health_check():
    """Liveness/readiness probe for the host platform."""
    status = {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
    code = 200
    try:
        db.session.execute(db.text('SELECT 1'))
        status["database"] = "ok"
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Health check DB failure: {e}")
        status["status"] = "degraded"
        status["database"] = "error"
        code = 503
    status["ai"] = "ok" if groq_client else "disabled"
    status["news_api"] = "ok" if newsapi else "disabled"
    status["caches"] = {
        "article_store": MASTER_ARTICLE_STORE.stats(),
        "api_cache": API_CACHE.stats(),
    }
    # Also called by the keep alive job, which keeps a free hosted database from idling.
    return jsonify(status), code, {'Cache-Control': 'no-store'}

# ==============================================================================
# --- 7. HTML Templates (Stored in memory) ---
# ==============================================================================
BASE_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {% set _theme_pref = request.cookies.get('darkMode') %}
    {% if _theme_pref == 'enabled' %}<meta name="theme-color" content="#131209">{% elif _theme_pref == 'disabled' %}<meta name="theme-color" content="#F7F5EF">{% else %}<meta name="theme-color" content="#F7F5EF" media="(prefers-color-scheme: light)"><meta name="theme-color" content="#131209" media="(prefers-color-scheme: dark)">{% endif %}
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <link rel="manifest" href="{{ url_for('web_manifest') }}">
    <link rel="icon" href="{{ url_for('app_icon', size=192, v=2) }}" type="image/svg+xml">
    <link rel="alternate" type="application/rss+xml" title="BrieflyAI - Community Stories" href="{{ url_for('rss_feed') }}">
    {% if adsense_client %}<meta name="google-adsense-account" content="{{ adsense_client }}">{% endif %}
    {% if ads_on_page %}<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={{ adsense_client }}" crossorigin="anonymous"></script>{% endif %}
    <title>{% block title %}BrieflyAI{% endblock %}</title>

    {# --- SEO / social sharing. Pages override the inner blocks. --- #}
    {% block meta %}
    <meta name="description" content="{% block meta_description %}AI-summarized, India-centric news. Get the key facts in seconds.{% endblock %}">
    <link rel="canonical" href="{% block canonical_url %}{{ request.base_url }}{% endblock %}">
    <meta property="og:site_name" content="BrieflyAI">
    <meta property="og:type" content="{% block og_type %}website{% endblock %}">
    <meta property="og:title" content="{% block og_title %}BrieflyAI{% endblock %}">
    <meta property="og:description" content="{% block og_description %}AI-summarized, India-centric news. Get the key facts in seconds.{% endblock %}">
    <meta property="og:url" content="{% block og_url %}{{ request.base_url }}{% endblock %}">
    {% block og_image %}{% endblock %}
    <meta name="twitter:card" content="{% block twitter_card %}summary_large_image{% endblock %}">
    <meta name="twitter:title" content="{{ self.og_title() }}">
    <meta name="twitter:description" content="{{ self.og_description() }}">
    {% endblock %}

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>

    {# One typeface for the whole site (Geist 400, 500, 600, the same request as the landing page), upright only. #}
    <link rel="preload" as="style" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&display=swap">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&display=swap">

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

    {# Phosphor icons (regular + fill). Decorative only, so they load without blocking first paint. #}
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.2/src/regular/style.css" media="print" onload="this.media='all';this.onload=null;">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.2/src/fill/style.css" media="print" onload="this.media='all';this.onload=null;">
    <noscript><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.2/src/regular/style.css"><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@phosphor-icons/web@2.1.2/src/fill/style.css"></noscript>

    <style>
        /* ==========================================================================
           DESIGN TOKENS
           ========================================================================== */
        :root {
            /* The landing page's palette: warm paper, ink, one teal accent. */
            --primary-color: #12645E; --primary-light: #12645E; --primary-dark: #0D4C47;
            --secondary-color: #12645E; --secondary-light: #12645E; --accent-color: #12645E;
            --ink: #16140F; --ink-hover: #2E2B23; --on-ink: #F7F5EF;
            --text-color: #16140F; --text-muted-color: #5A564C; --text-subtle-color: #777266;
            --light-bg: #F7F5EF; --card-bg: #FBFAF6; --card-border-color: #DEDACF;
            --surface-color: #EFECE4; --surface-muted: #E6E2D8;
            --footer-bg: #F7F5EF; --footer-text: #5A564C; --footer-link-hover: #16140F;
            --primary-color-rgb: 18, 100, 94; --secondary-color-rgb: 18, 100, 94; --accent-color-rgb: 18, 100, 94; --text-muted-color-rgb: 90, 86, 76;
            --card-bg-rgb: 251, 250, 246;
            --bookmark-active-color: var(--accent-color);
            --logo-gist: #6CC9BD;

            --glass-bg: rgba(247, 245, 239, 0.72);
            --glass-border: #DEDACF;
            --glass-blur: 16px;
            --panel-bg: rgba(247, 245, 239, 0.94);
            --surface-gradient: none;
            --photo-filter: none;
            --photo-blend: #FFFFFF;
            --grain: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0.08 0 0 0 0 0.07 0 0 0 0 0.04 0 0 0 0.9 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");

            --shadow-sm: 0 1px 2px 0 rgba(56, 44, 20, 0.06);
            --shadow-md: 0 16px 32px -24px rgba(56, 44, 20, 0.35);
            --shadow-lg: 0 32px 64px -32px rgba(56, 44, 20, 0.35);
            --shadow-glow: none;

            /* Tailwind radii: md 6, lg 8, xl 12, 2xl 16, 3xl 24. */
            --border-radius-xs: 6px; --border-radius-sm: 8px; --border-radius-md: 12px;
            --border-radius-lg: 16px; --border-radius-xl: 24px; --border-radius-pill: 9999px;

            --font-display: 'Geist', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
            --font-sans: 'Geist', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
            --font-body: var(--font-sans);
            --font-mono: ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace;

            /* Tailwind type scale only: 12 14 16 18 20 24 30 36 48 60 */
            --text-xs: 0.75rem; --text-sm: 0.875rem; --text-base: 1rem; --text-md: 1rem;
            --text-lg: 1.125rem; --text-xl: 1.25rem; --text-2xl: 1.5rem;
            --text-3xl: 1.875rem; --text-4xl: 2.25rem;

            --space-1: 0.25rem; --space-2: 0.5rem; --space-3: 0.75rem; --space-4: 1rem; --space-5: 1.5rem; --space-6: 2rem; --space-7: 3rem;

            /* One motion curve everywhere, as on the landing page. */
            --ease-standard: cubic-bezier(0.22, 1, 0.36, 1);
            --ease-premium: cubic-bezier(0.22, 1, 0.36, 1);
            --ease-spring: cubic-bezier(0.22, 1, 0.36, 1);
            --duration-fast: 200ms; --duration-base: 300ms; --duration-slow: 500ms;

            --skeleton-base: #E6E2D8; --skeleton-sheen: #EFECE4;
        }
        @media (min-width: 992px) { :root { --text-3xl: 3rem; --text-4xl: 3.75rem; } }
        em, i, cite, blockquote, address { font-style: normal; }
        .ph, .ph-fill { line-height: 1; vertical-align: -0.125em; }
        .ph-spin { display: inline-block; animation: ph-spin 900ms cubic-bezier(0.22, 1, 0.36, 1) infinite; }
        @keyframes ph-spin { to { transform: rotate(360deg); } }
        /* Bootstrap's muted text keeps its light theme grey unless told otherwise. */
        body.dark-mode { --bs-secondary-color: var(--text-muted-color); --bs-secondary-color-rgb: var(--text-muted-color-rgb); }
        .testimonial-form-wrap { max-width: 720px; margin-inline: auto; }
        .testimonial-mine { padding: 0.75rem 1rem; border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); background: var(--light-bg); }
        .testimonial-status { display: inline-flex; align-items: center; gap: 0.25rem; font-size: 0.75rem; font-weight: 600; color: var(--text-muted-color); }
        .testimonial-status-approved { color: var(--secondary-color); }
        .was-validated .form-control:invalid ~ .d-flex .d-block-when-invalid { display: block; }
        body.dark-mode {
            --primary-color: #6CC9BD; --primary-light: #6CC9BD; --primary-dark: #9ADBD2;
            --secondary-color: #6CC9BD; --secondary-light: #6CC9BD; --accent-color: #6CC9BD;
            --ink: #F2EFE6; --ink-hover: #DAD6CB; --on-ink: #131209;
            --text-color: #F2EFE6; --text-muted-color: #ABA697; --text-subtle-color: #8C887B;
            --light-bg: #131209; --card-bg: #181818; --card-border-color: #313131;
            --surface-color: #1F1F1F; --surface-muted: #272727;
            --footer-bg: #131209; --footer-text: #ABA697; --footer-link-hover: #F2EFE6;
            --primary-color-rgb: 108, 201, 189; --secondary-color-rgb: 108, 201, 189; --accent-color-rgb: 108, 201, 189; --text-muted-color-rgb: 171, 166, 151;
            --card-bg-rgb: 24, 24, 24;
            --glass-bg: rgba(19, 18, 9, 0.66);
            --glass-border: #313131;
            --panel-bg: rgba(19, 18, 9, 0.92);
            --photo-filter: brightness(0.5) saturate(0.85);
            --photo-blend: #808080;
            --bookmark-active-color: var(--accent-color);
            --logo-gist: #12645E;
            --shadow-sm: none;
            --shadow-md: 0 16px 32px -24px rgba(0, 0, 0, 0.8);
            --shadow-lg: 0 32px 64px -32px rgba(0, 0, 0, 0.8);
            --skeleton-base: #1F1F1F; --skeleton-sheen: #272727;
            color-scheme: dark;
        }

        @media (prefers-reduced-motion: reduce) {
            /* Delays must be zeroed too, not just durations: a staggered animation-delay with
               fill-mode "both" holds the element at its 0-opacity start state for the delay. */
            *, *::before, *::after {
                animation-duration: 0.001ms !important;
                animation-delay: 0s !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.001ms !important;
                transition-delay: 0s !important;
                scroll-behavior: auto !important;
            }
            [data-reveal] { opacity: 1 !important; transform: none !important; }
        }

        /* ==========================================================================
           BASE
           ========================================================================== */
        *, *::before, *::after { box-sizing: border-box; }
        /* overflow-x must live on html, not body: setting it on body makes body its own
           scroll container (overflow-y computes to auto), which stops window.scrollY from
           tracking the page and silently breaks the progress bar, parallax and nav state. */
        html { scroll-behavior: smooth; -webkit-text-size-adjust: 100%; overflow-x: hidden; }
        ::selection { background: rgba(var(--primary-color-rgb), 0.22); }
        body {
            padding-top: 96px; margin: 0; font-family: var(--font-sans); font-size: 1rem; line-height: 1.6;
            color: var(--text-color); background-color: var(--light-bg);
            display: flex; flex-direction: column; min-height: 100vh;
            transition: background-color 0.35s var(--ease-standard), color 0.35s var(--ease-standard);
            -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
        }
        .main-content { flex-grow: 1; }
        p { max-width: 75ch; }
        a { color: var(--primary-color); text-underline-offset: 0.2em; }
        a:hover { color: var(--primary-dark); }
        img { max-width: 100%; }

        h1, h2, h3, h4, h5 { font-family: var(--font-sans); font-weight: 600; letter-spacing: -0.022em; line-height: 1.22; }

        /* Editorial serif for the big moments only -- headlines, page titles, pull quotes. */
        .display-serif,
        .article-title-main,
        .featured-story-content h2,
        .page-header-static h1,
        .synthesis-text,
        .auth-header h1,
        .profile-header-card h1,
        .state-card-title {
            font-family: var(--font-display);
            font-weight: 600;
            letter-spacing: -0.02em;
            line-height: 1.12;
        }

        .eyebrow { font-family: var(--font-sans); font-size: var(--text-xs); font-weight: 600; color: var(--primary-color); }
        .section-heading { font-size: var(--text-2xl); font-weight: 600; margin: 0; letter-spacing: -0.025em; }

        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--light-bg); }
        ::-webkit-scrollbar-thumb { background: var(--card-border-color); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted-color); }

        a:focus-visible, button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible,
        .btn:focus-visible, .nav-link:focus-visible, .page-link:focus-visible, .dropdown-item:focus-visible,
        [tabindex]:focus-visible, summary:focus-visible {
            outline: 2.5px solid var(--primary-color); outline-offset: 2px; border-radius: var(--border-radius-xs);
        }

        /* ==========================================================================
           ANIMATION UTILITIES
           ========================================================================== */
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes skeletonShimmer { 0% { background-position: 150% 0; } 100% { background-position: -50% 0; } }
        @keyframes bookmarkPop { 0% { transform: scale(1); } 35% { transform: scale(1.34); } 60% { transform: scale(0.92); } 100% { transform: scale(1); } }
        .animate-fade-in { animation: fadeInUp 0.55s var(--ease-premium) both; }
        .bookmark-btn.is-popping { animation: bookmarkPop 0.45s var(--ease-spring); }

        /* Scroll-triggered reveal. The attribute is added by JS, so no-JS users never see hidden content. */
        [data-reveal] { opacity: 0; transform: translateY(22px); transition: opacity 0.7s var(--ease-premium), transform 0.7s var(--ease-premium); transition-delay: var(--reveal-delay, 0ms); will-change: opacity, transform; }
        [data-reveal].is-revealed { opacity: 1; transform: none; }

        /* ==========================================================================
           BUTTONS
           ========================================================================== */
        .btn { border-radius: var(--border-radius-sm); font-weight: 600; font-size: 0.875rem; letter-spacing: -0.005em; transition: background-color var(--duration-base) var(--ease-standard), border-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring), color var(--duration-base) var(--ease-standard); }
        .btn:hover { transform: translateY(-1px); }
        .btn:active { transform: translateY(0) scale(0.975); }
        .btn-primary { --bs-btn-bg: var(--primary-color); --bs-btn-border-color: var(--primary-color); --bs-btn-hover-bg: var(--primary-dark); --bs-btn-hover-border-color: var(--primary-dark); --bs-btn-active-bg: var(--primary-dark); --bs-btn-active-border-color: var(--primary-dark); --bs-btn-focus-shadow-rgb: var(--primary-color-rgb); }
        .btn-primary:hover, .btn-primary:focus { box-shadow: var(--shadow-glow); }
        .btn-primary-modal { background-image: none; border: none; color: #fff; border-radius: var(--border-radius-sm); font-weight: 600; padding: 0.65rem 1.5rem; transition: box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring), filter var(--duration-base) var(--ease-standard); }
        .btn-primary-modal:hover { color: #fff; filter: brightness(1.07); box-shadow: var(--shadow-glow); transform: translateY(-1px); }
        .btn-primary-modal:active { transform: translateY(0) scale(0.975); }
        .btn-outline-primary { --bs-btn-color: var(--primary-color); --bs-btn-border-color: var(--primary-color); --bs-btn-hover-bg: var(--primary-color); --bs-btn-hover-border-color: var(--primary-color); --bs-btn-active-bg: var(--primary-color); --bs-btn-active-border-color: var(--primary-color); }
        .btn-outline-secondary { --bs-btn-color: var(--text-muted-color); --bs-btn-border-color: var(--card-border-color); --bs-btn-hover-bg: var(--text-color); --bs-btn-hover-border-color: var(--text-color); }
        .btn-danger { --bs-btn-bg: #DC2626; --bs-btn-border-color: #DC2626; --bs-btn-hover-bg: #B91C1C; --bs-btn-hover-border-color: #B91C1C; }
        .btn .spinner-border { vertical-align: -0.15em; }
        /* Arrow nudge on hover. */
        .btn i.ph-arrow-right, .btn i.ph-caret-right, .read-more i, .read-more-btn i { transition: transform var(--duration-base) var(--ease-spring); }
        .btn:hover i.ph-arrow-right, .btn:hover i.ph-caret-right, .read-more:hover i, .read-more-btn:hover i { transform: translateX(4px); }

        /* ==========================================================================
           FORMS
           ========================================================================== */
        .form-control, .form-select { border-radius: var(--border-radius-sm); border-color: var(--card-border-color); background-color: var(--card-bg); color: var(--text-color); transition: border-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard); }
        .form-control:focus, .form-select:focus { border-color: var(--primary-color); box-shadow: 0 0 0 3.5px rgba(var(--primary-color-rgb), 0.16); background-color: var(--card-bg); color: var(--text-color); }
        .form-control::placeholder { color: var(--text-muted-color); opacity: 0.72; }
        .text-primary { color: var(--primary-color) !important; }

        /* ==========================================================================
           TOASTS
           ========================================================================== */
        #alert-placeholder { position: fixed; top: 98px; left: 50%; transform: translateX(-50%); z-index: 2050; display: flex; flex-direction: column; align-items: stretch; gap: 0.6rem; width: min(92vw, 420px); pointer-events: none; }
        #alert-placeholder .alert { pointer-events: auto; position: relative; margin: 0; width: 100%; text-align: left; box-shadow: var(--shadow-lg); border-radius: var(--border-radius-md); border: 1px solid var(--card-border-color); background-color: var(--card-bg); color: var(--text-color); display: flex; align-items: flex-start; gap: 0.65rem; padding: 0.9rem 2.6rem 0.9rem 1rem; animation: toastIn var(--duration-slow) var(--ease-spring) both; }
        @keyframes toastIn { from { opacity: 0; transform: translateY(-16px) scale(0.96); } to { opacity: 1; transform: translateY(0) scale(1); } }
        #alert-placeholder .alert .alert-icon { font-size: 1rem; margin-top: 0.15rem; flex-shrink: 0; }
        #alert-placeholder .alert.alert-success .alert-icon { color: var(--secondary-color); }
        #alert-placeholder .alert.alert-danger .alert-icon { color: #DC2626; }
        #alert-placeholder .alert.alert-warning .alert-icon { color: #D97706; }
        #alert-placeholder .alert.alert-info .alert-icon { color: var(--primary-color); }
        #alert-placeholder .alert .btn-close { position: absolute; top: 0.85rem; right: 0.85rem; font-size: 0.75rem; }
        body.dark-mode #alert-placeholder .alert .btn-close { filter: invert(1) grayscale(100%) brightness(200%); }

        /* ==========================================================================
           NAVBAR: the landing page's floating frosted bar
           ========================================================================== */
        .navbar-main {
            position: fixed; top: 16px; left: 0; right: 0; z-index: 1040;
            width: calc(100% - 24px); margin-inline: auto; padding: 8px;
            border: 1px solid var(--glass-border); border-radius: var(--border-radius-lg);
            background: var(--glass-bg);
            -webkit-backdrop-filter: blur(16px) saturate(140%);
            backdrop-filter: blur(16px) saturate(140%);
            transition: box-shadow var(--duration-base) var(--ease-standard), background-color var(--duration-base) var(--ease-standard);
        }
        @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
            .navbar-main { background: var(--light-bg); }
        }
        .navbar-main.is-scrolled { box-shadow: var(--shadow-md); }
        /* The bar lines up with the page content at every Bootstrap container width. */
        @media (min-width: 576px) { .navbar-main { max-width: 516px; } }
        @media (min-width: 768px) { .navbar-main { max-width: 696px; } }
        @media (min-width: 992px) { .navbar-main { max-width: 936px; } }
        @media (min-width: 1200px) { .navbar-main { max-width: 1116px; } }
        @media (min-width: 1400px) { .container { max-width: 1232px; } .navbar-main { max-width: 1208px; } }
        .navbar-content-wrapper { display: flex; align-items: center; justify-content: space-between; gap: 8px; width: 100%; }
        .navbar-left { flex-shrink: 0; }
        .navbar-center { flex-grow: 1; min-width: 150px; max-width: 480px; }
        .navbar-right { flex-shrink: 0; }
        .navbar-brand-custom {
            display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: var(--border-radius-sm);
            color: var(--text-color) !important; font-size: 1rem; line-height: 1.5rem; font-weight: 500; letter-spacing: -0.01em;
            text-decoration: none !important; transition: background-color var(--duration-base) var(--ease-standard);
        }
        .navbar-brand-custom:hover { background: var(--surface-muted); }

        .search-container { position: relative; width: 100%; }
        .navbar-search {
            width: 100%; height: 40px; padding: 8px 40px; border-radius: var(--border-radius-sm);
            border: 1px solid var(--card-border-color); background-color: var(--card-bg); color: var(--text-color);
            font-size: 0.875rem; line-height: 1.25rem;
            transition: border-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard);
        }
        .navbar-search::placeholder { color: var(--text-subtle-color); opacity: 1; }
        .navbar-search:focus { background-color: var(--card-bg); border-color: var(--text-color); box-shadow: 0 0 0 3px rgba(var(--primary-color-rgb), 0.16); outline: none; color: var(--text-color); }
        .search-icon { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); font-size: 1rem; color: var(--text-subtle-color); pointer-events: none; transition: color var(--duration-base) var(--ease-standard); }
        .search-container:focus-within .search-icon { color: var(--text-color); }
        .search-clear-btn {
            position: absolute; right: 6px; top: 50%; transform: translateY(-50%); width: 28px; height: 28px;
            display: none; align-items: center; justify-content: center; border: none; border-radius: var(--border-radius-xs);
            background: none; color: var(--text-muted-color); font-size: 0.875rem;
            transition: color var(--duration-base) var(--ease-standard), background-color var(--duration-base) var(--ease-standard);
        }
        .search-clear-btn:hover { color: var(--text-color); background: var(--surface-muted); }
        .search-container.has-value .search-clear-btn { display: flex; }

        .header-controls { display: flex; gap: 4px; align-items: center; }
        .header-btn {
            display: inline-flex; align-items: center; gap: 8px; height: 40px; padding: 8px 12px;
            border: 1px solid transparent; border-radius: var(--border-radius-sm); background: none; color: var(--text-color);
            font-size: 0.875rem; line-height: 1.25rem; font-weight: 500;
            transition: background-color var(--duration-base) var(--ease-standard), transform var(--duration-slow) var(--ease-standard);
        }
        .header-btn:hover { background: var(--surface-muted); }
        .header-btn:active { transform: scale(0.98); }
        .header-btn i { font-size: 1.125rem; }

        /* ==========================================================================
           OFFCANVAS: a paper panel, like the landing page's menu
           ========================================================================== */
        .offcanvas { background-color: var(--light-bg); color: var(--text-color); z-index: 1045; border-left: 1px solid var(--card-border-color) !important; width: min(360px, 88vw); }
        .offcanvas-header { border-bottom: 1px solid var(--card-border-color); padding: 16px 24px; }
        .offcanvas-title { display: inline-flex; align-items: center; gap: 8px; margin: 0; font-size: 1rem; line-height: 1.5rem; font-weight: 500; }
        .offcanvas-body { padding: 24px; }
        body.dark-mode .offcanvas-header .btn-close { filter: invert(1) grayscale(100%) brightness(200%); }
        .sidebar-section { margin-bottom: 24px; }
        .sidebar-heading { margin-bottom: 8px; font-size: 0.75rem; line-height: 1rem; font-weight: 500; color: var(--text-subtle-color); }
        .sidebar-btn {
            display: flex; align-items: center; width: 100%; padding: 12px 16px; text-align: left;
            border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md);
            background-color: var(--card-bg); color: var(--text-color); text-decoration: none;
            font-size: 0.875rem; line-height: 1.25rem; font-weight: 500;
            transition: background-color var(--duration-base) var(--ease-standard);
        }
        .sidebar-btn:hover { background-color: var(--surface-muted); color: var(--text-color); }
        .sidebar-avatar { width: 32px; height: 32px; border-radius: var(--border-radius-sm); background: var(--accent-color); color: var(--light-bg); display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.875rem; flex-shrink: 0; }
        .offcanvas .dropdown-toggle::after { margin-left: auto; color: var(--text-muted-color); }
        .offcanvas .dropdown-menu { width: 100%; }
        .offcanvas .nav-link {
            display: flex; align-items: center; padding: 8px 12px; border-radius: var(--border-radius-sm);
            color: var(--text-muted-color) !important; font-size: 0.875rem; line-height: 1.25rem; font-weight: 500;
            transition: background-color var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard);
        }
        .offcanvas .nav-link.active { background: var(--surface-muted); color: var(--text-color) !important; }
        .offcanvas .nav-link:not(.active):hover { background: var(--surface-muted); color: var(--text-color) !important; }
        .offcanvas hr { border-color: var(--card-border-color); opacity: 1; }

        /* ==========================================================================
           TABS
           ========================================================================== */
        .nav-tabs { border-bottom: 1px solid var(--card-border-color); gap: 0.4rem; }
        .nav-tabs .nav-link { border: none; background: transparent; color: var(--text-muted-color); font-weight: 600; font-size: 0.75rem; padding: 0.9rem 0.6rem; border-radius: 0; position: relative; transition: color var(--duration-base) var(--ease-standard); }
        .nav-tabs .nav-link::after { content: ''; position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; background-image: none; transform: scaleX(0); transition: transform var(--duration-base) var(--ease-premium); }
        .nav-tabs .nav-link.active { color: var(--text-color); background: transparent; }
        .nav-tabs .nav-link.active::after { transform: scaleX(1); }
        .nav-tabs .nav-link:not(.active):hover { color: var(--text-color); }
        .nav-tabs .nav-link:not(.active):hover::after { transform: scaleX(0.35); }

        /* ==========================================================================
           CARDS
           ========================================================================== */
        .article-card, .article-full-content-wrapper { background: var(--card-bg); border-radius: var(--border-radius-lg); border: 1px solid var(--card-border-color); box-shadow: var(--shadow-sm); transition: transform var(--duration-base) var(--ease-premium), box-shadow var(--duration-base) var(--ease-premium), border-color var(--duration-base) var(--ease-standard); }
        .article-card { position: relative; overflow: hidden; }
        .article-card::after { content: ''; position: absolute; inset: 0; border-radius: inherit; background: none; opacity: 0; transition: opacity var(--duration-base) var(--ease-standard); pointer-events: none; }
        .article-card:hover { transform: translateY(-6px); box-shadow: var(--shadow-lg); border-color: rgba(var(--primary-color-rgb), 0.35); }
        .article-card:hover::after { opacity: 1; }

        .article-image-container { height: 200px; overflow: hidden; position: relative; border-top-left-radius: var(--border-radius-lg); border-top-right-radius: var(--border-radius-lg); background: var(--skeleton-base); background-image: none; background-size: 200% 100%; animation: skeletonShimmer 1.5s ease-in-out infinite; }
        .article-image-container.is-loaded, .article-image-container.img-fallback { animation: none; background-image: none; }
        .article-image-container.img-fallback { background: var(--light-bg); }
        .article-image { width: 100%; height: 100%; object-fit: cover; opacity: 0; transition: transform 0.6s var(--ease-premium), opacity var(--duration-slow) var(--ease-standard); }
        .article-image.is-loaded { opacity: 1; }
        .img-fallback .article-image { display: none; }
        .article-card:hover .article-image { transform: scale(1.06); }
        .img-fallback-icon { display: none; position: absolute; inset: 0; align-items: center; justify-content: center; font-size: 1.5rem; color: var(--card-border-color); }
        .img-fallback .img-fallback-icon { display: flex; }

        .article-body { padding: 1.35rem 1.4rem 1.4rem; flex-grow: 1; display: flex; flex-direction: column; position: relative; z-index: 1; }
        .article-title { font-family: var(--font-sans); font-weight: 600; line-height: 1.27; margin-bottom: 0.55rem; font-size: var(--text-lg); letter-spacing: -0.025em; }
        .article-title a { color: var(--text-color); text-decoration: none; transition: color var(--duration-fast) var(--ease-standard); }
        .article-card:hover .article-title a { color: var(--primary-color) !important; }
        .article-meta { display: flex; align-items: center; margin-bottom: 0.85rem; flex-wrap: wrap; gap: 0.35rem 0.45rem; }
        .meta-item { display: flex; align-items: center; font-family: var(--font-sans); font-size: 0.75rem; color: var(--text-muted-color); background: var(--light-bg); border: 1px solid var(--card-border-color); padding: 0.22rem 0.6rem; border-radius: var(--border-radius-pill); font-weight: 600; }
        .meta-item i { font-size: 0.75rem; margin-right: 0.35rem; color: var(--secondary-color); }
        .article-description { color: var(--text-muted-color); margin-bottom: 1.15rem; font-size: 0.875rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .read-more { margin-top: auto; background: var(--text-color); color: var(--card-bg) !important; border: none; padding: 0.62rem 0; border-radius: var(--border-radius-sm); font-weight: 600; font-size: 0.75rem; transition: background-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); width: 100%; text-align: center; text-decoration: none; display: inline-block; }
        .read-more:hover { background-image: none; color: #fff !important; box-shadow: var(--shadow-glow); }
        .read-more:active { transform: scale(0.98); }

        /* Long card grids below the fold: let the browser skip offscreen layout work. */
        .tab-pane .row.g-4 { content-visibility: auto; contain-intrinsic-size: 1px 900px; }

        /* ==========================================================================
           PAGINATION
           ========================================================================== */
        .pagination { flex-wrap: wrap; }
        .page-item .page-link { border-radius: var(--border-radius-sm); width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; color: var(--text-muted-color); background-color: var(--card-bg); border: 1px solid var(--card-border-color); font-weight: 600; transition: color var(--duration-fast) var(--ease-standard), border-color var(--duration-fast) var(--ease-standard), background-color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); font-size: 0.75rem; margin: 0 0.2rem; }
        .page-item .page-link:hover { border-color: var(--primary-color); color: var(--primary-color); transform: translateY(-2px); }
        .page-item.active .page-link { background-image: none; border-color: transparent; color: #fff; }
        .page-item.disabled .page-link { color: var(--text-muted-color); pointer-events: none; background-color: var(--light-bg); opacity: 0.55; }
        .page-link-prev-next .page-link { width: auto; padding-left: 1.1rem; padding-right: 1.1rem; }

        /* ==========================================================================
           FOOTER: the landing page's last print with a frosted panel on its sky
           ========================================================================== */
        .site-footer { position: relative; isolation: isolate; margin-top: auto; padding: 160px 0 24px; color: var(--text-color); }
        .site-footer > .photo { position: absolute; inset: 0; z-index: -1; overflow: hidden; background: var(--surface-muted); }
        .site-footer > .photo img { width: 100%; height: 100%; object-fit: cover; object-position: 50% 0%; filter: var(--photo-filter); }
        .site-footer > .photo::after { content: ""; position: absolute; inset: 0; background-image: var(--grain); opacity: 0.14; mix-blend-mode: multiply; pointer-events: none; }
        .footer-panel {
            padding: 32px 24px; border-radius: var(--border-radius-xl); border: 1px solid var(--card-border-color);
            background: var(--panel-bg); color: var(--text-color);
            -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
        }
        @media (min-width: 768px) { .footer-panel { padding: 40px; } }
        .footer-top { display: grid; gap: 32px; }
        @media (min-width: 768px) { .footer-top { grid-template-columns: 2fr 1fr 1fr; } }
        .footer-brand { display: inline-flex; align-items: center; gap: 8px; font-size: 1rem; line-height: 1.5rem; font-weight: 500; color: var(--text-color); text-decoration: none; }
        .footer-brand:hover { color: var(--text-color); }
        .footer-blurb { margin: 12px 0 0; max-width: 320px; font-size: 0.875rem; line-height: 1.25rem; color: var(--text-muted-color); }
        .footer-newsletter { margin-top: 24px; max-width: 360px; }
        .footer-label { display: block; margin-bottom: 8px; font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; color: var(--text-color); }
        .footer-newsletter-input { background: var(--card-bg); }
        .footer-col h2 { margin: 0 0 12px; font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; letter-spacing: 0; }
        .footer-col ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; font-size: 0.875rem; line-height: 1.25rem; }
        .footer-col a { color: var(--text-muted-color); text-decoration: none; transition: color var(--duration-base) var(--ease-standard); }
        .footer-col a:hover { color: var(--text-color); }
        .footer-bottom {
            margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--card-border-color);
            display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
            font-size: 0.75rem; line-height: 1rem; color: var(--text-subtle-color);
        }
        .back-to-top-link { display: inline-flex; align-items: center; gap: 4px; padding: 0; border: none; background: none; color: var(--text-muted-color); font-size: 0.75rem; line-height: 1rem; transition: color var(--duration-base) var(--ease-standard); }
        .back-to-top-link:hover { color: var(--text-color); }

        /* ==========================================================================
           MODALS
           ========================================================================== */
        .modal-content { border-radius: var(--border-radius-lg); border: 1px solid var(--card-border-color); background-color: var(--card-bg); color: var(--text-color); box-shadow: var(--shadow-lg); }
        body.dark-mode .modal-header .btn-close { filter: invert(1) grayscale(100%) brightness(200%); }
        .admin-controls { position: fixed; bottom: 25px; right: 25px; z-index: 1030; }
        .add-article-btn { width: 56px; height: 56px; border-radius: var(--border-radius-md); color: #fff; border: none; display: flex; align-items: center; justify-content: center; font-size: 1.25rem; cursor: pointer; background-image: none; box-shadow: var(--shadow-lg); transition: box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-base) var(--ease-spring), filter var(--duration-base) var(--ease-standard); }
        .add-article-btn:hover { filter: brightness(1.08); transform: translateY(-4px) scale(1.04); box-shadow: var(--shadow-glow); }
        .add-article-btn:active { transform: translateY(-1px) scale(0.96); }

        /* ==========================================================================
           STATIC PAGES
           ========================================================================== */
        .page-header-static { background-color: var(--card-bg); border-radius: var(--border-radius-lg); padding: clamp(2rem, 5vw, 3.25rem); margin-bottom: 2rem; text-align: center; border: 1px solid var(--card-border-color); position: relative; overflow: hidden; }
        .page-header-static::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 3px; background-image: none; }
        .page-header-static h1 { color: var(--text-color); font-size: var(--text-4xl); margin: 0; }
        .static-content-container { background-color: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: clamp(1.5rem, 5vw, 3rem); font-size: 1rem; line-height: 1.8; box-shadow: var(--shadow-sm); }
        .static-content-container h2 { font-family: var(--font-sans); font-weight: 600; color: var(--text-color); border-bottom: 2px solid var(--secondary-color); padding-bottom: 0.5rem; margin-top: 2.5rem; margin-bottom: 1.5rem; display: inline-block; font-size: var(--text-xl); }
        .static-content-container h2 .icon { margin-right: 0.75rem; color: var(--primary-color); }
        .static-content-container p, .static-content-container li { max-width: 68ch; }
        .static-content-container p.lead { font-family: var(--font-display); font-weight: 400; font-size: 1.25rem; line-height: 1.4; color: var(--text-color); max-width: 40ch; }
        .static-content-container ul { padding-left: 25px; }
        .static-content-container li { margin-bottom: 0.5rem; }
        .contact-card { background-color: var(--light-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); padding: 1.75rem 1.5rem; height: 100%; text-align: center; transition: transform var(--duration-base) var(--ease-premium), box-shadow var(--duration-base) var(--ease-premium), border-color var(--duration-base) var(--ease-standard); }
        .contact-card:hover { transform: translateY(-5px); box-shadow: var(--shadow-md); border-color: rgba(var(--primary-color-rgb), 0.35); }
        .contact-card .icon { font-size: 2.25rem; color: var(--primary-color); margin-bottom: 1rem; }
        body.dark-mode .contact-card { background-color: var(--card-bg); }
        .contact-social-links { display: flex; gap: 1.5rem; justify-content: center; font-size: 1.5rem; }
        .contact-social-links a { color: var(--text-muted-color); transition: color var(--duration-base) var(--ease-standard), transform var(--duration-base) var(--ease-spring); }
        .contact-social-links a:hover { color: var(--secondary-color); transform: translateY(-3px) scale(1.1); }

        /* ==========================================================================
           AUTH
           ========================================================================== */
        .auth-card { max-width: 440px; margin: 3rem auto; background: var(--card-bg); border-radius: var(--border-radius-lg); box-shadow: var(--shadow-lg); border: 1px solid var(--card-border-color); overflow: hidden; }
        .auth-header { padding: 2.25rem 2rem 2rem; background-image: none; text-align: center; }
        .auth-header .icon { font-size: 1.875rem; color: #fff; width: 64px; height: 64px; margin: 0 auto; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.16); border-radius: 50%; }
        .auth-header h1 { font-size: var(--text-2xl); color: #fff; margin-top: 1rem; margin-bottom: 0; }
        .auth-body { padding: 2rem 2.25rem; }
        .input-group-icon { position: relative; }
        .input-group-icon .form-control { padding-left: 2.5rem; }
        .input-group-icon .input-icon { position: absolute; left: 0.8rem; top: 50%; transform: translateY(-50%); color: var(--text-muted-color); transition: color var(--duration-base) var(--ease-standard); }
        .input-group-icon:focus-within .input-icon { color: var(--primary-color); }
        .auth-body .btn { padding: 0.75rem; font-weight: 600; font-size: 1rem; }
        .auth-footer { padding: 1.25rem 2.25rem; text-align: center; border-top: 1px solid var(--card-border-color); background: var(--light-bg); }

        /* ==========================================================================
           PROFILE
           ========================================================================== */
        .profile-header-card { background: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: 2rem; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; align-items: center; text-align: center; position: relative; overflow: hidden; }
        .profile-header-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 96px; background-image: none; }
        .profile-avatar-wrapper { position: relative; margin-bottom: 1rem; z-index: 1; }
        .profile-avatar { width: 116px; height: 116px; border-radius: 50%; background-image: none; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 3rem; font-family: var(--font-sans); border: 5px solid var(--card-bg); box-shadow: var(--shadow-md); }
        .profile-header-card h1 { margin-bottom: 0.25rem; font-size: var(--text-3xl); position: relative; z-index: 1; }
        .profile-header-card .username { color: var(--text-muted-color); font-weight: 600; margin-bottom: 1rem; }
        .profile-stats { display: flex; gap: 1.5rem; margin-top: 1.5rem; border-top: 1px solid var(--card-border-color); padding-top: 1.5rem; width: 100%; justify-content: center; }
        .stat-item { text-align: center; background: none; border: none; padding: 0.4rem 1rem; border-radius: var(--border-radius-md); transition: background-color var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); }
        button.stat-item { cursor: pointer; }
        button.stat-item:hover { background-color: var(--light-bg); transform: translateY(-2px); }
        .stat-item .icon { font-size: 1.25rem; color: var(--secondary-color); margin-bottom: 0.5rem; }
        .stat-item .count { font-size: 1.25rem; font-weight: 600; color: var(--text-color); letter-spacing: -0.03em; }
        .stat-item .label { font-size: 0.75rem; font-weight: 600; color: var(--text-muted-color); }
        .profile-tabs .nav-link { padding: 0.75rem 1rem; }

        /* ==========================================================================
           EMPTY / ERROR STATES
           ========================================================================== */
        .state-card { background-color: var(--card-bg); border-radius: var(--border-radius-lg); text-align: center; padding: clamp(2.25rem, 6vw, 3.5rem) 2rem; border: 1px dashed var(--card-border-color); }
        .state-card.state-card-narrow { max-width: 600px; margin-left: auto; margin-right: auto; }
        .state-card.state-card-solid { border-style: solid; box-shadow: var(--shadow-sm); }
        .state-card-icon { width: 72px; height: 72px; border-radius: 50%; background-image: none; color: var(--primary-color); display: inline-flex; align-items: center; justify-content: center; font-size: 1.5rem; margin-bottom: 1.25rem; }
        .state-card-icon.state-card-icon-sm { width: 52px; height: 52px; font-size: 1rem; margin-bottom: 0.85rem; }
        .state-card.state-card-danger .state-card-icon { background-image: none; color: #DC2626; }
        .state-card-title { font-size: var(--text-2xl); margin-bottom: 0.5rem; }
        .state-card-text { color: var(--text-muted-color); max-width: 46ch; margin: 0 auto; }
        .state-card-actions { margin-top: 1.5rem; display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; }

        /* ==========================================================================
           BOOKMARK
           ========================================================================== */
        .bookmark-btn { background: none; border: none; font-size: 1.5rem; color: var(--text-muted-color); cursor: pointer; padding: 0.25rem 0.5rem; transition: color var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); vertical-align: middle; border-radius: 50%; }
        .bookmark-btn.active { color: var(--bookmark-active-color); transform: scale(1.1); }
        .bookmark-btn:hover { color: var(--secondary-light); transform: scale(1.12); }
        .article-card .bookmark-btn { font-size: 1.125rem; position: relative; z-index: 2; }

        /* ==========================================================================
           AI SUMMARY / TAKEAWAYS / SKELETON
           ========================================================================== */
        .summary-box, .takeaways-box { background-color: var(--card-bg); border: 1px solid rgba(var(--primary-color-rgb), 0.16); border-radius: var(--border-radius-md); margin: 1.6rem 0; padding: 1.5rem 1.65rem; }
        .summary-box h2, .takeaways-box h2 { font-family: var(--font-sans); font-weight: 600; font-size: 0.75rem; color: var(--primary-color); margin-bottom: 0.75rem; }
        .summary-box p { font-size: 1rem; line-height: 1.6; max-width: 62ch; margin: 0; }
        .takeaways-box { border-left: 3px solid var(--secondary-color); }
        .takeaways-box h2 { color: var(--secondary-color); }
        .takeaways-box ul { margin: 0; padding-left: 1.15rem; }
        .takeaways-box li { margin-bottom: 0.45rem; }
        .takeaways-box li:last-child { margin-bottom: 0; }
        .content-text { white-space: pre-wrap; line-height: 1.8; font-size: 1rem; color: var(--text-color); max-width: 70ch; }
        .content-divider-heading { font-size: var(--text-xl); margin: 0 0 1rem; }
        .comment-login-prompt { text-align: center; margin-top: 1.5rem; padding: 1.75rem; background: var(--light-bg); border-radius: var(--border-radius-md); border: 1px solid var(--card-border-color); }
        .comment-login-prompt i { font-size: 1.5rem; color: var(--text-muted-color); margin-bottom: 0.5rem; display: block; }

        .ai-skeleton-box { border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); padding: 1.25rem 1.4rem; margin-bottom: 1.25rem; }
        .ai-skeleton-label { height: 0.72rem; width: 120px; border-radius: var(--border-radius-xs); margin-bottom: 0.7rem; }
        .ai-skeleton-line { height: 0.82rem; border-radius: var(--border-radius-xs); margin-bottom: 0.6rem; }
        .ai-skeleton-line:last-child { margin-bottom: 0; }
        .ai-skeleton-line.w-100 { width: 100%; } .ai-skeleton-line.w-95 { width: 95%; } .ai-skeleton-line.w-90 { width: 90%; } .ai-skeleton-line.w-80 { width: 80%; } .ai-skeleton-line.w-70 { width: 70%; }
        .ai-skeleton-label, .ai-skeleton-line { background: none; background-size: 200% 100%; animation: skeletonShimmer 1.5s ease-in-out infinite; }
        .ai-skeleton-caption { color: var(--text-muted-color); font-size: var(--text-sm); text-align: center; margin: 0.25rem 0 0; }
        .ai-skeleton-caption i { color: var(--primary-color); }

        .hero-image-wrap { position: relative; overflow: hidden; border-radius: var(--border-radius-md); box-shadow: var(--shadow-md); margin: 1.25rem 0; aspect-ratio: 16 / 8; max-height: 460px; background: var(--skeleton-base); background-image: none; background-size: 200% 100%; animation: skeletonShimmer 1.5s ease-in-out infinite; }
        .hero-image-wrap.is-loaded, .hero-image-wrap.img-fallback { animation: none; background-image: none; }
        .hero-image-wrap.img-fallback { background: var(--light-bg); }
        .hero-image-wrap .hero-image { width: 100%; height: 100%; object-fit: cover; display: block; opacity: 0; transition: opacity var(--duration-slow) var(--ease-standard); }
        .hero-image-wrap .hero-image.is-loaded { opacity: 1; }
        .hero-image-wrap.img-fallback .hero-image { display: none; }
        .hero-image-wrap .img-fallback-icon { font-size: 2.25rem; }
        .article-full-content-wrapper { padding: clamp(1.25rem, 4vw, 3rem); margin: 1rem auto 2rem; max-width: 860px; box-shadow: var(--shadow-md); }
        .article-title-main { font-size: var(--text-3xl); color: var(--text-color); margin-bottom: 0.75rem; }
        .article-meta-detailed .meta-item { font-size: 0.75rem; }

        /* ==========================================================================
           COMMENTS
           ========================================================================== */
        .comment-section h2 { padding-bottom: 0.75rem; border-bottom: 1px solid var(--card-border-color); font-size: var(--text-xl); }
        .comment-form-heading { font-size: var(--text-lg); font-weight: 600; margin-bottom: 1rem; }
        .comment-thread { position: relative; }
        #comments-list > .comment-thread + .comment-thread { margin-top: 1.75rem; padding-top: 1.75rem; border-top: 1px solid var(--card-border-color); }
        .comment-container { display: flex; gap: 1rem; align-items: flex-start; }
        .comment-replies { margin-left: 3.5rem; padding-left: 1.25rem; margin-top: 1.25rem; border-left: 2px solid var(--card-border-color); }
        .comment-replies.comment-replies-flat { margin-left: 1.5rem; padding-left: 1rem; }
        .comment-replies:empty { display: none; margin: 0; padding: 0; border: 0; }
        .comment-replies > .comment-thread + .comment-thread { margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px dashed var(--card-border-color); }
        .comment-avatar { width: 44px; height: 44px; border-radius: 50%; background-image: none; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0; box-shadow: var(--shadow-sm); }
        .comment-replies .comment-avatar { width: 38px; height: 38px; font-size: 0.875rem; }
        .comment-body { flex-grow: 1; min-width: 0; }
        .comment-header { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.25rem; }
        .comment-author { font-weight: 600; color: var(--text-color); }
        .comment-date { font-size: 0.75rem; color: var(--text-muted-color); }
        .comment-content { word-wrap: break-word; }
        .comment-actions { position: relative; display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.5rem; }
        .comment-actions button { background: none; border: none; color: var(--text-muted-color); padding: 0.28rem 0.6rem; border-radius: var(--border-radius-pill); font-size: 0.75rem; font-weight: 600; display: flex; align-items: center; gap: 0.3rem; transition: color var(--duration-fast) var(--ease-standard), background-color var(--duration-fast) var(--ease-standard); }
        .comment-actions button:hover { color: var(--primary-color); background-color: rgba(var(--primary-color-rgb), 0.1); }
        .react-btn { position: relative; }
        .reaction-box { display: none; position: absolute; bottom: 100%; left: 0; margin-bottom: 8px; background-color: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-pill); padding: 4px 8px; box-shadow: var(--shadow-lg); z-index: 10; white-space: nowrap; animation: fadeInUp 0.22s var(--ease-spring); }
        .reaction-box.show { display: flex; gap: 2px; }
        .reaction-emoji { font-size: 1.25rem; cursor: pointer; transition: transform 0.18s var(--ease-spring), background-color var(--duration-fast) var(--ease-standard); padding: 4px; background: none; border: none; border-radius: 50%; line-height: 1; }
        .reaction-emoji:hover { transform: scale(1.3) translateY(-2px); }
        .reaction-emoji.is-selected { background-color: rgba(var(--primary-color-rgb), 0.16); box-shadow: inset 0 0 0 1.5px rgba(var(--primary-color-rgb), 0.5); }
        .reaction-summary { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
        .reaction-pill { display: flex; align-items: center; background-color: rgba(var(--primary-color-rgb), 0.08); border: 1px solid transparent; border-radius: var(--border-radius-pill); padding: 2px 9px; font-size: 0.75rem; font-weight: 600; cursor: default; transition: background-color var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard); }
        .reaction-pill.user-reacted { background-image: none; color: #fff; border-color: transparent; }
        .reaction-pill .emoji { font-size: 0.875rem; margin-right: 4px; }
        .reply-form-container, .edit-form-container { padding: 1rem; border-radius: var(--border-radius-md); margin-top: 0.75rem; background-color: var(--light-bg); border: 1px solid var(--card-border-color); display: none; }

        /* ==========================================================================
           COMMUNITY HUB + LIST HEADERS
           ========================================================================== */
        .community-hub-header { display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap; background-color: var(--card-bg); background-image: none; border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: clamp(1.5rem, 4vw, 2.5rem); margin-bottom: 2rem; }
        .community-hub-header-text { max-width: 640px; }
        .community-hub-header h1 { font-family: var(--font-display); font-weight: 600; font-size: var(--text-3xl); margin: 0.4rem 0 0.6rem; }
        .community-hub-sub { color: var(--text-muted-color); margin: 0; max-width: 58ch; }
        .community-hub-cta { flex-shrink: 0; white-space: nowrap; }
        .list-page-header { margin-bottom: 1.75rem; padding-bottom: 1.25rem; border-bottom: 1px solid var(--card-border-color); }
        .list-page-header .eyebrow { display: block; margin-bottom: 0.45rem; }
        .list-page-header h1 { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: var(--text-3xl); }

        /* ==========================================================================
           FEATURED STORY (parallax layer) + AI SYNTHESIS
           ========================================================================== */
        .featured-story { display: flex; background-color: var(--card-bg); border-radius: var(--border-radius-lg); box-shadow: var(--shadow-lg); margin-bottom: 2rem; overflow: hidden; border: 1px solid var(--card-border-color); position: relative; }
        .featured-story-image { flex: 0 0 55%; background-size: cover; background-position: center; background-color: var(--skeleton-base); min-height: 440px; position: relative; will-change: transform; transform: translate3d(0, var(--parallax-y, 0px), 0) scale(1.08); transition: transform 80ms linear; }
        .featured-story-image::after { content: ''; position: absolute; inset: 0; background: none; }
        .featured-story-content { flex: 0 0 45%; padding: clamp(1.5rem, 4vw, 3rem); display: flex; flex-direction: column; justify-content: center; position: relative; }
        .featured-story-content .meta-item:first-child { background-image: none; color: var(--accent-color); border-color: transparent; font-weight: 600; }
        .featured-story-content .meta-item:first-child i { color: var(--accent-color); }
        .featured-story-content h2 { font-size: var(--text-3xl); margin: 1rem 0; }
        .featured-story-content h2 a { color: var(--text-color); text-decoration: none; transition: color var(--duration-base) var(--ease-standard); }
        .featured-story-content h2 a:hover { color: var(--primary-color); }
        .featured-story-content .description { font-size: 1rem; color: var(--text-muted-color); margin-bottom: 2rem; }
        .featured-story-content .read-more-btn { background: var(--text-color); color: var(--card-bg); padding: 0.8rem 1.6rem; text-decoration: none; border-radius: var(--border-radius-sm); font-weight: 600; font-size: 0.875rem; transition: background-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); align-self: flex-start; }
        .featured-story-content .read-more-btn:hover { background-image: none; color: #fff; box-shadow: var(--shadow-glow); transform: translateY(-2px); }

        .ai-synthesis-card { background-color: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: clamp(1.5rem, 4vw, 2.5rem); margin-bottom: 2rem; box-shadow: var(--shadow-sm); position: relative; overflow: hidden; }
        .ai-synthesis-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background-image: none; }
        .synthesis-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.1rem; }
        .synthesis-header i { font-size: 0.875rem; color: #fff; background-image: none; width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center; border-radius: var(--border-radius-sm); flex-shrink: 0; }
        .synthesis-header h2 { font-size: 0.75rem; margin: 0; font-weight: 600; color: var(--text-muted-color); }
        .synthesis-text { font-size: 1.5rem; line-height: 1.35; text-align: left; color: var(--text-color); position: relative; max-width: 40ch; margin: 0; }
        .synthesis-keywords { margin-top: 1.5rem; padding-top: 1.25rem; border-top: 1px solid var(--card-border-color); text-align: left; position: relative; }
        .synthesis-keywords .keyword-tag { display: inline-block; background-color: var(--light-bg); border: 1px solid var(--card-border-color); color: var(--text-color); padding: 0.35rem 0.95rem; border-radius: var(--border-radius-pill); margin: 0.2rem 0.3rem 0.2rem 0; font-size: 0.75rem; font-weight: 600; text-decoration: none; transition: background-color var(--duration-fast) var(--ease-standard), color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); }
        .synthesis-keywords .keyword-tag:hover { background-image: none; color: #fff; border-color: transparent; transform: translateY(-2px); }

        /* ==========================================================================
           READING PROGRESS BAR
           ========================================================================== */
        .reading-progress { position: fixed; top: 0; left: 0; height: 3px; width: 100%; z-index: 1050; background: transparent; pointer-events: none; }
        .reading-progress__fill { height: 100%; width: 0%; background-image: none; transform-origin: left; transition: width 80ms linear; }

        /* ==========================================================================
           ARTICLE TOOLBAR (read time, listen, text size, share)
           ========================================================================== */
        .article-toolbar { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; padding: 0.85rem 0; margin: 1.25rem 0; border-top: 1px solid var(--card-border-color); border-bottom: 1px solid var(--card-border-color); }
        .toolbar-btn { display: inline-flex; align-items: center; gap: 0.4rem; background: none; border: 1px solid var(--card-border-color); color: var(--text-muted-color); border-radius: var(--border-radius-pill); padding: 0.35rem 0.85rem; font-size: 0.75rem; font-weight: 600; cursor: pointer; transition: color var(--duration-fast) var(--ease-standard), border-color var(--duration-fast) var(--ease-standard), background-color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); white-space: nowrap; }
        .toolbar-btn:hover { color: var(--primary-color); border-color: var(--primary-color); transform: translateY(-1px); }
        .toolbar-btn.is-active { background-image: none; color: #fff; border-color: transparent; }
        .read-time-badge { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.75rem; font-weight: 600; color: var(--text-muted-color); margin-right: auto; }
        .read-time-badge i { color: var(--secondary-color); }
        .toolbar-spacer { margin-left: auto; }

        /* Share menu */
        .share-wrap { position: relative; }
        .share-menu { display: none; position: absolute; right: 0; bottom: calc(100% + 8px); background: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); box-shadow: var(--shadow-lg); padding: 0.4rem; min-width: 190px; z-index: 20; animation: fadeInUp 0.22s var(--ease-spring); }
        .share-menu.show { display: block; }
        .share-menu button, .share-menu a { display: flex; align-items: center; gap: 0.65rem; width: 100%; background: none; border: none; color: var(--text-color); text-decoration: none; padding: 0.55rem 0.75rem; border-radius: var(--border-radius-sm); font-size: 0.75rem; font-weight: 500; text-align: left; transition: background-color var(--duration-fast) var(--ease-standard); }
        .share-menu button:hover, .share-menu a:hover { background-color: var(--light-bg); color: var(--text-color); }
        .share-menu i { width: 1.1rem; text-align: center; }
        .share-menu .i-whatsapp { color: #25D366; } .share-menu .i-x { color: var(--text-color); }
        .share-menu .i-facebook { color: #1877F2; } .share-menu .i-linkedin { color: #0A66C2; }
        .share-menu .i-link { color: var(--primary-color); }

        /* Text-size presets, applied to the article body */
        .article-full-content-wrapper[data-text-size="large"] .content-text,
        .article-full-content-wrapper[data-text-size="large"] .summary-box p,
        .article-full-content-wrapper[data-text-size="large"] .takeaways-box li { font-size: 1.125rem; line-height: 1.85; }
        .article-full-content-wrapper[data-text-size="xlarge"] .content-text,
        .article-full-content-wrapper[data-text-size="xlarge"] .summary-box p,
        .article-full-content-wrapper[data-text-size="xlarge"] .takeaways-box li { font-size: 1.25rem; line-height: 1.9; }
        .tts-highlight { background: rgba(var(--accent-color-rgb), 0.22); border-radius: 3px; }

        /* ==========================================================================
           RECENTLY VIEWED STRIP
           ========================================================================== */
        .recent-strip { margin-bottom: 2rem; }
        .recent-strip__head { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin-bottom: 0.85rem; }
        .recent-strip__list { display: flex; gap: 0.85rem; overflow-x: auto; padding-bottom: 0.5rem; scroll-snap-type: x proximity; -webkit-overflow-scrolling: touch; }
        .recent-strip__list::-webkit-scrollbar { height: 6px; }
        .recent-item { flex: 0 0 clamp(200px, 44vw, 260px); scroll-snap-align: start; background: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); padding: 0.85rem 1rem; text-decoration: none; color: var(--text-color); transition: transform var(--duration-base) var(--ease-premium), box-shadow var(--duration-base) var(--ease-premium), border-color var(--duration-base) var(--ease-standard); }
        .recent-item:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); border-color: rgba(var(--primary-color-rgb), 0.35); color: var(--text-color); }
        .recent-item__title { font-size: 0.875rem; font-weight: 600; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; margin: 0 0 0.35rem; letter-spacing: -0.02em; }
        .recent-item__source { font-size: 0.75rem; font-weight: 600; color: var(--text-muted-color); }
        .link-btn { background: none; border: none; color: var(--text-muted-color); font-size: 0.75rem; font-weight: 600; text-decoration: underline; text-underline-offset: 0.2em; padding: 0; }
        .link-btn:hover { color: var(--primary-color); }

        /* ==========================================================================
           KEYBOARD SHORTCUTS
           ========================================================================== */
        .kbd-list { display: grid; grid-template-columns: 1fr auto; gap: 0.6rem 1.5rem; align-items: center; }
        .kbd-list dt { color: var(--text-color); font-size: 0.875rem; }
        .kbd-list dd { margin: 0; text-align: right; }
        kbd { font-family: var(--font-mono); font-size: 0.75rem; background: var(--light-bg); border: 1px solid var(--card-border-color); border-bottom-width: 2px; border-radius: var(--border-radius-xs); padding: 0.15rem 0.45rem; color: var(--text-color); }

        /* Comment sort control */
        .comment-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
        .sort-select { font-size: 0.75rem; font-weight: 600; padding: 0.35rem 2rem 0.35rem 0.75rem; border-radius: var(--border-radius-pill); border: 1px solid var(--card-border-color); background-color: var(--card-bg); color: var(--text-color); width: auto; }

        /* ==========================================================================
           ONE THEME: every component on the landing page's paper, ink and teal
           ========================================================================== */
        body {
            --bs-body-color: var(--text-color); --bs-body-bg: var(--light-bg); --bs-border-color: var(--card-border-color);
            --bs-secondary-color: var(--text-muted-color); --bs-secondary-color-rgb: var(--text-muted-color-rgb);
            --bs-emphasis-color: var(--text-color); --bs-heading-color: var(--text-color);
        }
        b, strong { font-weight: 600; }
        .fw-bold { font-weight: 600 !important; }
        h1, h2, h3, h4, h5, h6, .section-heading { text-wrap: balance; }
        p, li, dd, figcaption { text-wrap: pretty; }
        a { color: var(--accent-color); }
        a:hover { color: var(--text-color); }
        hr { border-color: var(--card-border-color); opacity: 1; }
        ::selection { background: var(--text-color); color: var(--light-bg); }

        /* Brand mark: ink tile, paper lines, the short teal line is the gist. */
        .logo-mark { display: block; width: 24px; height: 24px; flex: none; }
        .logo-tile { fill: var(--ink); }
        .logo-line { fill: var(--light-bg); }
        .logo-gist { fill: var(--logo-gist); }

        /* Named text styles, identical to the landing page. */
        .heading-1 { font-size: 3.75rem; line-height: 1; font-weight: 600; letter-spacing: -0.03em; }
        .heading-2 { font-size: 3rem; line-height: 1; font-weight: 600; letter-spacing: -0.03em; }
        .heading-3 { font-size: 2.25rem; line-height: 2.5rem; font-weight: 600; letter-spacing: -0.025em; }
        .heading-4 { font-size: 1.875rem; line-height: 2.25rem; font-weight: 600; letter-spacing: -0.02em; }
        .heading-5 { font-size: 1.5rem; line-height: 2rem; font-weight: 600; letter-spacing: -0.015em; }
        .heading-6 { font-size: 1.25rem; line-height: 1.75rem; font-weight: 600; }
        .title-1 { font-size: 1.25rem; line-height: 1.75rem; font-weight: 500; letter-spacing: -0.01em; }
        .title-2 { font-size: 1.125rem; line-height: 1.75rem; font-weight: 500; }
        .title-3 { font-size: 1rem; line-height: 1.5rem; font-weight: 500; }
        .title-4 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; }
        .body-6 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 400; }
        .caption-3 { font-size: 0.75rem; line-height: 1rem; font-weight: 600; letter-spacing: 0.04em; }
        .caption-5 { font-size: 0.75rem; line-height: 1rem; font-weight: 400; }

        .eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; color: var(--accent-color); }
        .eyebrow::before { content: ""; width: 24px; height: 1px; background: currentColor; }
        .section-heading { font-weight: 600; letter-spacing: -0.015em; }

        /* Photographs share one treatment: theme filter and film grain. */
        .photo-credit {
            position: absolute; top: 12px; right: 12px; z-index: 2; padding: 4px 8px; border-radius: var(--border-radius-xs);
            background: var(--glass-bg); color: var(--text-muted-color); font-size: 0.75rem; line-height: 1rem;
            -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
        }

        /* Buttons: ink primary and paper ghost, exactly like the landing page. */
        .btn { border-radius: var(--border-radius-sm); font-weight: 600; letter-spacing: 0; }
        .btn:hover { transform: none; }
        .btn:active { transform: scale(0.98); }
        .btn-primary {
            --bs-btn-color: var(--on-ink); --bs-btn-bg: var(--ink); --bs-btn-border-color: var(--ink);
            --bs-btn-hover-color: var(--on-ink); --bs-btn-hover-bg: var(--ink-hover); --bs-btn-hover-border-color: var(--ink-hover);
            --bs-btn-active-color: var(--on-ink); --bs-btn-active-bg: var(--ink-hover); --bs-btn-active-border-color: var(--ink-hover);
            --bs-btn-disabled-color: var(--on-ink); --bs-btn-disabled-bg: var(--ink); --bs-btn-disabled-border-color: var(--ink);
            --bs-btn-focus-shadow-rgb: var(--primary-color-rgb);
        }
        .btn-primary:hover, .btn-primary:focus { box-shadow: none; }
        .btn-primary-modal { background: var(--ink); color: var(--on-ink); border-radius: var(--border-radius-sm); }
        .btn-primary-modal:hover { background: var(--ink-hover); color: var(--on-ink); filter: none; box-shadow: none; transform: none; }
        .btn-outline-primary {
            --bs-btn-color: var(--text-color); --bs-btn-bg: var(--card-bg); --bs-btn-border-color: var(--card-border-color);
            --bs-btn-hover-color: var(--text-color); --bs-btn-hover-bg: var(--surface-muted); --bs-btn-hover-border-color: var(--card-border-color);
            --bs-btn-active-color: var(--text-color); --bs-btn-active-bg: var(--surface-muted); --bs-btn-active-border-color: var(--card-border-color);
        }
        .btn-outline-secondary {
            --bs-btn-color: var(--text-muted-color); --bs-btn-border-color: var(--card-border-color);
            --bs-btn-hover-color: var(--text-color); --bs-btn-hover-bg: var(--surface-muted); --bs-btn-hover-border-color: var(--card-border-color);
            --bs-btn-active-color: var(--text-color); --bs-btn-active-bg: var(--surface-muted); --bs-btn-active-border-color: var(--card-border-color);
        }
        .form-control:focus, .form-select:focus { border-color: var(--text-color); }
        .form-check-input:checked { background-color: #12645E; border-color: #12645E; }
        .form-check-input:focus { border-color: var(--text-color); box-shadow: 0 0 0 3px rgba(var(--primary-color-rgb), 0.16); }
        .dropdown-menu {
            --bs-dropdown-bg: var(--card-bg); --bs-dropdown-color: var(--text-color); --bs-dropdown-border-color: var(--card-border-color);
            --bs-dropdown-border-radius: var(--border-radius-md); --bs-dropdown-padding-x: 4px; --bs-dropdown-padding-y: 4px;
            --bs-dropdown-link-color: var(--text-color); --bs-dropdown-link-hover-color: var(--text-color); --bs-dropdown-link-hover-bg: var(--surface-muted);
            --bs-dropdown-link-active-color: var(--on-ink); --bs-dropdown-link-active-bg: var(--ink); --bs-dropdown-divider-bg: var(--card-border-color);
            box-shadow: var(--shadow-lg) !important;
        }
        .dropdown-item { border-radius: var(--border-radius-sm); font-size: 0.875rem; line-height: 1.25rem; padding: 8px 12px; }
        .table { --bs-table-bg: transparent; --bs-table-color: var(--text-color); --bs-table-border-color: var(--card-border-color); }
        .badge.bg-secondary { background: var(--surface-muted) !important; color: var(--text-color); font-weight: 500; }
        .modal-content { border-radius: var(--border-radius-xl); }
        #alert-placeholder .alert { border-radius: var(--border-radius-md); }

        /* Tabs: sentence case with an ink underline. */
        .nav-tabs .nav-link { font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; padding: 12px 8px; }
        .nav-tabs .nav-link::after { background: var(--text-color); }

        /* Story cards. */
        .article-card, .article-full-content-wrapper { box-shadow: none; }
        .article-card::after { content: none; }
        .article-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-lg); border-color: var(--card-border-color); }
        .article-title { font-weight: 600; letter-spacing: -0.015em; }
        .meta-item { border-radius: var(--border-radius-xs); background: transparent; padding: 2px 8px; line-height: 1rem; font-weight: 500; }
        .meta-item i { color: var(--accent-color); }
        /* One primary action per view: card links are ghost buttons that fill with ink on hover. */
        .read-more { padding: 8px 12px; font-size: 0.875rem; line-height: 1.25rem; background: var(--card-bg); border: 1px solid var(--card-border-color); color: var(--text-color) !important; }
        .read-more:hover, .article-card:hover .read-more { background: var(--ink); border-color: var(--ink); color: var(--on-ink) !important; box-shadow: none; }

        /* Loading placeholders pulse flat instead of shimmering. */
        @keyframes skeletonPulse { from { background-color: var(--skeleton-base); } to { background-color: var(--skeleton-sheen); } }
        .article-image-container, .hero-image-wrap, .ai-skeleton-label, .ai-skeleton-line {
            background-image: none; background-color: var(--skeleton-base);
            animation: skeletonPulse 1200ms var(--ease-standard) infinite alternate;
        }
        .article-image-container.is-loaded, .article-image-container.img-fallback,
        .hero-image-wrap.is-loaded, .hero-image-wrap.img-fallback { animation: none; background-color: var(--surface-color); }

        .page-item .page-link { font-weight: 500; }
        .page-item .page-link:hover { border-color: var(--text-color); color: var(--text-color); transform: none; }
        .page-item.active .page-link { background: var(--ink); border-color: var(--ink); color: var(--on-ink); }

        .add-article-btn { background: var(--ink); color: var(--on-ink); border-radius: var(--border-radius-lg); }
        .add-article-btn:hover { background: var(--ink-hover); filter: none; transform: translateY(-2px); box-shadow: var(--shadow-lg); }

        /* Page titles read like the landing page's section titles. */
        .page-header-static { background: none; border: 0; border-bottom: 1px solid var(--card-border-color); border-radius: 0; padding: 24px 0 32px; margin-bottom: 32px; text-align: left; overflow: visible; }
        .page-header-static::before { content: none; }
        .page-header-static h1 { font-size: 2.25rem; line-height: 2.5rem; font-weight: 600; letter-spacing: -0.025em; }
        @media (min-width: 768px) { .page-header-static h1 { font-size: 3rem; line-height: 1; letter-spacing: -0.03em; } }
        .static-content-container { border-radius: var(--border-radius-xl); box-shadow: none; }
        .static-content-container h2 { display: flex; align-items: center; gap: 12px; border-bottom: 0; padding-bottom: 0; font-size: 1.25rem; line-height: 1.75rem; }
        .static-content-container h2 .icon { margin-right: 0; color: var(--accent-color); }
        .static-content-container > h2:first-child { margin-top: 0; }
        .static-content-container p.lead { font-size: 1.125rem; line-height: 1.75rem; color: var(--text-muted-color); max-width: 60ch; }
        .contact-card { border-radius: var(--border-radius-lg); }
        .contact-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-lg); border-color: var(--card-border-color); }
        .contact-card .icon { color: var(--accent-color); }
        .list-page-header h1, .community-hub-header h1 { font-weight: 600; letter-spacing: -0.025em; }
        .community-hub-header { background: var(--card-bg); border-radius: var(--border-radius-xl); }

        /* Sign in and sign up. */
        .auth-card { border-radius: var(--border-radius-xl); box-shadow: var(--shadow-lg); }
        .auth-header { background: none; padding: 40px 32px 8px; }
        .auth-header .icon { width: 48px; height: 48px; border-radius: var(--border-radius-md); background: var(--surface-muted); color: var(--text-color); font-size: 1.5rem; }
        .auth-header .icon.icon-logo { background: none; }
        .auth-header .icon .logo-mark { width: 48px; height: 48px; }
        .auth-header h1 { color: var(--text-color); font-size: 1.5rem; line-height: 2rem; }
        .auth-footer { background: var(--surface-color); }

        /* Profile. */
        .profile-header-card { border-radius: var(--border-radius-xl); box-shadow: none; }
        .profile-header-card::before { background: var(--surface-color); }
        .profile-avatar { background: var(--accent-color); color: var(--light-bg); box-shadow: none; }
        .stat-item .icon { color: var(--accent-color); }

        /* Empty and error states. */
        .state-card { border-radius: var(--border-radius-xl); }
        .state-card-icon { background: var(--surface-muted); color: var(--text-color); border-radius: var(--border-radius-lg); }
        .state-card.state-card-danger .state-card-icon { background: rgba(220, 38, 38, 0.12); color: #DC2626; }

        .bookmark-btn:hover { color: var(--accent-color); }

        /* Summary and takeaways, laid out like the story card on the landing page. */
        .summary-box, .takeaways-box { background: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: 24px; }
        .summary-box h2, .takeaways-box h2 { display: flex; align-items: center; margin-bottom: 8px; font-size: 0.75rem; line-height: 1rem; font-weight: 600; letter-spacing: 0.04em; color: var(--text-subtle-color); }
        .summary-box h2 i, .takeaways-box h2 i { color: var(--accent-color); }
        .takeaways-box h2 { color: var(--text-subtle-color); }
        .takeaways-box ul { list-style: none; padding: 0; display: flex; flex-direction: column; gap: 12px; }
        .takeaways-box li { position: relative; padding-left: 28px; margin: 0; }
        .takeaways-box li::before {
            content: ""; position: absolute; left: 0; top: 2px; width: 20px; height: 20px; background: var(--accent-color);
            -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 256 256'%3E%3Cpath d='M173.66,98.34a8,8,0,0,1,0,11.32l-56,56a8,8,0,0,1-11.32,0l-24-24a8,8,0,0,1,11.32-11.32L112,148.69l50.34-50.35A8,8,0,0,1,173.66,98.34ZM232,128A104,104,0,1,1,128,24,104.11,104.11,0,0,1,232,128Zm-16,0a88,88,0,1,0-88,88A88.1,88.1,0,0,0,216,128Z'/%3E%3C/svg%3E") center / contain no-repeat; mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 256 256'%3E%3Cpath d='M173.66,98.34a8,8,0,0,1,0,11.32l-56,56a8,8,0,0,1-11.32,0l-24-24a8,8,0,0,1,11.32-11.32L112,148.69l50.34-50.35A8,8,0,0,1,173.66,98.34ZM232,128A104,104,0,1,1,128,24,104.11,104.11,0,0,1,232,128Zm-16,0a88,88,0,1,0-88,88A88.1,88.1,0,0,0,216,128Z'/%3E%3C/svg%3E") center / contain no-repeat;
        }

        /* Featured story and today's briefing. */
        .featured-story { border-radius: var(--border-radius-xl); box-shadow: none; }
        .featured-story-image::after { content: none; }
        .featured-story-content .meta-item:first-child { background: var(--surface-muted); color: var(--text-color); border-color: transparent; font-weight: 500; }
        .featured-story-content .meta-item:first-child i { color: var(--accent-color); }
        .featured-story-content h2 { font-size: 1.875rem; line-height: 2.25rem; letter-spacing: -0.02em; }
        .featured-story-content .read-more-btn { display: inline-flex; align-items: center; gap: 8px; align-self: flex-start; padding: 8px 12px; background: var(--ink); color: var(--on-ink); }
        .featured-story-content .read-more-btn:hover { background: var(--ink-hover); color: var(--on-ink); box-shadow: none; transform: none; }

        .ai-synthesis-card {
            position: relative; isolation: isolate; overflow: hidden; padding: 160px 8px 8px;
            border: 1px solid var(--card-border-color); border-radius: var(--border-radius-xl); box-shadow: none;
            background-color: var(--photo-blend); background-blend-mode: multiply;
            background-image: url("{{ url_for('static', filename='landing/darjeeling-1200.webp') }}");
            background-image: image-set(url("{{ url_for('static', filename='landing/darjeeling-1200.webp') }}") 1x, url("{{ url_for('static', filename='landing/darjeeling-2400.webp') }}") 2x);
            background-size: cover; background-position: 50% 40%;
        }
        .ai-synthesis-card::before { content: none; }
        .ai-synthesis-card::after { content: ""; position: absolute; inset: 0; z-index: -1; background-image: var(--grain); opacity: 0.14; mix-blend-mode: multiply; pointer-events: none; }
        .synthesis-panel {
            padding: 24px; border-radius: var(--border-radius-lg); border: 1px solid var(--card-border-color);
            background: var(--panel-bg); -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
        }
        @media (min-width: 768px) { .ai-synthesis-card { padding: 200px 8px 8px; } .synthesis-panel { padding: 32px; } }
        .synthesis-header { margin-bottom: 16px; }
        .synthesis-header h2 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; color: var(--accent-color); }
        .synthesis-text { font-size: 1.5rem; line-height: 2rem; font-weight: 600; letter-spacing: -0.015em; max-width: 52ch; }
        @media (min-width: 992px) { .synthesis-text { font-size: 1.875rem; line-height: 2.25rem; letter-spacing: -0.02em; } }
        .synthesis-keywords .keyword-tag { border-radius: var(--border-radius-xs); background: var(--surface-muted); border-color: transparent; padding: 4px 8px; font-weight: 500; }
        .synthesis-keywords .keyword-tag:hover { background: var(--ink); color: var(--on-ink); transform: none; }

        /* Comments. */
        .comment-avatar { background: var(--surface-muted); color: var(--text-color); box-shadow: none; }
        .comment-actions button { border-radius: var(--border-radius-xs); }
        .reaction-pill.user-reacted { background: var(--ink); color: var(--on-ink); }

        /* Article toolbar and reading progress. */
        .reading-progress__fill { background: var(--accent-color); }
        .toolbar-btn { border-radius: var(--border-radius-sm); font-weight: 500; }
        .toolbar-btn:hover { color: var(--text-color); border-color: var(--text-color); transform: none; }
        .toolbar-btn.is-active { background: var(--ink); color: var(--on-ink); border-color: var(--ink); }
        .read-time-badge i { color: var(--accent-color); }

        .recent-item { border-radius: var(--border-radius-lg); }
        .recent-item:hover { border-color: var(--card-border-color); transform: translateY(-2px); }
        .recent-item__title { font-weight: 600; }

        /* Ad spaces: labelled, on the page's own surface, and collapsed when Google has no ad. */
        .ad-slot { margin: 32px 0; padding: 12px; border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); background: var(--card-bg); min-height: 132px; }
        .ad-slot-label { display: block; margin-bottom: 8px; font-size: 0.75rem; line-height: 1rem; color: var(--text-subtle-color); }
        .ad-slot:has(ins.adsbygoogle[data-ad-status="unfilled"]) { display: none; }

        /* Admin storage page. */
        .storage-meter { height: 8px; border-radius: var(--border-radius-pill); background: var(--surface-muted); overflow: hidden; margin: 12px 0 8px; }
        .storage-meter span { display: block; height: 100%; background: var(--accent-color); border-radius: inherit; }
        .storage-meter.is-high span { background: #DC2626; }
        .storage-stats { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin: 24px 0; }
        .storage-stat { padding: 16px; border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); background: var(--light-bg); }
        .storage-stat dt { font-size: 0.75rem; line-height: 1rem; font-weight: 500; color: var(--text-subtle-color); }
        .storage-stat dd { margin: 4px 0 0; font-size: 1rem; line-height: 1.5rem; font-weight: 500; }
        .storage-table { width: 100%; font-size: 0.875rem; line-height: 1.25rem; }
        .storage-table th { font-weight: 500; color: var(--text-subtle-color); }
        .storage-table th, .storage-table td { padding: 8px 12px; border-bottom: 1px solid var(--card-border-color); }
        .storage-table td.num, .storage-table th.num { text-align: right; font-variant-numeric: tabular-nums; }

        /* ==========================================================================
           PRINT
           ========================================================================== */
        @media print {
            .navbar-main, .offcanvas, footer, .admin-controls, .article-toolbar, .reading-progress,
            #alert-placeholder, .comment-section, .recent-strip, .modal, .offcanvas-backdrop,
            .bookmark-btn, .read-more, .pagination, .btn, .ad-slot { display: none !important; }
            body { padding-top: 0; background: #fff; color: #000; font-size: 1rem; }
            .article-full-content-wrapper { box-shadow: none; border: none; padding: 0; max-width: 100%; }
            .summary-box, .takeaways-box { border: 1px solid #ccc; background: none !important; break-inside: avoid; }
            .content-text, p { max-width: none; }
            a[href^="http"]::after { content: " (" attr(href) ")"; font-size: 0.75rem; color: #555; word-break: break-all; }
            .hero-image-wrap { max-height: 240px; break-inside: avoid; }
        }

        /* ==========================================================================
           RESPONSIVE
           ========================================================================== */
        @media (max-width: 991.98px) {
            .featured-story { flex-direction: column; }
            .featured-story-image { flex-basis: auto; width: 100%; min-height: 300px; transform: none; }
            .featured-story-content { flex-basis: auto; }
        }
        @media (max-width: 767.98px) {
            .navbar-center { display: none; }
            .navbar-left { flex-grow: 1; }
            .featured-story-image { min-height: 220px; }
            .community-hub-cta, .community-hub-cta .btn { width: 100%; }
        }
        @media (max-width: 575.98px) {
            body { padding-top: 88px; }
            .comment-replies { margin-left: 1.25rem; padding-left: 1rem; }
            .comment-replies.comment-replies-flat { margin-left: 0.85rem; padding-left: 0.75rem; }
            .comment-container { gap: 0.75rem; }
            .comment-avatar { width: 38px; height: 38px; }
            .comment-replies .comment-avatar { width: 33px; height: 33px; }
            .profile-avatar { width: 96px; height: 96px; font-size: 2.25rem; }
            .profile-stats { gap: 0.5rem; flex-wrap: wrap; }
            .auth-card { margin: 1rem auto; border: none; box-shadow: none; }
            .auth-body, .article-body { padding: 1.4rem; }
            #alert-placeholder { top: 84px; width: 94vw; }
        }
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body class="{% if request.cookies.get('darkMode') == 'enabled' %}dark-mode{% endif %} {% block body_class %}{% endblock %}">
    {# No saved choice yet: follow the device, before first paint, exactly like the landing page does. #}
    <script>try { if (!/(^|; )darkMode=/.test(document.cookie) && window.matchMedia('(prefers-color-scheme: dark)').matches) { document.body.classList.add('dark-mode'); } } catch (e) {}</script>

    <a class="visually-hidden-focusable" href="#main-content">Skip to main content</a>

    <header>
        <nav class="navbar-main" id="mainNavbar" aria-label="Main">
            <div>
                <div class="navbar-content-wrapper">
                    <div class="navbar-left">
                        <a class="navbar-brand-custom" href="{{ url_for('index') }}" aria-label="BrieflyAI, today's briefing">
                            <svg class="logo-mark" viewBox="0 0 32 32" width="24" height="24" aria-hidden="true" focusable="false"><rect class="logo-tile" x="1" y="1" width="30" height="30" rx="9"/><rect class="logo-line" x="8" y="8.5" width="16" height="3" rx="1.5"/><rect class="logo-line" x="8" y="14.5" width="11" height="3" rx="1.5"/><rect class="logo-gist" x="8" y="20.5" width="6" height="3" rx="1.5"/></svg>
                            <span>BrieflyAI</span>
                        </a>
                    </div>
                    <div class="navbar-center">
                        <form action="{{ url_for('search_results') }}" method="GET" class="search-container" id="navbarSearchForm">
                            <label for="navbarSearchInput" class="visually-hidden">Search news articles</label>
                            <input type="search" name="query" id="navbarSearchInput" class="form-control navbar-search" placeholder="Search the news" value="{{ request.args.get('query', '') }}" autocomplete="off">
                            <i class="ph ph-magnifying-glass search-icon" aria-hidden="true"></i>
                            <button type="button" class="search-clear-btn" aria-label="Clear search"><i class="ph ph-x" aria-hidden="true"></i></button>
                        </form>
                    </div>
                    <div class="navbar-right">
                        <div class="header-controls">
                            <button class="header-btn" type="button" data-bs-toggle="offcanvas" data-bs-target="#mainOffcanvas" aria-controls="mainOffcanvas" aria-label="Open menu">
                                <i class="ph ph-list" aria-hidden="true"></i>
                                <span class="d-none d-sm-inline">Menu</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    </header>

    <div class="offcanvas offcanvas-end" tabindex="-1" id="mainOffcanvas" aria-labelledby="mainOffcanvasLabel">
        <div class="offcanvas-header">
            <h2 class="offcanvas-title" id="mainOffcanvasLabel"><svg class="logo-mark" viewBox="0 0 32 32" width="24" height="24" aria-hidden="true" focusable="false"><rect class="logo-tile" x="1" y="1" width="30" height="30" rx="9"/><rect class="logo-line" x="8" y="8.5" width="16" height="3" rx="1.5"/><rect class="logo-line" x="8" y="14.5" width="11" height="3" rx="1.5"/><rect class="logo-gist" x="8" y="20.5" width="6" height="3" rx="1.5"/></svg><span>Menu</span></h2>
            <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close menu"></button>
        </div>
        <div class="offcanvas-body d-flex flex-column">

            <div class="sidebar-section d-md-none">
                <h6 class="sidebar-heading">Search</h6>
                <form action="{{ url_for('search_results') }}" method="GET" class="search-container offcanvas-search">
                    <label for="offcanvasSearchInput" class="visually-hidden">Search news articles</label>
                    <input type="search" name="query" id="offcanvasSearchInput" class="form-control navbar-search" placeholder="Search the news" value="{{ request.args.get('query', '') }}" autocomplete="off">
                    <i class="ph ph-magnifying-glass search-icon" aria-hidden="true"></i>
                </form>
            </div>

            <div class="sidebar-section">
                {% if session.user_id %}
                    <div class="dropdown">
                        <a href="#" class="sidebar-btn dropdown-toggle" id="offcanvasUserDropdown" data-bs-toggle="dropdown" aria-expanded="false">
                            <div class="sidebar-avatar" aria-hidden="true">{{ session.user_name[0]|upper }}</div>
                            <strong class="ms-3">{{ session.user_name|truncate(20) }}</strong>
                        </a>
                        <ul class="dropdown-menu shadow" aria-labelledby="offcanvasUserDropdown">
                            <li><a class="dropdown-item" href="{{ url_for('profile') }}"><i class="ph ph-identification-card me-2" aria-hidden="true"></i>Profile</a></li>
                            <li><a class="dropdown-item" href="{{ url_for('share_testimonial') }}"><i class="ph ph-chat-circle-text me-2" aria-hidden="true"></i>Share your experience</a></li>
                            {% if session.get('is_admin') %}<li><a class="dropdown-item" href="{{ url_for('admin_testimonials') }}"><i class="ph ph-quotes me-2" aria-hidden="true"></i>Review testimonials</a></li><li><a class="dropdown-item" href="{{ url_for('admin_storage') }}"><i class="ph ph-database me-2" aria-hidden="true"></i>Storage</a></li>{% endif %}
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item" href="{{ url_for('logout') }}"><i class="ph ph-sign-out me-2" aria-hidden="true"></i>Logout</a></li>
                        </ul>
                    </div>
                {% else %}
                    <a href="{{ url_for('login') }}" class="sidebar-btn">
                        <i class="ph ph-sign-in me-2" aria-hidden="true"></i> Sign in or create an account
                    </a>
                {% endif %}
            </div>

            <div class="sidebar-section">
                 <button class="sidebar-btn dark-mode-toggle w-100" type="button" aria-pressed="false">
                    <i class="ph ph-moon me-2" aria-hidden="true"></i> <span class="theme-text">Dark mode</span>
                 </button>
            </div>

            <hr class="my-2">

            <div class="sidebar-section flex-grow-1">
                <h6 class="sidebar-heading">Categories</h6>
                <ul class="nav nav-pills flex-column mb-auto">
                    {% for cat_item in categories %}
                        {% set cat_url_params = {'category_name': cat_item, 'page': 1} %}
                        {% if cat_item == 'All Articles' and selected_category == 'All Articles' and request.args.get('filter_date') %}
                            {% set _ = cat_url_params.update({'filter_date': request.args.get('filter_date')}) %}
                        {% endif %}
                        <li class="nav-item">
                            <a href="{{ url_for('index', **cat_url_params) }}" class="nav-link {% if selected_category == cat_item %}active{% endif %}" {% if selected_category == cat_item %}aria-current="page"{% endif %}>
                                <i class="ph ph-{% if cat_item == 'All Articles' %}globe-hemisphere-east{% elif cat_item == 'Popular Stories' %}fire{% elif cat_item == "Yesterday's Headlines" %}clock-counter-clockwise{% elif cat_item == 'Community Hub' %}users{% endif %} me-2" aria-hidden="true"></i>
                                {{ cat_item }}
                            </a>
                        </li>
                    {% endfor %}
                </ul>
            </div>

            <hr class="my-2">

            <div class="sidebar-section">
                <h6 class="sidebar-heading">Filter by date</h6>
                <form id="dateFilterForm" class="mt-2">
                    <label for="articleDateFilter" class="visually-hidden">Filter articles by date</label>
                    <div class="input-group">
                        <input type="date" id="articleDateFilter" class="form-control" title="Filter 'All Articles' by date" value="{{ current_filter_date | default('', true) }}">
                        <button class="btn btn-primary" type="submit" title="Apply Date Filter">Go</button>
                    </div>
                     {% if current_filter_date %}
                        <button class="btn btn-sm btn-outline-danger mt-2 w-100" type="button" id="clearDateFilter">Clear date filter</button>
                    {% endif %}
                </form>
            </div>
        </div>
    </div>

    <div class="modal fade" id="confirmActionModal" tabindex="-1" aria-hidden="true" aria-labelledby="confirmActionModalLabel">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header border-0 pb-0">
                    <h2 class="modal-title h5" id="confirmActionModalLabel">Please confirm</h2>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p class="mb-0" id="confirmActionModalBody">Are you sure?</p>
                </div>
                <div class="modal-footer border-0 pt-0">
                    <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-danger" id="confirmActionModalConfirm">Confirm</button>
                </div>
            </div>
        </div>
    </div>

    <div class="modal fade" id="shortcutsModal" tabindex="-1" aria-hidden="true" aria-labelledby="shortcutsModalLabel">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header border-0 pb-2">
                    <h2 class="modal-title h5" id="shortcutsModalLabel"><i class="ph ph-keyboard me-2" aria-hidden="true"></i>Keyboard shortcuts</h2>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <dl class="kbd-list mb-0">
                        <dt>Focus search</dt><dd><kbd>/</kbd></dd>
                        <dt>Open menu</dt><dd><kbd>m</kbd></dd>
                        <dt>Toggle dark mode</dt><dd><kbd>d</kbd></dd>
                        <dt>Go to homepage</dt><dd><kbd>g</kbd> <kbd>h</kbd></dd>
                        <dt>Go to profile</dt><dd><kbd>g</kbd> <kbd>p</kbd></dd>
                        <dt>Bookmark this article</dt><dd><kbd>b</kbd></dd>
                        <dt>Jump to comments</dt><dd><kbd>c</kbd></dd>
                        <dt>Back to top</dt><dd><kbd>t</kbd></dd>
                        <dt>Show this help</dt><dd><kbd>?</kbd></dd>
                    </dl>
                </div>
            </div>
        </div>
    </div>

    <div id="alert-placeholder">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="alert alert-{{ category }} alert-dismissible fade show alert-top" role="alert">
                    <i class="ph {{ {'success': 'ph-check-circle', 'danger': 'ph-warning-circle', 'warning': 'ph-warning'}.get(category, 'ph-info') }} alert-icon" aria-hidden="true"></i>
                    <span>{{ message }}</span>
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
    </div>

    <main class="container main-content my-4" id="main-content">
        {% block content %}{% endblock %}
    </main>

    {% if session.user_id %}
    <div class="admin-controls">
        <button class="add-article-btn" data-bs-toggle="modal" data-bs-target="#addArticleModal" title="Post a New Article" aria-label="Post a new article">
            <i class="ph ph-note-pencil" aria-hidden="true"></i>
        </button>
    </div>
    <div class="modal fade" id="addArticleModal" tabindex="-1" aria-hidden="true" aria-labelledby="addArticleModalLabel">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content p-4">
                <div class="modal-header border-0 pb-0">
                    <h2 class="modal-title h4" id="addArticleModalLabel">Post a new article</h2>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p class="small text-muted d-flex align-items-center gap-2 mb-3" id="draftStatus" hidden>
                        <i class="ph ph-cloud-arrow-up" aria-hidden="true"></i><span id="draftStatusText">Draft saved</span>
                        <button type="button" class="link-btn ms-auto" id="discardDraftBtn">Discard draft</button>
                    </p>
                    <form id="addArticleForm" action="{{ url_for('post_article') }}" method="POST">
                        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                        <div class="mb-3"><label for="articleTitle" class="form-label">Article title</label><input type="text" id="articleTitle" name="title" class="form-control" required></div>
                        <div class="mb-3"><label for="articleDescription" class="form-label">Short description</label><textarea id="articleDescription" name="description" class="form-control" rows="3" required></textarea></div>
                        <div class="mb-3"><label for="articleSource" class="form-label">Source name</label><input type="text" id="articleSource" name="sourceName" class="form-control" value="Community Post" required></div>
                        <div class="mb-3"><label for="articleImage" class="form-label">Image URL (optional)</label><input type="url" id="articleImage" name="imageUrl" class="form-control"></div>
                        <div class="mb-3"><label for="articleContent" class="form-label">Full article content</label><textarea id="articleContent" name="content" class="form-control" rows="7" required></textarea></div>
                        <div class="d-flex justify-content-end gap-2 mt-4"><button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">Cancel</button><button type="submit" class="btn btn-primary">Post article</button></div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <footer class="site-footer">
        <div class="photo" aria-hidden="true">
            <img src="{{ url_for('static', filename='landing/bombay-harbour-1200.webp') }}"
                 srcset="{{ url_for('static', filename='landing/bombay-harbour-1200.webp') }} 1200w, {{ url_for('static', filename='landing/bombay-harbour-2400.webp') }} 2400w"
                 sizes="100vw" width="2400" height="1241" loading="lazy" decoding="async" alt="">
        </div>
        <div class="container">
            <div class="footer-panel">
                <div class="footer-top">
                    <div>
                        <a class="footer-brand" href="{{ url_for('landing') }}"><svg class="logo-mark" viewBox="0 0 32 32" width="24" height="24" aria-hidden="true" focusable="false"><rect class="logo-tile" x="1" y="1" width="30" height="30" rx="9"/><rect class="logo-line" x="8" y="8.5" width="16" height="3" rx="1.5"/><rect class="logo-line" x="8" y="14.5" width="11" height="3" rx="1.5"/><rect class="logo-gist" x="8" y="20.5" width="6" height="3" rx="1.5"/></svg><span>BrieflyAI</span></a>
                        <p class="footer-blurb">Short AI summaries and key takeaways for India's top stories.</p>
                        <form action="{{ url_for('subscribe') }}" method="POST" class="footer-newsletter">
                            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                            <label for="footerNewsletterEmail" class="footer-label">Newsletter</label>
                            <div class="input-group">
                                <input type="email" id="footerNewsletterEmail" name="email" class="form-control footer-newsletter-input" placeholder="Your email" autocomplete="email" required>
                                <button class="btn btn-primary" type="submit">Subscribe</button>
                            </div>
                        </form>
                    </div>
                    <nav class="footer-col" aria-label="Read">
                        <h2>Read</h2>
                        <ul>
                            <li><a href="{{ url_for('index') }}">Today's briefing</a></li>
                            {% for cat_item in categories %}{% if cat_item != 'All Articles' %}<li><a href="{{ url_for('index', category_name=cat_item, page=1) }}">{{ cat_item }}</a></li>{% endif %}{% endfor %}
                            <li><a href="{{ url_for('rss_feed') }}">RSS feed</a></li>
                        </ul>
                    </nav>
                    <nav class="footer-col" aria-label="BrieflyAI">
                        <h2>BrieflyAI</h2>
                        <ul>
                            <li><a href="{{ url_for('landing') }}">Home</a></li>
                            <li><a href="{{ url_for('about') }}">About</a></li>
                            <li><a href="{{ url_for('contact') }}">Contact</a></li>
                            <li><a href="{{ url_for('privacy') }}">Privacy policy</a></li>
                            <li><a href="{{ url_for('terms') }}">Terms of service</a></li>
                            {% if session.user_id %}<li><a href="{{ url_for('profile') }}">My profile</a></li>{% endif %}
                        </ul>
                    </nav>
                </div>
                <div class="footer-bottom">
                    <span>&copy; {{ current_year }} BrieflyAI. Made in India. Summaries are AI generated; always check the original. Photo: Bombay harbour, c. 1890, Library of Congress.</span>
                    <button type="button" class="back-to-top-link" id="backToTopBtn">Back to top <i class="ph ph-arrow-up" aria-hidden="true"></i></button>
                </div>
            </div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" defer></script>
    <script>
    window.BrieflyAI = window.BrieflyAI || {};
    BrieflyAI.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* --- CSRF ---------------------------------------------------------------
       Every state-changing request must carry the session's CSRF token, or the
       server rejects it. postJSON() is the single place that guarantees this. */
    BrieflyAI.csrfToken = (function () {
        var meta = document.querySelector('meta[name="csrf-token"]');
        return meta ? meta.getAttribute('content') : '';
    })();

    BrieflyAI.postJSON = function (url, body, options) {
        options = options || {};
        return fetch(url, {
            method: options.method || 'POST',
            credentials: 'same-origin',
            headers: Object.assign({
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-CSRFToken': BrieflyAI.csrfToken
            }, options.headers || {}),
            body: body === undefined ? undefined : JSON.stringify(body)
        });
    };

    BrieflyAI.debounce = function (fn, wait) {
        wait = wait || 300;
        var t;
        return function () {
            var args = arguments, ctx = this;
            clearTimeout(t);
            t = setTimeout(function () { fn.apply(ctx, args); }, wait);
        };
    };

    BrieflyAI.showToast = function (message, type, timeout) {
        type = type || 'info';
        timeout = timeout || 5000;
        var placeholder = document.getElementById('alert-placeholder');
        if (!placeholder) { return null; }
        var icons = { success: 'ph-check-circle', danger: 'ph-warning-circle', warning: 'ph-warning', info: 'ph-info' };
        var wrap = document.createElement('div');
        wrap.className = 'alert alert-' + type + ' alert-dismissible fade show alert-top';
        wrap.setAttribute('role', 'alert');
        var icon = document.createElement('i');
        icon.className = 'ph ' + (icons[type] || icons.info) + ' alert-icon';
        icon.setAttribute('aria-hidden', 'true');
        var text = document.createElement('span');
        text.textContent = message;
        var closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'btn-close';
        closeBtn.setAttribute('data-bs-dismiss', 'alert');
        closeBtn.setAttribute('aria-label', 'Close');
        wrap.appendChild(icon); wrap.appendChild(text); wrap.appendChild(closeBtn);
        placeholder.appendChild(wrap);
        setTimeout(function () {
            if (!document.body.contains(wrap)) { return; }
            if (window.bootstrap && bootstrap.Alert) { bootstrap.Alert.getOrCreateInstance(wrap).close(); }
            else { wrap.remove(); }
        }, timeout);
        return wrap;
    };

    BrieflyAI.confirmAction = function (options) {
        options = options || {};
        return new Promise(function (resolve) {
            var modalEl = document.getElementById('confirmActionModal');
            if (!modalEl || typeof bootstrap === 'undefined') { resolve(window.confirm(options.message || 'Are you sure?')); return; }
            modalEl.querySelector('#confirmActionModalLabel').textContent = options.title || 'Please confirm';
            modalEl.querySelector('#confirmActionModalBody').textContent = options.message || 'Are you sure?';
            var confirmBtn = modalEl.querySelector('#confirmActionModalConfirm');
            confirmBtn.textContent = options.confirmText || 'Confirm';
            confirmBtn.className = 'btn ' + (options.danger === false ? 'btn-primary' : 'btn-danger');
            var modal = bootstrap.Modal.getOrCreateInstance(modalEl);
            var decided = false;
            function cleanup() {
                confirmBtn.removeEventListener('click', onConfirm);
                modalEl.removeEventListener('hidden.bs.modal', onHidden);
            }
            function onConfirm() { decided = true; modal.hide(); resolve(true); cleanup(); }
            function onHidden() { if (!decided) { resolve(false); } cleanup(); }
            confirmBtn.addEventListener('click', onConfirm);
            modalEl.addEventListener('hidden.bs.modal', onHidden);
            modal.show();
        });
    };

    BrieflyAI.initImageLoadStates = function (root) {
        (root || document).querySelectorAll('.article-image, .hero-image').forEach(function (img) {
            if (img.dataset.loadStateInit) { return; }
            img.dataset.loadStateInit = '1';
            var wrap = img.closest('.article-image-container, .hero-image-wrap');
            var markLoaded = function () { img.classList.add('is-loaded'); if (wrap) { wrap.classList.add('is-loaded'); } };
            var markFallback = function () { if (wrap) { wrap.classList.add('img-fallback'); } };
            if (img.complete) {
                if (img.naturalWidth > 0) { markLoaded(); } else { markFallback(); }
            } else {
                img.addEventListener('load', markLoaded, { once: true });
                img.addEventListener('error', markFallback, { once: true });
            }
        });
    };

    /* Scroll-triggered reveal. Elements only get the hiding attribute once JS is running,
       so users without JS (or with reduced motion) always see fully visible content. */
    BrieflyAI.initScrollReveal = function (root) {
        if (BrieflyAI.reducedMotion || !('IntersectionObserver' in window)) { return; }
        var selector = '.article-card, .ai-synthesis-card, .featured-story, .community-hub-header, .state-card, .contact-card, .profile-header-card, .summary-box, .takeaways-box, .list-page-header';
        var items = Array.prototype.slice.call((root || document).querySelectorAll(selector));
        if (!items.length) { return; }

        if (!BrieflyAI._revealObserver) {
            BrieflyAI._revealObserver = new IntersectionObserver(function (entries, obs) {
                entries.forEach(function (entry) {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('is-revealed');
                        obs.unobserve(entry.target);
                    }
                });
            }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
        }

        items.forEach(function (el, i) {
            if (el.dataset.reveal !== undefined) { return; }
            // Hand over from the CSS entrance animation so the two don't fight.
            el.classList.remove('animate-fade-in');
            el.style.animation = 'none';
            el.dataset.reveal = '';
            el.style.setProperty('--reveal-delay', Math.min(i % 6, 5) * 55 + 'ms');
            var rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight * 0.95) {
                requestAnimationFrame(function () { el.classList.add('is-revealed'); });
            } else {
                BrieflyAI._revealObserver.observe(el);
            }
        });
    };

    /* Subtle parallax on the featured story image. Desktop + motion-allowed only. */
    BrieflyAI.initParallax = function () {
        var layer = document.querySelector('.featured-story-image');
        if (!layer || BrieflyAI.reducedMotion || window.innerWidth < 992) { return; }
        var ticking = false;
        function update() {
            var rect = layer.getBoundingClientRect();
            if (rect.bottom > 0 && rect.top < window.innerHeight) {
                var progress = (rect.top + rect.height / 2 - window.innerHeight / 2) / window.innerHeight;
                layer.style.setProperty('--parallax-y', (progress * -26).toFixed(1) + 'px');
            }
            ticking = false;
        }
        window.addEventListener('scroll', function () {
            if (!ticking) { ticking = true; requestAnimationFrame(update); }
        }, { passive: true });
        update();
    };

    /* Analytics is deferred until the page is idle so it never competes with content.
       AdSense is not loaded here: it sits in <head> on reading pages only, where Google's
       crawler can see it and where ads are allowed (never on sign in or error pages). */
    BrieflyAI.loadDeferredScripts = function () {
        if (BrieflyAI._deferredLoaded) { return; }
        BrieflyAI._deferredLoaded = true;
        {% if ga_measurement_id %}
        window.dataLayer = window.dataLayer || [];
        window.gtag = function () { window.dataLayer.push(arguments); };
        window.gtag('js', new Date());
        window.gtag('config', '{{ ga_measurement_id }}');
        var s = document.createElement('script');
        s.src = 'https://www.googletagmanager.com/gtag/js?id={{ ga_measurement_id }}'; s.async = true;
        document.head.appendChild(s);
        {% endif %}
    };

    /* --- Sharing ------------------------------------------------------------
       Uses the native share sheet where available (mobile), falls back to a
       menu of per-network intent URLs plus copy-to-clipboard. */
    BrieflyAI.share = {
        targets: function (url, title) {
            var u = encodeURIComponent(url), t = encodeURIComponent(title || '');
            return {
                whatsapp: 'https://wa.me/?text=' + t + '%20' + u,
                x: 'https://twitter.com/intent/tweet?url=' + u + '&text=' + t,
                facebook: 'https://www.facebook.com/sharer/sharer.php?u=' + u,
                linkedin: 'https://www.linkedin.com/sharing/share-offsite/?url=' + u
            };
        },
        native: function (url, title, text) {
            if (!navigator.share) { return Promise.reject(new Error('unsupported')); }
            return navigator.share({ title: title, text: text || title, url: url });
        },
        copy: function (url) {
            if (navigator.clipboard && window.isSecureContext) {
                return navigator.clipboard.writeText(url);
            }
            // Fallback for non-HTTPS / older browsers.
            return new Promise(function (resolve, reject) {
                try {
                    var ta = document.createElement('textarea');
                    ta.value = url;
                    ta.setAttribute('readonly', '');
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    document.body.appendChild(ta);
                    ta.select();
                    var ok = document.execCommand('copy');
                    document.body.removeChild(ta);
                    ok ? resolve() : reject(new Error('copy failed'));
                } catch (e) { reject(e); }
            });
        }
    };

    /* --- Recently viewed (per-browser, never leaves the device) -------------- */
    BrieflyAI.recent = {
        KEY: 'brieflyai:recent',
        MAX: 8,
        read: function () {
            try {
                var raw = localStorage.getItem(this.KEY);
                var list = raw ? JSON.parse(raw) : [];
                return Array.isArray(list) ? list : [];
            } catch (e) { return []; }
        },
        add: function (item) {
            if (!item || !item.url || !item.title) { return; }
            try {
                var list = this.read().filter(function (x) { return x.url !== item.url; });
                list.unshift({ url: item.url, title: item.title, source: item.source || '' });
                localStorage.setItem(this.KEY, JSON.stringify(list.slice(0, this.MAX)));
            } catch (e) { /* storage unavailable -- feature simply does nothing */ }
        },
        clear: function () {
            try { localStorage.removeItem(this.KEY); } catch (e) {}
        }
    };

    /* --- Keyboard shortcuts -------------------------------------------------- */
    BrieflyAI.initShortcuts = function () {
        var pendingG = false, gTimer = null;
        function isTyping(el) {
            if (!el) { return false; }
            var tag = el.tagName;
            return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
        }
        document.addEventListener('keydown', function (e) {
            if (e.ctrlKey || e.metaKey || e.altKey) { return; }
            if (isTyping(document.activeElement)) { return; }
            // Don't hijack keys while a dialog is open.
            if (document.querySelector('.modal.show') && e.key !== '?') { return; }

            var k = e.key;

            if (pendingG) {
                pendingG = false;
                clearTimeout(gTimer);
                if (k === 'h') { e.preventDefault(); window.location.href = "{{ url_for('index') }}"; return; }
                if (k === 'p') { e.preventDefault(); window.location.href = "{{ url_for('profile') if session.user_id else url_for('login') }}"; return; }
            }

            switch (k) {
                case '/':
                    e.preventDefault();
                    var search = document.getElementById('navbarSearchInput');
                    if (search && search.offsetParent !== null) { search.focus(); search.select(); }
                    else { document.querySelector('[data-bs-target="#mainOffcanvas"]').click(); }
                    break;
                case '?':
                    e.preventDefault();
                    if (window.bootstrap) { bootstrap.Modal.getOrCreateInstance(document.getElementById('shortcutsModal')).show(); }
                    break;
                case 'm':
                    e.preventDefault();
                    document.querySelector('[data-bs-target="#mainOffcanvas"]').click();
                    break;
                case 'd':
                    e.preventDefault();
                    var t = document.querySelector('.dark-mode-toggle');
                    if (t) { t.click(); }
                    break;
                case 't':
                    e.preventDefault();
                    window.scrollTo({ top: 0, behavior: BrieflyAI.reducedMotion ? 'auto' : 'smooth' });
                    break;
                case 'b':
                    var bm = document.getElementById('bookmarkBtn');
                    if (bm) { e.preventDefault(); bm.click(); }
                    break;
                case 'c':
                    var cs = document.getElementById('comment-section');
                    if (cs) { e.preventDefault(); cs.scrollIntoView({ behavior: BrieflyAI.reducedMotion ? 'auto' : 'smooth' }); }
                    break;
                case 'g':
                    pendingG = true;
                    gTimer = setTimeout(function () { pendingG = false; }, 1200);
                    break;
            }
        });
    };

    document.addEventListener('DOMContentLoaded', function () {
        try {
            BrieflyAI.initImageLoadStates();
            BrieflyAI.initScrollReveal();
            BrieflyAI.initParallax();
            BrieflyAI.initShortcuts();

            var navbar = document.getElementById('mainNavbar');
            if (navbar) {
                var navTicking = false;
                var syncNav = function () {
                    navbar.classList.toggle('is-scrolled', window.scrollY > 12);
                    navTicking = false;
                };
                window.addEventListener('scroll', function () {
                    if (!navTicking) { navTicking = true; requestAnimationFrame(syncNav); }
                }, { passive: true });
                syncNav();
            }

            const darkModeToggle = document.querySelector('.dark-mode-toggle');
            if (darkModeToggle) {
                const body = document.body;
                const updateThemeUI = () => {
                    const isDarkMode = body.classList.contains('dark-mode');
                    const themeIcon = darkModeToggle.querySelector('i');
                    const themeText = darkModeToggle.querySelector('.theme-text');
                    if (themeIcon) { themeIcon.className = isDarkMode ? 'ph ph-sun me-2' : 'ph ph-moon me-2'; }
                    if (themeText) { themeText.textContent = isDarkMode ? 'Light mode' : 'Dark mode'; }
                    darkModeToggle.setAttribute('aria-pressed', isDarkMode ? 'true' : 'false');
                };
                const applyTheme = (theme) => {
                    body.classList.toggle('dark-mode', theme === 'enabled');
                    try {
                        localStorage.setItem('darkMode', theme);
                        document.cookie = "darkMode=" + theme + ";path=/;max-age=31536000;SameSite=Lax";
                    } catch (storageErr) {
                        console.warn("Could not persist theme preference:", storageErr);
                    }
                    updateThemeUI();
                };
                darkModeToggle.addEventListener('click', () => {
                    applyTheme(body.classList.contains('dark-mode') ? 'disabled' : 'enabled');
                });
                // First visit only: follow the OS setting rather than defaulting to light.
                try {
                    if (!localStorage.getItem('darkMode')) {
                        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
                            applyTheme('enabled');
                        }
                    }
                } catch (e) { /* storage blocked - keep the server-rendered theme */ }

                updateThemeUI();
            }

            // Draft autosave for the article composer, so a mis-click or refresh
            // doesn't discard a long post.
            (function () {
                var form = document.getElementById('addArticleForm');
                var statusEl = document.getElementById('draftStatus');
                if (!form || !statusEl) { return; }
                var KEY = 'brieflyai:articleDraft';
                var fields = ['title', 'description', 'sourceName', 'imageUrl', 'content'];
                var statusText = document.getElementById('draftStatusText');
                var submitted = false;

                function fieldEl(name) { return form.querySelector('[name="' + name + '"]'); }

                function restore() {
                    try {
                        var raw = localStorage.getItem(KEY);
                        if (!raw) { return; }
                        var draft = JSON.parse(raw);
                        var restoredAny = false;
                        fields.forEach(function (f) {
                            var el = fieldEl(f);
                            if (el && draft[f]) { el.value = draft[f]; restoredAny = true; }
                        });
                        if (restoredAny) {
                            statusEl.hidden = false;
                            statusText.textContent = 'Unsent draft restored';
                        }
                    } catch (e) { /* corrupt draft - ignore it */ }
                }

                var save = BrieflyAI.debounce(function () {
                    if (submitted) { return; }
                    var draft = {};
                    var hasContent = false;
                    fields.forEach(function (f) {
                        var el = fieldEl(f);
                        if (el && el.value.trim()) { draft[f] = el.value; hasContent = true; }
                    });
                    try {
                        if (hasContent) {
                            localStorage.setItem(KEY, JSON.stringify(draft));
                            statusEl.hidden = false;
                            statusText.textContent = 'Draft saved';
                        } else {
                            localStorage.removeItem(KEY);
                            statusEl.hidden = true;
                        }
                    } catch (e) { /* storage full or blocked */ }
                }, 700);

                fields.forEach(function (f) {
                    var el = fieldEl(f);
                    if (el) { el.addEventListener('input', save); }
                });

                form.addEventListener('submit', function () {
                    submitted = true;
                    try { localStorage.removeItem(KEY); } catch (e) {}
                });

                var discard = document.getElementById('discardDraftBtn');
                if (discard) {
                    discard.addEventListener('click', function () {
                        try { localStorage.removeItem(KEY); } catch (e) {}
                        fields.forEach(function (f) {
                            var el = fieldEl(f);
                            if (el) { el.value = el.name === 'sourceName' ? 'Community Post' : ''; }
                        });
                        statusEl.hidden = true;
                    });
                }

                restore();
            })();

            document.querySelectorAll('#alert-placeholder .alert').forEach(function (alert) {
                setTimeout(function () {
                    if (!document.body.contains(alert)) { return; }
                    if (window.bootstrap && bootstrap.Alert) { bootstrap.Alert.getOrCreateInstance(alert).close(); }
                    else { alert.remove(); }
                }, 7000);
            });

            const dateFilterForm = document.getElementById('dateFilterForm');
            if (dateFilterForm) {
                dateFilterForm.addEventListener('submit', function (event) {
                    event.preventDefault();
                    const dateInput = document.getElementById('articleDateFilter');
                    if (dateInput && dateInput.value) {
                       let targetUrl = new URL("{{ url_for('index', category_name='All Articles') }}", window.location.origin);
                       targetUrl.searchParams.set('filter_date', dateInput.value);
                       window.location.href = targetUrl.toString();
                    }
                });
                const clearDateFilterBtn = document.getElementById('clearDateFilter');
                if (clearDateFilterBtn) {
                    clearDateFilterBtn.addEventListener('click', function () {
                        window.location.href = "{{ url_for('index', category_name='All Articles') }}";
                    });
                }
            }

            document.querySelectorAll('.search-container').forEach(function (container) {
                const input = container.querySelector('input[type="search"]');
                const clearBtn = container.querySelector('.search-clear-btn');
                const form = container.tagName === 'FORM' ? container : container.closest('form');
                if (!input) { return; }
                const syncState = BrieflyAI.debounce(function () {
                    container.classList.toggle('has-value', input.value.trim().length > 0);
                }, 120);
                input.addEventListener('input', syncState);
                syncState();
                if (clearBtn) {
                    clearBtn.addEventListener('click', function () {
                        input.value = '';
                        container.classList.remove('has-value');
                        input.focus();
                    });
                }
                if (form) {
                    form.addEventListener('submit', function () {
                        if (!input.value.trim()) { return; }
                        const icon = container.querySelector('.search-icon');
                        if (icon) { icon.className = 'ph ph-circle-notch ph-spin search-icon'; }
                        input.setAttribute('aria-busy', 'true');
                    });
                }
            });

            const backToTopBtn = document.getElementById('backToTopBtn');
            if (backToTopBtn) {
                backToTopBtn.addEventListener('click', function () {
                    window.scrollTo({ top: 0, behavior: BrieflyAI.reducedMotion ? 'auto' : 'smooth' });
                });
            }
        } catch (e) {
            console.error("An error occurred in the base layout script:", e);
        }
    });

    window.addEventListener('load', function () {
        if ('requestIdleCallback' in window) { requestIdleCallback(BrieflyAI.loadDeferredScripts, { timeout: 3000 }); }
        else { setTimeout(BrieflyAI.loadDeferredScripts, 1800); }

        // Offline support. Registration failures are non-fatal by design.
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js').catch(function (err) {
                console.warn('Service worker registration failed:', err);
            });
        }
    });
    </script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
"""
_COMMENT_TEMPLATE = """
{% set depth = depth|default(0) %}
{% set has_reaction_data = comment_data is defined and comment.id in comment_data %}
{% set reactions = comment_data[comment.id].reactions if has_reaction_data else {} %}
{% set user_reaction = comment_data[comment.id].user_reaction if has_reaction_data else none %}
<div class="comment-thread" id="comment-{{ comment.id }}" data-depth="{{ depth }}">
    <div class="comment-container">
        <div class="comment-avatar" aria-hidden="true" title="{{ comment.author.name if comment.author else 'Unknown' }}">{{ (comment.author.name[0]|upper if comment.author and comment.author.name else 'U') }}</div>
        <div class="comment-body">
            <div class="comment-header">
                {% if comment.author %}
                <a href="{{ url_for('public_profile', username=comment.author.username) }}" class="comment-author text-decoration-none">{{ comment.author.name }}</a>
                {% else %}
                <span class="comment-author">Anonymous</span>
                {% endif %}
                <span class="comment-date">{{ comment.timestamp | to_ist }}</span>
            </div>

            <p class="comment-content mb-2">{{ comment.content }}</p>

            <div class="edit-form-container">
                <form class="edit-comment-form">
                    <label class="visually-hidden" for="edit-content-{{ comment.id }}">Edit your comment</label>
                    <textarea class="form-control form-control-sm mb-2" id="edit-content-{{ comment.id }}" name="content" rows="3" required>{{ comment.content }}</textarea>
                    <div class="d-flex justify-content-end gap-2">
                        <button type="button" class="btn btn-sm btn-outline-secondary cancel-edit-btn">Cancel</button>
                        <button type="submit" class="btn btn-sm btn-primary">Save changes</button>
                    </div>
                </form>
            </div>

            {% if session.user_id %}
            <div class="comment-actions">
                <div class="reaction-box" id="reaction-box-{{ comment.id }}" role="menu" aria-label="Choose a reaction">
                    {% for emoji, emoji_label in [('\U0001F44D','Like'), ('\u2764\uFE0F','Love'), ('\U0001F602','Laugh'), ('\U0001F62E','Wow'), ('\U0001F622','Sad'), ('\U0001F620','Angry')] %}
                        <button type="button" class="reaction-emoji {% if user_reaction == emoji %}is-selected{% endif %}" data-emoji="{{ emoji }}" data-comment-id="{{ comment.id }}" role="menuitem" title="{{ emoji_label }}" aria-label="React with {{ emoji_label }}">{{ emoji }}</button>
                    {% endfor %}
                </div>
                <button type="button" class="react-btn" data-comment-id="{{ comment.id }}" title="React" aria-haspopup="true" aria-expanded="false" aria-controls="reaction-box-{{ comment.id }}"><i class="ph ph-smiley" aria-hidden="true"></i> React</button>
                <button type="button" class="reply-btn" data-comment-id="{{ comment.id }}" title="Reply"><i class="ph ph-arrow-bend-up-left" aria-hidden="true"></i> Reply</button>

                {% if session.user_id == comment.user_id %}
                    <button type="button" class="edit-btn" data-comment-id="{{ comment.id }}" title="Edit"><i class="ph ph-pencil-simple" aria-hidden="true"></i> Edit</button>
                    <button type="button" class="delete-btn" data-comment-id="{{ comment.id }}" title="Delete"><i class="ph ph-trash" aria-hidden="true"></i> Delete</button>
                {% endif %}
            </div>
            <div class="reply-form-container" id="reply-form-container-{{ comment.id }}">
                <form class="reply-form">
                    <input type="hidden" name="parent_id" value="{{ comment.id }}">
                    <label class="visually-hidden" for="reply-content-{{ comment.id }}">Write a reply to {{ comment.author.name if comment.author else 'this comment' }}</label>
                    <div class="mb-2"><textarea class="form-control form-control-sm" id="reply-content-{{ comment.id }}" name="content" rows="2" placeholder="Write a reply..." required></textarea></div>
                    <div class="d-flex justify-content-end gap-2">
                        <button type="button" class="btn btn-sm btn-outline-secondary cancel-reply-btn">Cancel</button>
                        <button type="submit" class="btn btn-sm btn-primary">Post reply</button>
                    </div>
                </form>
            </div>
            {% endif %}

            <div class="reaction-summary" id="reaction-summary-{{ comment.id }}">
                {% for emoji, count in reactions.items() %}
                    {% if count and count > 0 %}
                    <div class="reaction-pill {% if user_reaction == emoji %}user-reacted{% endif %}" data-emoji="{{ emoji }}"><span class="emoji">{{ emoji }}</span><span class="count">{{ count }}</span></div>
                    {% endif %}
                {% endfor %}
            </div>
        </div>
    </div>
    <div class="comment-replies {% if depth >= 3 %}comment-replies-flat{% endif %}" id="replies-of-{{ comment.id }}">
        {% for comment in comment.replies %}
            {% set depth = depth + 1 %}
            {% include '_COMMENT_TEMPLATE' %}
        {% endfor %}
    </div>
</div>
"""
INDEX_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}
    {% if query %}Search: {{ query|truncate(30) }}
    {% elif selected_category == 'All Articles' and is_main_homepage == False %}All Articles
    {% elif selected_category and not is_main_homepage %}{{ selected_category }}
    {% else %}Popular & Latest News from India{% endif %} - BrieflyAI
{% endblock %}

{% block content %}

{# This is the main controller: It shows the full homepage, or the list view for categories. #}
{% if is_main_homepage %}

    {# ============== LAYOUT 1: MAIN HOMEPAGE (WITH ALL FEATURES) ============== #}
    <div class="animate-fade-in">

        {% if synthesis %}
        <section class="ai-synthesis-card" aria-labelledby="briefingHeading">
            <span class="photo-credit" aria-hidden="true">Darjeeling, c. 1890</span>
            <div class="synthesis-panel">
                <div class="synthesis-header">
                    <h2 class="eyebrow" id="briefingHeading">Today's briefing</h2>
                </div>
                <p class="synthesis-text">{{ synthesis }}</p>
                {% if keywords %}
                <div class="synthesis-keywords">
                    {% for keyword in keywords %}
                        <a href="{{ url_for('search_results', query=keyword) }}" class="keyword-tag">{{ keyword }}</a>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
        </section>
        {% endif %}

        {% if featured_article %}
        <article class="featured-story">
            <div class="featured-story-image" {% if featured_article.urlToImage %}style="background-image: url('{{ featured_article.urlToImage }}')"{% endif %} role="img" aria-label="{{ featured_article.title|truncate(80) }}"></div>
            <div class="featured-story-content">
                <div class="article-meta">
                    <span class="meta-item"><i class="ph ph-fire" aria-hidden="true"></i> Top story</span>
                    <span class="meta-item"><i class="ph ph-buildings" aria-hidden="true"></i> {{ featured_article.source.name|truncate(20) }}</span>
                </div>
                <h2><a href="{{ url_for('article_detail', article_hash_id=featured_article.id) }}">{{ featured_article.title }}</a></h2>
                <p class="description">{{ featured_article.description|truncate(150) }}</p>
                <a href="{{ url_for('article_detail', article_hash_id=featured_article.id) }}" class="read-more-btn">Read the full story <i class="ph ph-arrow-right ms-1" aria-hidden="true"></i></a>
            </div>
        </article>
        {% endif %}

        {% with ad_name='feed', ad_slot=ad_slots.get('feed') %}{% include "_AD_SLOT_TEMPLATE" %}{% endwith %}

        <section class="recent-strip" id="trendingStrip" hidden aria-labelledby="trendingHeading">
            <div class="recent-strip__head">
                <h2 class="section-heading h5 mb-0" id="trendingHeading"><i class="ph ph-trend-up me-2" aria-hidden="true"></i>Trending this week</h2>
            </div>
            <div class="recent-strip__list" id="trendingList"></div>
        </section>

        <section class="recent-strip" id="recentStrip" hidden aria-labelledby="recentStripHeading">
            <div class="recent-strip__head">
                <h2 class="section-heading h5 mb-0" id="recentStripHeading"><i class="ph ph-clock-counter-clockwise me-2" aria-hidden="true"></i>Pick up where you left off</h2>
                <button type="button" class="link-btn" id="clearRecentBtn">Clear</button>
            </div>
            <div class="recent-strip__list" id="recentStripList"></div>
        </section>

        <ul class="nav nav-tabs nav-fill mb-3" id="newsTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="popular-tab" data-bs-toggle="tab" data-bs-target="#popular-tab-pane" type="button" role="tab" aria-controls="popular-tab-pane" aria-selected="true">
                    <i class="ph ph-fire me-1" aria-hidden="true"></i> Popular stories
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="yesterday-tab" data-bs-toggle="tab" data-bs-target="#yesterday-tab-pane" type="button" role="tab" aria-controls="yesterday-tab-pane" aria-selected="false">
                    <i class="ph ph-clock-counter-clockwise me-1" aria-hidden="true"></i> Yesterday's headlines
                </button>
            </li>
        </ul>

        <div class="tab-content" id="newsTabContent">
            <div class="tab-pane fade show active" id="popular-tab-pane" role="tabpanel" aria-labelledby="popular-tab">
                <div class="row g-4 pt-3">
                    {% if popular_articles %}
                        {% for art in popular_articles %}
                            <div class="col-md-6 col-lg-4 d-flex">
                                <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ (loop.index0 * 0.05)|round(2) }}s">
                                    {% set article_url = url_for('article_detail', article_hash_id=art.id) %}
                                    <div class="article-image-container {% if not art.urlToImage %}img-fallback{% endif %}">
                                        <a href="{{ article_url }}" tabindex="-1" aria-hidden="true">{% if art.urlToImage %}<img src="{{ art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}" loading="{{ 'eager' if loop.index0 < 3 else 'lazy' }}" decoding="async">{% endif %}</a>
                                        <div class="img-fallback-icon" aria-hidden="true"><i class="ph ph-newspaper"></i></div>
                                    </div>
                                    <div class="article-body d-flex flex-column">
                                        <div class="d-flex justify-content-between align-items-start">
                                            <h3 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                                            {% if session.user_id %}<button type="button" class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" aria-label="{% if art.is_bookmarked %}Remove bookmark{% else %}Add bookmark{% endif %} for {{ art.title|truncate(50) }}" data-article-hash-id="{{ art.id }}" data-is-community="false" data-title="{{ art.title|e }}" data-source-name="{{ art.source.name|e }}" data-image-url="{{ (art.urlToImage or '')|e }}" data-description="{{ (art.description if art.description else '')|e }}" data-published-at="{{ (art.publishedAt if art.publishedAt else '')|e }}"><i class="ph-fill ph-bookmark-simple" aria-hidden="true"></i></button>{% endif %}
                                        </div>
                                        <div class="article-meta small mb-2">
                                            <span class="meta-item text-muted"><i class="ph ph-buildings" aria-hidden="true"></i> {{ art.source.name|truncate(20) }}</span>
                                            <span class="meta-item text-muted"><i class="ph ph-calendar-blank" aria-hidden="true"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                                        </div>
                                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read more <i class="ph ph-caret-right ms-1 small" aria-hidden="true"></i></a>
                                    </div>
                                </article>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="col-12">
                            <div class="state-card state-card-solid py-4">
                                <div class="state-card-icon state-card-icon-sm"><i class="ph ph-broadcast" aria-hidden="true"></i></div>
                                <p class="state-card-text mb-0">More popular stories are on their way &mdash; please check back shortly.</p>
                            </div>
                        </div>
                    {% endif %}
                </div>
                {% if popular_articles %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('index', category_name='Popular Stories') }}" class="btn btn-outline-primary">View all popular stories <i class="ph ph-arrow-right ms-1" aria-hidden="true"></i></a>
                </div>
                {% endif %}
            </div>
            <div class="tab-pane fade" id="yesterday-tab-pane" role="tabpanel" aria-labelledby="yesterday-tab">
                <div class="row g-4 pt-3">
                    {% if latest_yesterday_articles %}
                        {% for art in latest_yesterday_articles %}
                             <div class="col-md-6 col-lg-4 d-flex">
                                <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ (loop.index0 * 0.05)|round(2) }}s">
                                    {% set article_url = url_for('article_detail', article_hash_id=art.id) %}
                                    <div class="article-image-container {% if not art.urlToImage %}img-fallback{% endif %}">
                                        <a href="{{ article_url }}" tabindex="-1" aria-hidden="true">{% if art.urlToImage %}<img src="{{ art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}" loading="lazy" decoding="async">{% endif %}</a>
                                        <div class="img-fallback-icon" aria-hidden="true"><i class="ph ph-newspaper"></i></div>
                                    </div>
                                    <div class="article-body d-flex flex-column">
                                        <div class="d-flex justify-content-between align-items-start">
                                            <h3 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                                            {% if session.user_id %}<button type="button" class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" aria-label="{% if art.is_bookmarked %}Remove bookmark{% else %}Add bookmark{% endif %} for {{ art.title|truncate(50) }}" data-article-hash-id="{{ art.id }}" data-is-community="false" data-title="{{ art.title|e }}" data-source-name="{{ art.source.name|e }}" data-image-url="{{ (art.urlToImage or '')|e }}" data-description="{{ (art.description if art.description else '')|e }}" data-published-at="{{ (art.publishedAt if art.publishedAt else '')|e }}"><i class="ph-fill ph-bookmark-simple" aria-hidden="true"></i></button>{% endif %}
                                        </div>
                                        <div class="article-meta small mb-2">
                                            <span class="meta-item text-muted"><i class="ph ph-buildings" aria-hidden="true"></i> {{ art.source.name|truncate(20) }}</span>
                                            <span class="meta-item text-muted"><i class="ph ph-calendar-blank" aria-hidden="true"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                                        </div>
                                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read more <i class="ph ph-caret-right ms-1 small" aria-hidden="true"></i></a>
                                    </div>
                                </article>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="col-12">
                            <div class="state-card state-card-solid py-4">
                                <div class="state-card-icon state-card-icon-sm"><i class="ph ph-broadcast" aria-hidden="true"></i></div>
                                <p class="state-card-text mb-0">Yesterday's headlines couldn't be loaded right now &mdash; please check back shortly.</p>
                            </div>
                        </div>
                    {% endif %}
                </div>
                {% if latest_yesterday_articles %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('index', category_name="Yesterday's Headlines") }}" class="btn btn-outline-primary">View all of yesterday's headlines <i class="ph ph-arrow-right ms-1" aria-hidden="true"></i></a>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

{% else %}

    {# ============ LAYOUT 2: STANDARD PAGINATED LIST VIEW (RESTORED) ============ #}
    {# This block handles all other pages like categories, search, and date filters. #}

    {% if selected_category == 'All Articles' and current_filter_date %}
        <div class="list-page-header">
            <span class="eyebrow"><i class="ph ph-calendar-blank me-1" aria-hidden="true"></i> Filtered by date</span>
            <h1>{{ current_filter_date }}</h1>
        </div>
    {% elif selected_category == 'Community Hub' %}
        <div class="community-hub-header">
            <div class="community-hub-header-text">
                <span class="eyebrow"><i class="ph ph-users me-1" aria-hidden="true"></i> Community Hub</span>
                <h1>Stories from our readers</h1>
                <p class="community-hub-sub">Real perspectives, posted by the BrieflyAI community. Have something to say about today's news?</p>
            </div>
            {% if session.user_id %}
                <button type="button" class="btn btn-primary-modal community-hub-cta" data-bs-toggle="modal" data-bs-target="#addArticleModal"><i class="ph ph-note-pencil me-2" aria-hidden="true"></i>Share your story</button>
            {% else %}
                <a href="{{ url_for('login', next=request.full_path) }}" class="btn btn-outline-primary community-hub-cta"><i class="ph ph-sign-in me-2" aria-hidden="true"></i>Log in to post</a>
            {% endif %}
        </div>
    {% elif query %}
        <div class="list-page-header">
            <span class="eyebrow"><i class="ph ph-magnifying-glass me-1" aria-hidden="true"></i> Search results</span>
            <h1>&ldquo;{{ query|truncate(40) }}&rdquo;</h1>
        </div>
    {% elif selected_category != 'All Articles' %}
        <div class="list-page-header">
            <span class="eyebrow"><i class="ph ph-{% if selected_category == 'Popular Stories' %}fire{% elif selected_category == "Yesterday's Headlines" %}clock-counter-clockwise{% else %}stack{% endif %} me-1" aria-hidden="true"></i> Category</span>
            <h1>{{ selected_category }}</h1>
        </div>
    {% endif %}

    {% if articles and not is_main_homepage %} {# This section is for the paginated list view only #}
        <div class="row g-4">
            {% for art in articles %}
            <div class="col-md-6 col-lg-4 d-flex">
                <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ (loop.index0 * 0.05)|round(2) }}s">
                    {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
                    {% set img_src = art.image_url if art.is_community_article else art.urlToImage %}
                    <div class="article-image-container {% if not img_src %}img-fallback{% endif %}">
                        <a href="{{ article_url }}" tabindex="-1" aria-hidden="true">
                        {% if img_src %}<img src="{{ img_src }}" class="article-image" alt="{{ art.title|truncate(50) }}" loading="{{ 'eager' if loop.index0 < 3 else 'lazy' }}" decoding="async">{% endif %}</a>
                        <div class="img-fallback-icon" aria-hidden="true"><i class="ph ph-newspaper"></i></div>
                    </div>
                    <div class="article-body d-flex flex-column">
                        <div class="d-flex justify-content-between align-items-start">
                            <h3 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                            {% if session.user_id %}
                            <button type="button" class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}"
                                    title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}"
                                    aria-label="{% if art.is_bookmarked %}Remove bookmark{% else %}Add bookmark{% endif %} for {{ art.title|truncate(50) }}"
                                    data-article-hash-id="{{ art.article_hash_id if art.is_community_article else art.id }}"
                                    data-is-community="{{ 'true' if art.is_community_article else 'false' }}"
                                    data-title="{{ art.title|e }}"
                                    data-source-name="{{ (art.author.name if art.is_community_article and art.author else art.source.name)|e }}"
                                    data-image-url="{{ ((art.image_url if art.is_community_article else art.urlToImage) or '')|e }}"
                                    data-description="{{ (art.description if art.description else '')|e }}"
                                    data-published-at="{{ (art.published_at.isoformat() if art.is_community_article and art.published_at else (art.publishedAt if not art.is_community_article and art.publishedAt else ''))|e }}">
                                <i class="ph-fill ph-bookmark-simple" aria-hidden="true"></i>
                            </button>
                            {% endif %}
                        </div>
                        <div class="article-meta small mb-2">
                            <span class="meta-item text-muted"><i class="ph ph-{{ 'user-circle' if art.is_community_article else 'buildings' }}" aria-hidden="true"></i> {% if art.is_community_article and art.author %}<a href="{{ url_for('public_profile', username=art.author.username) }}" class="text-muted text-decoration-none">{{ art.author.name|truncate(20) }}</a>{% else %}{{ art.source.name|truncate(20) }}{% endif %}</span>
                            <span class="meta-item text-muted"><i class="ph ph-calendar-blank" aria-hidden="true"></i> {{ (art.published_at | to_ist if art.is_community_article else (art.publishedAt | to_ist if art.publishedAt else 'N/A')) }}</span>
                        </div>
                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read more <i class="ph ph-caret-right ms-1 small" aria-hidden="true"></i></a>
                    </div>
                </article>
            </div>
            {% endfor %}
        </div>
    {% elif not articles %}
        {% if query %}
        <div class="state-card">
            <div class="state-card-icon"><i class="ph ph-magnifying-glass" aria-hidden="true"></i></div>
            <h2 class="state-card-title">No results for &ldquo;{{ query|truncate(40) }}&rdquo;</h2>
            <p class="state-card-text">Try a different search term, or browse by category instead.</p>
            <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Browse all articles</a></div>
        </div>
        {% else %}
        <div class="state-card">
            <div class="state-card-icon"><i class="ph ph-newspaper" aria-hidden="true"></i></div>
            <h2 class="state-card-title">No articles here yet</h2>
            <p class="state-card-text">Try a different category, or check back again soon.</p>
            <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Back to homepage</a></div>
        </div>
        {% endif %}
    {% endif %}

    {% with ad_name='list', ad_slot=ad_slots.get('list') %}{% include "_AD_SLOT_TEMPLATE" %}{% endwith %}

    {% if total_pages and total_pages > 1 %}
    <nav aria-label="Page navigation" class="mt-5"><ul class="pagination justify-content-center">
        {% set filter_date_for_url = request.args.get('filter_date') if selected_category == 'All Articles' and request.args.get('filter_date') else None %}
        <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}">
            <a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) if current_page > 1 else '#' }}" {% if current_page == 1 %}aria-disabled="true" tabindex="-1"{% endif %}>&laquo; Prev</a>
        </li>
        {% set page_window = 1 %}{% set show_first = 1 %}{% set show_last = total_pages %}
        {% if current_page - page_window > show_first %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">1</a></li>{% if current_page - page_window > show_first + 1 %}<li class="page-item disabled"><span class="page-link">&hellip;</span></li>{% endif %}{% endif %}
        {% for p in range(1, total_pages + 1) %}{% if p == current_page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% elif p >= current_page - page_window and p <= current_page + page_window %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">{{ p }}</a></li>{% endif %}{% endfor %}
        {% if current_page + page_window < show_last %}{% if current_page + page_window < show_last - 1 %}<li class="page-item disabled"><span class="page-link">&hellip;</span></li>{% endif %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=total_pages, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">{{ total_pages }}</a></li>{% endif %}
        <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}">
            <a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) if current_page < total_pages else '#' }}" {% if current_page == total_pages %}aria-disabled="true" tabindex="-1"{% endif %}>Next &raquo;</a>
        </li>
    </ul></nav>
    {% endif %}
{% endif %}
{% endblock %}

{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    /* --- Recently viewed strip (reads local-only history; renders nothing if empty) --- */
    (function () {
        var strip = document.getElementById('recentStrip');
        var list = document.getElementById('recentStripList');
        if (!strip || !list || !window.BrieflyAI || !BrieflyAI.recent) { return; }

        function render() {
            var items = BrieflyAI.recent.read();
            if (!items.length) { strip.hidden = true; return; }
            list.textContent = '';
            items.forEach(function (item) {
                var a = document.createElement('a');
                a.className = 'recent-item';
                a.href = item.url;
                var h3 = document.createElement('h3');
                h3.className = 'recent-item__title';
                h3.textContent = item.title;          // textContent, so stored titles can't inject markup
                var span = document.createElement('span');
                span.className = 'recent-item__source';
                span.textContent = item.source || '';
                a.appendChild(h3);
                a.appendChild(span);
                list.appendChild(a);
            });
            strip.hidden = false;
            if (BrieflyAI.initScrollReveal) { BrieflyAI.initScrollReveal(strip); }
        }

        var clearBtn = document.getElementById('clearRecentBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', function () {
                BrieflyAI.recent.clear();
                render();
                BrieflyAI.showToast('Reading history cleared.', 'success', 2500);
            });
        }
        render();
    })();

    /* --- Trending strip (hidden entirely when there is nothing to show) --- */
    (function () {
        var strip = document.getElementById('trendingStrip');
        var list = document.getElementById('trendingList');
        if (!strip || !list) { return; }
        fetch('{{ url_for("trending") }}', { credentials: 'same-origin' })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                if (!data || !data.success || !data.articles || !data.articles.length) { return; }
                data.articles.forEach(function (item) {
                    var a = document.createElement('a');
                    a.className = 'recent-item';
                    a.href = item.url;
                    var h3 = document.createElement('h3');
                    h3.className = 'recent-item__title';
                    h3.textContent = item.title;          // textContent: never trust server text as markup
                    var span = document.createElement('span');
                    span.className = 'recent-item__source';
                    span.textContent = item.source + ' \u00b7 ' + item.views + (item.views === 1 ? ' view' : ' views');
                    a.appendChild(h3); a.appendChild(span);
                    list.appendChild(a);
                });
                strip.hidden = false;
                if (BrieflyAI.initScrollReveal) { BrieflyAI.initScrollReveal(strip); }
            })
            .catch(function () { /* trending is a nicety; stay silent on failure */ });
    })();

    const isUserLoggedInForHomepage = {{ 'true' if session.user_id else 'false' }};
    document.querySelectorAll('.homepage-bookmark-btn').forEach(button => {
        if (isUserLoggedInForHomepage) {
            button.addEventListener('click', function(event) {
                event.preventDefault(); event.stopPropagation();
                const articleHashId = this.dataset.articleHashId;
                const isCommunity = this.dataset.isCommunity;
                const title = this.dataset.title;
                const sourceName = this.dataset.sourceName;
                const imageUrl = this.dataset.imageUrl;
                const description = this.dataset.description;
                const publishedAt = this.dataset.publishedAt;
                const btnRef = this;
                BrieflyAI.postJSON(
                    `{{ url_for('toggle_bookmark', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashId),
                    { is_community_article: isCommunity, title: title, source_name: sourceName, image_url: imageUrl, description: description, published_at: publishedAt }
                )
                .then(res => { if (!res.ok) { return res.json().then(err => { throw new Error(err.error || `HTTP error! status: ${res.status}`); }); } return res.json(); })
                .then(data => {
                    if (data.success) {
                        const nowActive = data.status === 'added';
                        btnRef.classList.toggle('active', nowActive);
                        btnRef.title = nowActive ? 'Remove Bookmark' : 'Add Bookmark';
                        btnRef.setAttribute('aria-label', (nowActive ? 'Remove bookmark' : 'Add bookmark') + ' for ' + title);
                        btnRef.classList.remove('is-popping');
                        void btnRef.offsetWidth;
                        btnRef.classList.add('is-popping');
                        BrieflyAI.showToast(data.message, 'success', 3000);
                    } else {
                        BrieflyAI.showToast(data.error || 'Could not update bookmark.', 'danger');
                    }
                })
                .catch(err => { console.error("Bookmark error on homepage:", err); BrieflyAI.showToast("Could not update bookmark: " + err.message, 'danger'); });
            });
        }
    });
});
</script>
{% endblock %}
"""
ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) if article else "Article" }} - BrieflyAI{% endblock %}

{# Jinja hoists block definitions and gives each its own scope, so the values are
   recomputed inside every block rather than shared via a top-level {% set %}. #}
{% block meta_description %}{% if article %}{{ ((article.groq_summary if is_community_article and article.groq_summary else article.description) or 'AI-summarized news from BrieflyAI.')|striptags|truncate(155) }}{% else %}AI-summarized, India-centric news.{% endif %}{% endblock %}
{% block og_type %}article{% endblock %}
{% block og_title %}{% if article %}{{ article.title }}{% else %}Article not found{% endif %}{% endblock %}
{% block og_description %}{% if article %}{{ ((article.groq_summary if is_community_article and article.groq_summary else article.description) or 'AI-summarized news from BrieflyAI.')|striptags|truncate(200) }}{% else %}AI-summarized, India-centric news.{% endif %}{% endblock %}
{% block og_image %}
    {% if article %}
        {% set art_image = article.image_url if is_community_article else article.urlToImage %}
        {% if art_image %}
        <meta property="og:image" content="{{ art_image }}">
        <meta name="twitter:image" content="{{ art_image }}">
        <meta property="og:image:alt" content="{{ article.title|truncate(100) }}">
        {% endif %}
    {% endif %}
{% endblock %}
{% block head_extra %}
{% if article %}
{% set art_image = article.image_url if is_community_article else article.urlToImage %}
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "NewsArticle",
  "headline": {{ article.title|truncate(110)|tojson }},
  "description": {{ ((article.groq_summary if is_community_article and article.groq_summary else article.description) or '')|striptags|truncate(250)|tojson }},
  {% if art_image %}"image": [{{ art_image|tojson }}],{% endif %}
  "datePublished": {{ (article.published_at.isoformat() if is_community_article and article.published_at else (article.publishedAt if not is_community_article and article.publishedAt else ''))|tojson }},
  "author": { "@type": "Person", "name": {{ (article.author.name if is_community_article and article.author else (article.source.name if not is_community_article and article.source else 'BrieflyAI'))|tojson }} },
  "publisher": { "@type": "Organization", "name": "BrieflyAI" },
  "mainEntityOfPage": { "@type": "WebPage", "@id": {{ request.base_url|tojson }} }
}
</script>
{% endif %}
{% endblock %}
{% block content %}
{% if not article %}
    <div class="state-card state-card-danger">
        <div class="state-card-icon"><i class="ph ph-newspaper" aria-hidden="true"></i></div>
        <h1 class="state-card-title">Article not found</h1>
        <p class="state-card-text">The article you're looking for may have been removed, or the link may be incorrect.</p>
        <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Go to homepage</a></div>
    </div>
{% else %}
<div class="reading-progress" aria-hidden="true"><div class="reading-progress__fill" id="readingProgressFill"></div></div>
<article class="article-full-content-wrapper animate-fade-in" id="articleWrapper" data-text-size="normal">
    <div class="mb-3 d-flex justify-content-between align-items-center flex-wrap gap-2">
        <a href="{{ previous_list_page }}" class="btn btn-sm btn-outline-secondary"><i class="ph ph-arrow-left me-2" aria-hidden="true"></i>Back to list</a>

        <div class="d-flex align-items-center gap-2">

            {% if session.get('is_admin') and is_community_article %}
            <button type="button" id="adminDeleteBtn" class="btn btn-sm btn-danger" title="Admin: Delete Post">
                <i class="ph ph-trash" aria-hidden="true"></i> Delete post
            </button>
            {% endif %}

            {% if session.user_id %}
                {% if is_community_article %}
                <button type="button" id="reportBtn" class="btn btn-sm btn-outline-warning" title="Report this article for review">
                    <i class="ph ph-flag" aria-hidden="true"></i> Report
                </button>
                {% endif %}

                <button type="button" id="bookmarkBtn" class="bookmark-btn {% if is_bookmarked %}active{% endif %}" title="{% if is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" aria-label="{% if is_bookmarked %}Remove bookmark{% else %}Add bookmark{% endif %} for this article" data-article-hash-id="{{ article.article_hash_id if is_community_article else article.id }}" data-is-community="{{ 'true' if is_community_article else 'false' }}" data-title="{{ article.title|e }}" data-source-name="{{ (article.author.name if is_community_article and article.author else article.source.name)|e }}" data-image-url="{{ ((article.image_url if is_community_article else article.urlToImage) or '')|e }}" data-description="{{ (article.description if article.description else '')|e }}" data-published-at="{{ (article.published_at.isoformat() if is_community_article and article.published_at else (article.publishedAt if not is_community_article and article.publishedAt else ''))|e }}"><i class="ph-fill ph-bookmark-simple" aria-hidden="true"></i></button>
            {% endif %}
        </div>
    </div>

    <h1 class="mb-2 article-title-main">{{ article.title }}</h1>
    <div class="article-meta-detailed d-flex align-items-center flex-wrap gap-2 text-muted small">
        <span class="meta-item" title="Source"><i class="ph ph-{{ 'user-circle' if is_community_article else 'buildings' }}" aria-hidden="true"></i> {{ article.author.name if is_community_article and article.author else article.source.name }}</span>
        <span class="meta-item" title="Published Date"><i class="ph ph-calendar-blank" aria-hidden="true"></i> {{ (article.published_at | to_ist if is_community_article else (article.publishedAt | to_ist if article.publishedAt else 'N/A')) }}</span>
    </div>
    {% set image_to_display = article.image_url if is_community_article else article.urlToImage %}
    {% if image_to_display %}
    <div class="hero-image-wrap">
        <img src="{{ image_to_display }}" alt="{{ article.title|truncate(50) }}" class="hero-image" loading="eager" decoding="async">
        <div class="img-fallback-icon" aria-hidden="true"><i class="ph ph-newspaper"></i></div>
    </div>
    {% endif %}

    <div class="article-toolbar" role="toolbar" aria-label="Article tools">
        <span class="read-time-badge" id="readTimeBadge" {% if not is_community_article %}hidden{% endif %}>
            <i class="ph ph-clock" aria-hidden="true"></i>
            <span id="readTimeValue">{% if is_community_article and article.full_text %}{{ [1, ((article.full_text|wordcount) / 220)|round(0, 'ceil')|int]|max }} min read{% endif %}</span>
        </span>

        <button type="button" class="toolbar-btn" id="listenBtn" hidden aria-pressed="false">
            <i class="ph ph-headphones" aria-hidden="true"></i> <span class="listen-label">Listen</span>
        </button>

        <button type="button" class="toolbar-btn" id="textSizeBtn" aria-label="Change text size">
            <i class="ph ph-text-aa" aria-hidden="true"></i> <span id="textSizeLabel">Normal</span>
        </button>

        <div class="share-wrap">
            <button type="button" class="toolbar-btn" id="shareBtn" aria-haspopup="true" aria-expanded="false" aria-controls="shareMenu">
                <i class="ph ph-share-network" aria-hidden="true"></i> Share
            </button>
            <div class="share-menu" id="shareMenu" role="menu" aria-label="Share this article">
                <a href="#" data-share="whatsapp" role="menuitem" target="_blank" rel="noopener noreferrer"><i class="ph ph-whatsapp-logo i-whatsapp" aria-hidden="true"></i> WhatsApp</a>
                <a href="#" data-share="x" role="menuitem" target="_blank" rel="noopener noreferrer"><i class="ph ph-x-logo i-x" aria-hidden="true"></i> X (Twitter)</a>
                <a href="#" data-share="facebook" role="menuitem" target="_blank" rel="noopener noreferrer"><i class="ph ph-facebook-logo i-facebook" aria-hidden="true"></i> Facebook</a>
                <a href="#" data-share="linkedin" role="menuitem" target="_blank" rel="noopener noreferrer"><i class="ph ph-linkedin-logo i-linkedin" aria-hidden="true"></i> LinkedIn</a>
                <button type="button" data-share="copy" role="menuitem"><i class="ph ph-link i-link" aria-hidden="true"></i> Copy link</button>
            </div>
        </div>

        <button type="button" class="toolbar-btn" id="printBtn" aria-label="Print this article">
            <i class="ph ph-printer" aria-hidden="true"></i>
        </button>
    </div>

    <div id="contentLoader" class="ai-skeleton my-4 {% if is_community_article %}d-none{% endif %}" aria-hidden="true">
        <div class="ai-skeleton-box">
            <div class="ai-skeleton-label"></div>
            <div class="ai-skeleton-line w-100"></div>
            <div class="ai-skeleton-line w-95"></div>
            <div class="ai-skeleton-line w-80"></div>
        </div>
        <div class="ai-skeleton-box mb-2">
            <div class="ai-skeleton-label"></div>
            <div class="ai-skeleton-line w-90"></div>
            <div class="ai-skeleton-line w-70"></div>
        </div>
        <p class="ai-skeleton-caption"><i class="ph ph-brain" aria-hidden="true"></i> Generating AI summary&hellip;</p>
    </div>
    <div id="articleAnalysisContainer">
    {% if is_community_article %}
        {% if article.groq_summary %}<div class="summary-box my-3"><h2><i class="ph ph-book-open me-2" aria-hidden="true"></i>AI summary</h2><p class="mb-0">{{ article.groq_summary|e|replace('\\n', '<br>')|safe }}</p></div>{% endif %}
        {% if article.parsed_takeaways %}<div class="takeaways-box my-3"><h2><i class="ph ph-list-checks me-2" aria-hidden="true"></i>Key takeaways</h2><ul>{% for takeaway in article.parsed_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul></div>{% endif %}
        <hr class="my-4"><h2 class="content-divider-heading">Full article content</h2><div class="content-text">{{ article.full_text }}</div>
    {% else %}<div id="apiArticleContent"></div>{% endif %}
    </div>

    {% with ad_name='article', ad_slot=ad_slots.get('article') %}{% include "_AD_SLOT_TEMPLATE" %}{% endwith %}

    <section class="comment-section mt-5" id="comment-section">
        <div class="comment-toolbar">
            <h2 class="mb-0">Community discussion (<span id="comment-count">{{ total_comment_count }}</span>)</h2>
            {% if comments %}
            <div class="d-flex align-items-center gap-2">
                <label for="commentSort" class="small text-muted mb-0">Sort</label>
                <select class="form-select sort-select" id="commentSort" aria-label="Sort comments">
                    <option value="newest">Newest first</option>
                    <option value="oldest">Oldest first</option>
                    <option value="reactions">Most reactions</option>
                </select>
            </div>
            {% endif %}
        </div>

        <div id="comments-list">
            {% for comment in comments %}
                {% include '_COMMENT_TEMPLATE' %}
            {% else %}
                <p id="no-comments-msg" class="text-muted mt-3"><i class="ph ph-chat-dots me-2" aria-hidden="true"></i>No comments yet. Start the discussion.</p>
            {% endfor %}
        </div>

        {% if session.user_id %}
            <div class="add-comment-form mt-4 pt-4 border-top">
                <h3 class="comment-form-heading">Leave a comment</h3>
                <form id="comment-form">
                    <label class="visually-hidden" for="comment-content">Write a comment</label>
                    <div class="mb-3"><textarea class="form-control" id="comment-content" name="content" rows="4" placeholder="Share your insights..." required></textarea></div>
                    <button type="submit" class="btn btn-primary">Post comment</button>
                </form>
            </div>
        {% else %}
            <div class="comment-login-prompt">
                <i class="ph ph-chats" aria-hidden="true"></i>
                <p class="mb-0">Please <a href="{{ url_for('login', next=request.url) }}" class="fw-bold">log in</a> to join the discussion.</p>
            </div>
        {% endif %}
    </section>
</article>
{% endif %}
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    try {
        {% if article %}
        const articleHashIdGlobal = {{ (article.article_hash_id if is_community_article else article.id) | tojson }};
        const isUserLoggedIn = {{ 'true' if session.user_id else 'false' }};
        const isCommunityArticle = {{ is_community_article | tojson }};
        const articleTitleGlobal = {{ article.title | tojson }};
        const articleSourceGlobal = {{ (article.author.name if is_community_article and article.author else (article.source.name if not is_community_article and article.source else 'BrieflyAI')) | tojson }};

        /* --- Remember this article locally for the "Recently viewed" strip --- */
        BrieflyAI.recent.add({
            url: window.location.pathname,
            title: articleTitleGlobal,
            source: articleSourceGlobal
        });

        /* --- Reading progress bar --- */
        (function () {
            var fill = document.getElementById('readingProgressFill');
            var wrapper = document.getElementById('articleWrapper');
            if (!fill || !wrapper) { return; }
            var ticking = false;
            function update() {
                var rect = wrapper.getBoundingClientRect();
                var total = rect.height - window.innerHeight;
                var pct = total <= 0 ? 100 : ((-rect.top) / total) * 100;
                fill.style.width = Math.min(100, Math.max(0, pct)).toFixed(1) + '%';
                ticking = false;
            }
            window.addEventListener('scroll', function () {
                if (!ticking) { ticking = true; requestAnimationFrame(update); }
            }, { passive: true });
            window.addEventListener('resize', update, { passive: true });
            update();
        })();

        /* --- Read time: server-rendered for community posts, computed after fetch for API ones --- */
        BrieflyAI.setReadTime = function (text) {
            if (!text) { return; }
            var words = String(text).trim().split(/\\s+/).length;
            var mins = Math.max(1, Math.ceil(words / 220));
            var badge = document.getElementById('readTimeBadge');
            var value = document.getElementById('readTimeValue');
            if (badge && value) { value.textContent = mins + ' min read'; badge.hidden = false; }
        };

        /* --- Listen (Web Speech API) --- */
        (function () {
            var btn = document.getElementById('listenBtn');
            if (!btn || !('speechSynthesis' in window)) { return; }
            btn.hidden = false;
            var speaking = false;
            function textToRead() {
                var parts = [];
                var h1 = document.querySelector('.article-title-main');
                if (h1) { parts.push(h1.textContent); }
                var summary = document.querySelector('.summary-box p');
                if (summary) { parts.push('Summary. ' + summary.textContent); }
                document.querySelectorAll('.takeaways-box li').forEach(function (li, i) {
                    if (i === 0) { parts.push('Key takeaways.'); }
                    parts.push(li.textContent);
                });
                var body = document.querySelector('.content-text');
                if (body) { parts.push(body.textContent); }
                return parts.join('. ');
            }
            function setState(on) {
                speaking = on;
                btn.classList.toggle('is-active', on);
                btn.setAttribute('aria-pressed', on ? 'true' : 'false');
                btn.querySelector('.listen-label').textContent = on ? 'Stop' : 'Listen';
                btn.querySelector('i').className = on ? 'ph ph-stop' : 'ph ph-headphones';
            }
            btn.addEventListener('click', function () {
                if (speaking) { window.speechSynthesis.cancel(); setState(false); return; }
                var text = textToRead();
                if (!text.trim()) { BrieflyAI.showToast('Nothing to read yet -- still loading.', 'info', 3000); return; }
                var utter = new SpeechSynthesisUtterance(text);
                utter.rate = 1.0;
                utter.lang = document.documentElement.lang || 'en';
                utter.onend = function () { setState(false); };
                utter.onerror = function () { setState(false); };
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utter);
                setState(true);
            });
            // Browsers keep speaking after navigation otherwise.
            window.addEventListener('beforeunload', function () { window.speechSynthesis.cancel(); });
        })();

        /* --- Text size preference --- */
        (function () {
            var btn = document.getElementById('textSizeBtn');
            var wrapper = document.getElementById('articleWrapper');
            var label = document.getElementById('textSizeLabel');
            if (!btn || !wrapper) { return; }
            var sizes = ['normal', 'large', 'xlarge'];
            var names = { normal: 'Normal', large: 'Large', xlarge: 'X-Large' };
            var current = 'normal';
            try { current = localStorage.getItem('brieflyai:textSize') || 'normal'; } catch (e) {}
            if (sizes.indexOf(current) === -1) { current = 'normal'; }
            function apply(size) {
                current = size;
                wrapper.dataset.textSize = size;
                if (label) { label.textContent = names[size]; }
                try { localStorage.setItem('brieflyai:textSize', size); } catch (e) {}
            }
            apply(current);
            btn.addEventListener('click', function () {
                apply(sizes[(sizes.indexOf(current) + 1) % sizes.length]);
            });
        })();

        /* --- Share --- */
        (function () {
            var btn = document.getElementById('shareBtn');
            var menu = document.getElementById('shareMenu');
            if (!btn || !menu) { return; }
            var url = window.location.href;
            var targets = BrieflyAI.share.targets(url, articleTitleGlobal);
            Object.keys(targets).forEach(function (k) {
                var link = menu.querySelector('[data-share="' + k + '"]');
                if (link) { link.href = targets[k]; }
            });

            function closeMenu() { menu.classList.remove('show'); btn.setAttribute('aria-expanded', 'false'); }

            btn.addEventListener('click', function () {
                // Prefer the OS share sheet where it exists (mobile), menu everywhere else.
                if (navigator.share) {
                    BrieflyAI.share.native(url, articleTitleGlobal)
                        .catch(function (err) {
                            if (err && err.name === 'AbortError') { return; }
                            menu.classList.add('show');
                            btn.setAttribute('aria-expanded', 'true');
                        });
                    return;
                }
                var show = !menu.classList.contains('show');
                menu.classList.toggle('show', show);
                btn.setAttribute('aria-expanded', show ? 'true' : 'false');
            });

            menu.querySelector('[data-share="copy"]').addEventListener('click', function () {
                BrieflyAI.share.copy(url)
                    .then(function () { BrieflyAI.showToast('Link copied to clipboard.', 'success', 2500); })
                    .catch(function () { BrieflyAI.showToast('Could not copy the link.', 'danger'); });
                closeMenu();
            });
            menu.querySelectorAll('a[data-share]').forEach(function (a) {
                a.addEventListener('click', closeMenu);
            });
            document.addEventListener('click', function (e) {
                if (!e.target.closest('.share-wrap')) { closeMenu(); }
            });
            document.addEventListener('keydown', function (e) {
                if (e.key === 'Escape' && menu.classList.contains('show')) { closeMenu(); btn.focus(); }
            });
        })();

        var printBtn = document.getElementById('printBtn');
        if (printBtn) { printBtn.addEventListener('click', function () { window.print(); }); }

        /* --- Comment sorting (client-side reorder of top-level threads) --- */
        (function () {
            var select = document.getElementById('commentSort');
            var list = document.getElementById('comments-list');
            if (!select || !list) { return; }
            function reactionScore(thread) {
                var total = 0;
                thread.querySelectorAll(':scope > .comment-container .reaction-pill .count').forEach(function (c) {
                    total += parseInt(c.textContent, 10) || 0;
                });
                return total;
            }
            select.addEventListener('change', function () {
                var mode = select.value;
                var threads = Array.prototype.slice.call(list.children).filter(function (el) {
                    return el.classList.contains('comment-thread');
                });
                // DOM order is oldest-first as rendered by the server.
                threads.sort(function (a, b) {
                    if (mode === 'reactions') {
                        var diff = reactionScore(b) - reactionScore(a);
                        if (diff !== 0) { return diff; }
                    }
                    var ai = parseInt(a.id.replace('comment-', ''), 10) || 0;
                    var bi = parseInt(b.id.replace('comment-', ''), 10) || 0;
                    return mode === 'oldest' ? ai - bi : bi - ai;
                });
                threads.forEach(function (t) { list.appendChild(t); });
            });
        })();

        const adminDeleteBtn = document.getElementById('adminDeleteBtn');
        if (adminDeleteBtn) {
            adminDeleteBtn.addEventListener('click', async function() {
                const confirmed = await BrieflyAI.confirmAction({
                    title: 'Delete this post?',
                    message: 'This will permanently delete the community post and cannot be undone.',
                    confirmText: 'Delete post',
                    danger: true
                });
                if (!confirmed) return;

                this.disabled = true;
                this.innerHTML = '<i class="ph ph-circle-notch ph-spin" aria-hidden="true"></i> Deleting...';

                BrieflyAI.postJSON(`/delete_community_article/${articleHashIdGlobal}`)
                .then(res => res.json().then(data => ({ ok: res.ok, data })))
                .then(({ ok, data }) => {
                    if (ok && data.success) {
                        window.location.href = data.redirect_url;
                    } else {
                        BrieflyAI.showToast('Deletion failed: ' + (data.error || 'Unknown error'), 'danger');
                        this.disabled = false;
                        this.innerHTML = '<i class="ph ph-trash" aria-hidden="true"></i> Delete post';
                    }
                })
                .catch(err => {
                    console.error("Admin delete error:", err);
                    BrieflyAI.showToast("A network error occurred. Could not delete the post.", 'danger');
                    this.disabled = false;
                    this.innerHTML = '<i class="ph ph-trash" aria-hidden="true"></i> Delete post';
                });
            });
        }

        if (!isCommunityArticle) {
            const contentLoader = document.getElementById('contentLoader');
            const apiArticleContent = document.getElementById('apiArticleContent');

            fetch(`{{ url_for('get_article_content_json', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashIdGlobal))
                .then(response => { if (!response.ok) { throw new Error(`Network error, status: ${response.status}`); } return response.json(); })
                .then(data => {
                    if (data.error) { throw new Error(data.error); }
                    let html = '';
                    const articleUrl = {{ article.url | tojson if article and not is_community_article else 'null' }};
                    const articleSourceName = {{ article.source.name | tojson if article and not is_community_article and article.source else 'Source'|tojson }};
                    const analysis = data.groq_analysis;
                    if (analysis) {
                        if (analysis.error) { html += `<div class="alert alert-secondary small p-3 mt-3">AI analysis could not be performed: ${analysis.error}</div>`; }
                        else {
                            if (analysis.groq_summary) { html += `<div class="summary-box my-3"><h2><i class="ph ph-book-open me-2" aria-hidden="true"></i>AI summary</h2><p class="mb-0">${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`; }
                            if (analysis.groq_takeaways && analysis.groq_takeaways.length > 0) { html += `<div class="takeaways-box my-3"><h2><i class="ph ph-list-checks me-2" aria-hidden="true"></i>Key takeaways</h2><ul>${analysis.groq_takeaways.map(t => `<li>${String(t)}</li>`).join('')}</ul></div>`; }
                        }
                    }
                    if (articleUrl) { html += `<hr class="my-4"><a href="${articleUrl}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">Read the original at ${articleSourceName} <i class="ph ph-arrow-square-out ms-1" aria-hidden="true"></i></a>`; }
                    apiArticleContent.innerHTML = html;
                    // Read time comes from whatever text we actually received.
                    BrieflyAI.setReadTime(data.full_text || (analysis && analysis.groq_summary) || '');
                })
                .catch(error => { console.error("Failed to load article content:", error); if (apiArticleContent) { apiArticleContent.innerHTML = `<div class="alert alert-danger small p-3">Failed to load article analysis. Details: ${error.message}</div>`; } })
                .finally(() => { if (contentLoader) contentLoader.style.display = 'none'; });
        }

        const commentSection = document.getElementById('comment-section');
        if (commentSection && isUserLoggedIn) {

            const handleCommentSubmit = (formElement) => {
                const content = formElement.querySelector('textarea[name="content"]').value;
                const parentId = formElement.querySelector('input[name="parent_id"]')?.value || null;
                if (!content.trim()) return;
                const submitButton = formElement.querySelector('button[type="submit"]');
                const originalButtonText = submitButton.innerHTML;
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Posting...';
                BrieflyAI.postJSON(
                    `{{ url_for('add_comment', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashIdGlobal),
                    { content, parent_id: parentId }
                )
                .then(res => {
                    if (res.status === 401) { throw new Error("Your session has expired. Please refresh the page and log in again."); }
                    if (!res.ok) { return res.json().then(err => { throw new Error(err.error || "An unknown server error occurred."); }); }
                    return res.json();
                })
                .then(data => {
                    if (data.success) {
                        const noCommentsMsg = document.getElementById('no-comments-msg');
                        if (noCommentsMsg) noCommentsMsg.remove();
                        if (data.parent_id) {
                            document.getElementById(`replies-of-${data.parent_id}`).insertAdjacentHTML('beforeend', data.html);
                            formElement.closest('.reply-form-container').style.display = 'none';
                        } else {
                            document.getElementById('comments-list').insertAdjacentHTML('beforeend', data.html);
                        }
                        const countEl = document.getElementById('comment-count');
                        countEl.textContent = parseInt(countEl.textContent) + 1;
                        formElement.reset();
                    } else { throw new Error(data.error || 'Could not post comment.'); }
                })
                .catch(err => { console.error("Comment submission error:", err); BrieflyAI.showToast("Error: " + err.message, 'danger'); })
                .finally(() => { submitButton.disabled = false; submitButton.innerHTML = originalButtonText; });
            };

            const updateReactionUI = (commentId, reactions, userReaction) => {
                const summaryContainer = document.getElementById(`reaction-summary-${commentId}`);
                if (summaryContainer) {
                    let summaryHTML = '';
                    if (reactions) {
                        for (const [emoji, count] of Object.entries(reactions)) {
                            if (count > 0) {
                                const userReactedClass = (userReaction === emoji) ? 'user-reacted' : '';
                                summaryHTML += `<div class="reaction-pill ${userReactedClass}" data-emoji="${emoji}"><span class="emoji">${emoji}</span><span class="count">${count}</span></div>`;
                            }
                        }
                    }
                    summaryContainer.innerHTML = summaryHTML;
                }
                const box = document.getElementById(`reaction-box-${commentId}`);
                if (box) {
                    box.querySelectorAll('.reaction-emoji').forEach(function (btn) {
                        btn.classList.toggle('is-selected', btn.dataset.emoji === userReaction);
                    });
                }
            };

            const closeAllReactionBoxes = () => {
                document.querySelectorAll('.reaction-box.show').forEach(box => box.classList.remove('show'));
                document.querySelectorAll('.react-btn[aria-expanded="true"]').forEach(btn => btn.setAttribute('aria-expanded', 'false'));
            };

            commentSection.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    const openBtn = document.querySelector('.react-btn[aria-expanded="true"]');
                    closeAllReactionBoxes();
                    if (openBtn) openBtn.focus();
                }
            });

            commentSection.addEventListener('click', async function(e) {
                const target = e.target;

                const deleteBtn = target.closest('.delete-btn');
                if (deleteBtn) {
                    e.preventDefault();
                    const commentId = deleteBtn.dataset.commentId;
                    const confirmed = await BrieflyAI.confirmAction({
                        title: 'Delete this comment?',
                        message: 'All replies will also be removed. This cannot be undone.',
                        confirmText: 'Delete',
                        danger: true
                    });
                    if (!confirmed) return;
                    BrieflyAI.postJSON(`/delete_comment/${commentId}`)
                        .then(res => {
                            if (!res.ok) { return res.json().then(err => { throw new Error(err.error) }); }
                            return res.json();
                        })
                        .then(data => {
                            if (data.success) {
                                const commentElement = document.getElementById(`comment-${commentId}`);
                                const repliesCount = commentElement.querySelectorAll('.comment-thread').length;
                                const totalCommentsToRemove = 1 + repliesCount;

                                const countEl = document.getElementById('comment-count');
                                countEl.textContent = Math.max(0, parseInt(countEl.textContent) - totalCommentsToRemove);

                                commentElement.style.transition = 'opacity 0.4s ease';
                                commentElement.style.opacity = '0';
                                setTimeout(() => commentElement.remove(), 400);
                            } else {
                                BrieflyAI.showToast('Error: ' + data.error, 'danger');
                            }
                        })
                        .catch(err => {
                            console.error("Delete error:", err);
                            BrieflyAI.showToast("Could not delete comment: " + err.message, 'danger');
                        });
                    return;
                }

                const editBtn = target.closest('.edit-btn');
                if (editBtn) {
                    e.preventDefault();
                    const commentId = editBtn.dataset.commentId;
                    const commentThread = document.getElementById(`comment-${commentId}`);
                    const commentBody = commentThread.querySelector('.comment-body');
                    commentBody.querySelector('.comment-content').style.display = 'none';
                    commentBody.querySelector('.comment-actions').style.display = 'none';
                    const editContainer = commentBody.querySelector('.edit-form-container');
                    editContainer.style.display = 'block';
                    editContainer.querySelector('textarea').focus();
                    return;
                }

                const cancelEditBtn = target.closest('.cancel-edit-btn');
                if (cancelEditBtn) {
                    e.preventDefault();
                    const commentBody = cancelEditBtn.closest('.comment-body');
                    commentBody.querySelector('.comment-content').style.display = 'block';
                    commentBody.querySelector('.comment-actions').style.display = 'flex';
                    commentBody.querySelector('.edit-form-container').style.display = 'none';
                    return;
                }

                const replyBtn = target.closest('.reply-btn');
                if (replyBtn) {
                    e.preventDefault();
                    const commentId = replyBtn.dataset.commentId;
                    const formContainer = document.getElementById(`reply-form-container-${commentId}`);
                    if (formContainer) {
                        const isDisplayed = formContainer.style.display === 'block';
                        document.querySelectorAll('.reply-form-container').forEach(fc => fc.style.display = 'none');
                        formContainer.style.display = isDisplayed ? 'none' : 'block';
                        if (!isDisplayed) formContainer.querySelector('textarea').focus();
                    }
                    return;
                }

                const reactBtn = target.closest('.react-btn');
                if (reactBtn) {
                    e.preventDefault();
                    const commentId = reactBtn.dataset.commentId;
                    const reactionBox = document.getElementById(`reaction-box-${commentId}`);
                    if (reactionBox) {
                        const isShown = reactionBox.classList.contains('show');
                        closeAllReactionBoxes();
                        if (!isShown) {
                            reactionBox.classList.add('show');
                            reactBtn.setAttribute('aria-expanded', 'true');
                        }
                    }
                    return;
                }

                const reactionEmoji = target.closest('.reaction-emoji');
                if (reactionEmoji) {
                    e.preventDefault();
                    const commentId = reactionEmoji.dataset.commentId;
                    const emoji = reactionEmoji.dataset.emoji;
                    closeAllReactionBoxes();
                    BrieflyAI.postJSON(`/vote_comment/${commentId}`, { emoji: emoji })
                    .then(res => res.json())
                    .then(data => {
                        if (data.success) { updateReactionUI(commentId, data.reactions, data.user_reaction); }
                        else { throw new Error(data.error || "Failed to vote."); }
                    })
                    .catch(err => { console.error("Reaction error:", err); BrieflyAI.showToast("Error: " + err.message, 'danger'); });
                    return;
                }

                if (!target.closest('.reaction-box') && !target.closest('.react-btn')) {
                    closeAllReactionBoxes();
                }
            });

            commentSection.addEventListener('submit', function(e) {
                e.preventDefault();

                if (e.target.matches('.edit-comment-form')) {
                    const form = e.target;
                    const commentId = form.closest('.comment-thread').id.replace('comment-', '');
                    const newContent = form.querySelector('textarea[name="content"]').value.trim();
                    if (!newContent) return;

                    BrieflyAI.postJSON(`/edit_comment/${commentId}`, { content: newContent })
                    .then(res => {
                        if (!res.ok) { return res.json().then(err => { throw new Error(err.error) }); }
                        return res.json();
                    })
                    .then(data => {
                        if (data.success) {
                            const commentBody = form.closest('.comment-body');
                            const contentP = commentBody.querySelector('.comment-content');
                            contentP.textContent = data.new_content;
                            contentP.style.display = 'block';
                            commentBody.querySelector('.comment-actions').style.display = 'flex';
                            form.closest('.edit-form-container').style.display = 'none';
                        } else {
                            BrieflyAI.showToast('Error: ' + data.error, 'danger');
                        }
                    })
                    .catch(err => {
                        console.error("Edit error:", err);
                        BrieflyAI.showToast("Could not save changes: " + err.message, 'danger');
                    });
                    return;
                }

                if (e.target.id === 'comment-form' || e.target.matches('.reply-form')) {
                    handleCommentSubmit(e.target);
                }
            });
        }

        const reportBtn = document.getElementById('reportBtn');
        if (reportBtn) {
            reportBtn.addEventListener('click', async function() {
                const confirmed = await BrieflyAI.confirmAction({
                    title: 'Report this article?',
                    message: 'This will flag the article for review by our moderators.',
                    confirmText: 'Report',
                    danger: true
                });
                if (!confirmed) return;

                this.disabled = true;
                this.innerHTML = '<i class="ph ph-circle-notch ph-spin" aria-hidden="true"></i> Reporting...';
                BrieflyAI.postJSON(`/report_article/${articleHashIdGlobal}`)
                .then(res => res.json().then(data => ({ ok: res.ok, status: res.status, data })))
                .then(({ ok, status, data }) => {
                    if (ok) {
                        this.innerHTML = '<i class="ph ph-check" aria-hidden="true"></i> Reported';
                        BrieflyAI.showToast(data.message, 'success');
                    } else {
                        this.disabled = false;
                        this.innerHTML = '<i class="ph ph-flag" aria-hidden="true"></i> Report';
                        BrieflyAI.showToast('Error: ' + data.error, 'danger');
                    }
                })
                .catch(err => {
                    console.error("Report error:", err);
                    BrieflyAI.showToast("A network error occurred. Please try again.", 'danger');
                    this.disabled = false;
                    this.innerHTML = '<i class="ph ph-flag" aria-hidden="true"></i> Report';
                });
            });
        }

        const bookmarkBtn = document.getElementById('bookmarkBtn');
        if (bookmarkBtn && isUserLoggedIn) {
            bookmarkBtn.addEventListener('click', function() {
                const articleHashId = this.dataset.articleHashId;
                const isCommunity = this.dataset.isCommunity;
                const title = this.dataset.title;
                const sourceName = this.dataset.sourceName;
                const imageUrl = this.dataset.imageUrl;
                const description = this.dataset.description;
                const publishedAt = this.dataset.publishedAt;
                const btnRef = this;
                BrieflyAI.postJSON(
                    `{{ url_for('toggle_bookmark', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashId),
                    { is_community_article: isCommunity, title, source_name: sourceName, image_url: imageUrl, description, published_at: publishedAt }
                )
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        const nowActive = data.status === 'added';
                        btnRef.classList.toggle('active', nowActive);
                        btnRef.title = nowActive ? 'Remove Bookmark' : 'Add Bookmark';
                        btnRef.setAttribute('aria-label', (nowActive ? 'Remove bookmark' : 'Add bookmark') + ' for this article');
                        btnRef.classList.remove('is-popping');
                        void btnRef.offsetWidth;
                        btnRef.classList.add('is-popping');
                        BrieflyAI.showToast(data.message, 'success', 3000);
                    } else { BrieflyAI.showToast(data.error || 'Could not update bookmark.', 'danger'); }
                })
                .catch(err => { console.error("Bookmark error:", err); BrieflyAI.showToast("Could not update bookmark: " + err.message, 'danger'); });
            });
        }
        {% endif %}
    } catch (e) {
        console.error("A critical error occurred on the article page:", e);
    }
});
</script>
{% endblock %}
"""
LOGIN_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Login - BrieflyAI{% endblock %}
{% block body_class %}body-auth{% endblock %}

{% block content %}
<div class="auth-card animate-fade-in">
    <div class="auth-header">
        <div class="icon icon-logo"><svg class="logo-mark" viewBox="0 0 32 32" width="48" height="48" aria-hidden="true" focusable="false"><rect class="logo-tile" x="1" y="1" width="30" height="30" rx="9"/><rect class="logo-line" x="8" y="8.5" width="16" height="3" rx="1.5"/><rect class="logo-line" x="8" y="14.5" width="11" height="3" rx="1.5"/><rect class="logo-gist" x="8" y="20.5" width="6" height="3" rx="1.5"/></svg></div>
        <h1>Welcome back</h1>
    </div>
    <div class="auth-body">
        <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}" id="loginForm">
            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
            <div class="mb-3">
                <label for="username" class="form-label fw-medium">Username</label>
                <div class="input-group-icon">
                    <i class="ph ph-user input-icon" aria-hidden="true"></i>
                    <input type="text" class="form-control" id="username" name="username" required placeholder="e.g. user123" autocomplete="username">
                </div>
            </div>
            <div class="mb-4">
                <label for="password" class="form-label fw-medium">Password</label>
                <div class="input-group-icon">
                    <i class="ph ph-lock input-icon" aria-hidden="true"></i>
                    <input type="password" class="form-control" id="password" name="password" required placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;" autocomplete="current-password">
                </div>
            </div>
            <button type="submit" class="btn btn-primary w-100" id="loginSubmitBtn">Sign in</button>
        </form>
    </div>
    <div class="auth-footer">
        <p class="mb-0 small">
            Don't have an account? <a href="{{ url_for('register', next=request.args.get('next')) }}" class="fw-bold text-decoration-none">Sign up now</a>
        </p>
    </div>
</div>
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('loginForm');
    const btn = document.getElementById('loginSubmitBtn');
    if (form && btn) {
        form.addEventListener('submit', function () {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Signing in...';
        });
    }
});
</script>
{% endblock %}
"""

REGISTER_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Register - BrieflyAI{% endblock %}
{% block body_class %}body-auth{% endblock %}

{% block content %}
<div class="auth-card animate-fade-in">
     <div class="auth-header">
        <div class="icon icon-logo"><svg class="logo-mark" viewBox="0 0 32 32" width="48" height="48" aria-hidden="true" focusable="false"><rect class="logo-tile" x="1" y="1" width="30" height="30" rx="9"/><rect class="logo-line" x="8" y="8.5" width="16" height="3" rx="1.5"/><rect class="logo-line" x="8" y="14.5" width="11" height="3" rx="1.5"/><rect class="logo-gist" x="8" y="20.5" width="6" height="3" rx="1.5"/></svg></div>
        <h1>Create your account</h1>
    </div>
    <div class="auth-body">
        <form method="POST" action="{{ url_for('register') }}" id="registerForm">
            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
             <div class="mb-3">
                <label for="name" class="form-label fw-medium">Full name</label>
                <div class="input-group-icon">
                    <i class="ph ph-identification-card input-icon" aria-hidden="true"></i>
                    <input type="text" class="form-control" id="name" name="name" required placeholder="e.g. John Doe" autocomplete="name">
                </div>
            </div>
            <div class="mb-3">
                <label for="username" class="form-label fw-medium">Username</label>
                <div class="input-group-icon">
                    <i class="ph ph-user input-icon" aria-hidden="true"></i>
                    <input type="text" class="form-control" id="username" name="username" required minlength="3" placeholder="e.g. johndoe (min 3 chars)" autocomplete="username">
                </div>
            </div>
            <div class="mb-4">
                <label for="password" class="form-label fw-medium">Password</label>
                <div class="input-group-icon">
                    <i class="ph ph-lock input-icon" aria-hidden="true"></i>
                    <input type="password" class="form-control" id="password" name="password" required minlength="6" placeholder="min 6 chars" autocomplete="new-password">
                </div>
            </div>
            <button type="submit" class="btn btn-primary w-100" id="registerSubmitBtn">Create account</button>
        </form>
    </div>
    <div class="auth-footer">
        <p class="mb-0 small">
            Already have an account? <a href="{{ url_for('login') }}" class="fw-bold text-decoration-none">Sign in</a>
        </p>
    </div>
</div>
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('registerForm');
    const btn = document.getElementById('registerSubmitBtn');
    if (form && btn) {
        form.addEventListener('submit', function () {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Creating account...';
        });
    }
});
</script>
{% endblock %}
"""
PROFILE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ user.name }}'s Profile - BrieflyAI{% endblock %}
{% block content %}
<div class="profile-header-card animate-fade-in">
    <div class="profile-avatar-wrapper">
        <div class="profile-avatar" aria-hidden="true">{{ user.name[0]|upper }}</div>
    </div>
    <h1>{{ user.name }}</h1>
    <p class="username">@{{ user.username }}</p>
    <p class="small text-muted mb-0">Joined: {{ user.created_at | to_ist }}</p>
    <div class="profile-stats">
        <button type="button" class="stat-item" data-target-tab="posted-tab" aria-label="View articles you've posted">
            <div class="icon"><i class="ph ph-note-pencil" aria-hidden="true"></i></div>
            <div class="count">{{ posted_articles|length }}</div>
            <div class="label">Articles posted</div>
        </button>
        <button type="button" class="stat-item" data-target-tab="bookmarks-tab" aria-label="View your bookmarks">
            <div class="icon"><i class="ph-fill ph-bookmark-simple" aria-hidden="true"></i></div>
            <div class="count">{{ bookmarks_pagination.total if bookmarks_pagination else 0 }}</div>
            <div class="label">Bookmarks</div>
        </button>
    </div>
</div>

<div class="mt-4 animate-fade-in" style="animation-delay: 0.1s;">
    <ul class="nav nav-tabs profile-tabs nav-fill mb-4" id="profileTab" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="bookmarks-tab" data-bs-toggle="tab" data-bs-target="#bookmarks-content" type="button" role="tab" aria-controls="bookmarks-content" aria-selected="true"><i class="ph-fill ph-bookmark-simple me-2" aria-hidden="true"></i>My bookmarks</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="posted-tab" data-bs-toggle="tab" data-bs-target="#posted-content" type="button" role="tab" aria-controls="posted-content" aria-selected="false"><i class="ph ph-note-pencil me-2" aria-hidden="true"></i>My articles</button>
        </li>
    </ul>
    <div class="tab-content" id="profileTabContent">
        <div class="tab-pane fade show active" id="bookmarks-content" role="tabpanel" aria-labelledby="bookmarks-tab">
            {% if bookmarked_articles %}
            <div class="row g-4">
                {% for art in bookmarked_articles %}
                <div class="col-md-6 col-lg-4 d-flex">
                    <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ (loop.index0 * 0.05)|round(2) }}s">
                        <div class="article-image-container {% if not art.urlToImage %}img-fallback{% endif %}">
                            <a href="{{ art.article_url }}" tabindex="-1" aria-hidden="true">{% if art.urlToImage %}<img src="{{ art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}" loading="lazy" decoding="async">{% endif %}</a>
                            <div class="img-fallback-icon" aria-hidden="true"><i class="ph ph-newspaper"></i></div>
                            {% if art.is_stale_bookmark %}<span class="badge bg-secondary position-absolute top-0 end-0 m-2">Cached bookmark</span>{% endif %}
                        </div>
                        <div class="article-body d-flex flex-column">
                            <h3 class="article-title mb-2"><a href="{{ art.article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                            <div class="article-meta small mb-2">
                                <span class="meta-item text-muted"><i class="ph ph-{{ 'user-circle' if art.is_community_article else 'buildings' }}" aria-hidden="true"></i> {{ art.source.name|truncate(20) }}</span>
                                <span class="meta-item text-muted"><i class="ph ph-calendar-blank" aria-hidden="true"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                            </div>
                            <p class="article-description small">{{ art.description|truncate(100) }}</p>
                            <a href="{{ art.article_url }}" class="read-more btn btn-sm mt-auto">Read more <i class="ph ph-caret-right ms-1 small" aria-hidden="true"></i></a>
                        </div>
                    </article>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <div class="state-card">
                    <div class="state-card-icon"><i class="ph-fill ph-bookmark-simple" aria-hidden="true"></i></div>
                    <h2 class="state-card-title">No bookmarks yet</h2>
                    <p class="state-card-text">Find an article you like and tap the bookmark icon to save it here.</p>
                    <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Browse articles</a></div>
                </div>
            {% endif %}

            {% if bookmarks_pagination and bookmarks_pagination.pages > 1 %}
            <nav aria-label="Bookmarks navigation" class="mt-5">
                <ul class="pagination justify-content-center">
                    <li class="page-item page-link-prev-next {% if not bookmarks_pagination.has_prev %}disabled{% endif %}"><a class="page-link" href="{{ url_for('profile', page=bookmarks_pagination.prev_num) if bookmarks_pagination.has_prev else '#' }}" {% if not bookmarks_pagination.has_prev %}aria-disabled="true" tabindex="-1"{% endif %}>&laquo; Prev</a></li>
                    {% for p in bookmarks_pagination.iter_pages(left_edge=1, right_edge=1, left_current=1, right_current=2) %}{% if p %}{% if p == bookmarks_pagination.page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% else %}<li class="page-item"><a class="page-link" href="{{ url_for('profile', page=p) }}">{{ p }}</a></li>{% endif %}{% else %}<li class="page-item disabled"><span class="page-link">&hellip;</span></li>{% endif %}{% endfor %}
                    <li class="page-item page-link-prev-next {% if not bookmarks_pagination.has_next %}disabled{% endif %}"><a class="page-link" href="{{ url_for('profile', page=bookmarks_pagination.next_num) if bookmarks_pagination.has_next else '#' }}" {% if not bookmarks_pagination.has_next %}aria-disabled="true" tabindex="-1"{% endif %}>Next &raquo;</a></li>
                </ul>
            </nav>
            {% endif %}
        </div>
        <div class="tab-pane fade" id="posted-content" role="tabpanel" aria-labelledby="posted-tab">
            {% if posted_articles %}
            <div class="row g-4">
                {% for art in posted_articles %}
                <div class="col-md-6 col-lg-4 d-flex">
                    <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ (loop.index0 * 0.05)|round(2) }}s">
                        {% set article_url = url_for('article_detail', article_hash_id=art.article_hash_id) %}
                        <div class="article-image-container {% if not art.image_url %}img-fallback{% endif %}">
                            <a href="{{ article_url }}" tabindex="-1" aria-hidden="true">{% if art.image_url %}<img src="{{ art.image_url }}" class="article-image" alt="{{ art.title|truncate(50) }}" loading="lazy" decoding="async">{% endif %}</a>
                            <div class="img-fallback-icon" aria-hidden="true"><i class="ph ph-newspaper"></i></div>
                        </div>
                        <div class="article-body d-flex flex-column">
                            <h3 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                            <div class="article-meta small mb-2">
                                <span class="meta-item text-muted"><i class="ph ph-user-circle" aria-hidden="true"></i> {{ art.author.name|truncate(20) }}</span>
                                <span class="meta-item text-muted"><i class="ph ph-calendar-blank" aria-hidden="true"></i> {{ art.published_at | to_ist }}</span>
                            </div>
                            <p class="article-description small">{{ art.description|truncate(100) }}</p>
                            <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read more <i class="ph ph-caret-right ms-1 small" aria-hidden="true"></i></a>
                        </div>
                    </article>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <div class="state-card">
                    <div class="state-card-icon"><i class="ph ph-note-pencil" aria-hidden="true"></i></div>
                    <h2 class="state-card-title">Nothing posted yet</h2>
                    <p class="state-card-text">Share your first story with the BrieflyAI community &mdash; it only takes a minute.</p>
                    <div class="state-card-actions"><button type="button" class="btn btn-primary-modal" data-bs-toggle="modal" data-bs-target="#addArticleModal"><i class="ph ph-note-pencil me-2" aria-hidden="true"></i>Write a post</button></div>
                </div>
            {% endif %}
        </div>
    </div>
</div>
<section class="mt-5 pt-4 border-top" aria-labelledby="accountHeading">
    <h2 class="section-heading h5 mb-3" id="accountHeading"><i class="ph ph-user-gear me-2" aria-hidden="true"></i>Your account &amp; data</h2>
    <div class="row g-3">
        <div class="col-lg-4 col-md-6 d-flex">
            <div class="contact-card text-start w-100">
                <h3 class="h6 mb-2"><i class="ph ph-chat-circle-text me-2" aria-hidden="true"></i>Share your experience</h3>
                <p class="small text-muted mb-3">Tell other readers what BrieflyAI does for you. Approved notes appear on our welcome page.</p>
                <a href="{{ url_for('share_testimonial') }}" class="btn btn-sm btn-outline-primary">Write a note</a>
            </div>
        </div>
        <div class="col-lg-4 col-md-6 d-flex">
            <div class="contact-card text-start w-100">
                <h3 class="h6 mb-2"><i class="ph ph-download-simple me-2" aria-hidden="true"></i>Download your data</h3>
                <p class="small text-muted mb-3">Get a JSON copy of your profile, articles, comments and bookmarks.</p>
                <a href="{{ url_for('export_my_data') }}" class="btn btn-sm btn-outline-primary">Export my data</a>
            </div>
        </div>
        <div class="col-lg-4 col-md-6 d-flex">
            <div class="contact-card text-start w-100">
                <h3 class="h6 mb-2"><i class="ph ph-warning me-2" aria-hidden="true"></i>Delete your account</h3>
                <p class="small text-muted mb-3">Permanently removes your account, articles, comments and bookmarks. This cannot be undone.</p>
                <button type="button" class="btn btn-sm btn-outline-danger" data-bs-toggle="modal" data-bs-target="#deleteAccountModal">Delete account</button>
            </div>
        </div>
    </div>
</section>

<div class="modal fade" id="deleteAccountModal" tabindex="-1" aria-hidden="true" aria-labelledby="deleteAccountLabel">
    <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
            <form method="POST" action="{{ url_for('delete_account') }}">
                <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                <div class="modal-header border-0 pb-0">
                    <h2 class="modal-title h5" id="deleteAccountLabel">Delete your account?</h2>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p>This permanently deletes your account and everything attached to it &mdash; articles, comments and bookmarks. It cannot be undone.</p>
                    <p class="small text-muted">Consider <a href="{{ url_for('export_my_data') }}">exporting your data</a> first.</p>
                    <label for="deleteAccountPassword" class="form-label fw-medium">Confirm your password</label>
                    <div class="input-group-icon">
                        <i class="ph ph-lock input-icon" aria-hidden="true"></i>
                        <input type="password" class="form-control" id="deleteAccountPassword" name="password" required autocomplete="current-password">
                    </div>
                </div>
                <div class="modal-footer border-0 pt-0">
                    <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="submit" class="btn btn-danger">Delete permanently</button>
                </div>
            </form>
        </div>
    </div>
</div>
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.stat-item[data-target-tab]').forEach(function (btn) {
        btn.addEventListener('click', function () {
            const realTabBtn = document.getElementById(this.dataset.targetTab);
            if (realTabBtn && typeof bootstrap !== 'undefined') {
                bootstrap.Tab.getOrCreateInstance(realTabBtn).show();
                realTabBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    });
});
</script>
{% endblock %}
"""
PUBLIC_PROFILE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ user.name }}'s Profile - BrieflyAI{% endblock %}
{% block content %}
<div class="animate-fade-in">
    <div class="profile-header-card mb-4">
        <div class="profile-avatar-wrapper">
            <div class="profile-avatar" aria-hidden="true">{{ user.name[0]|upper }}</div>
        </div>
        <h1>{{ user.name }}</h1>
        <p class="username">@{{ user.username }}</p>
        <p class="small text-muted mb-0">Member Since: {{ user.created_at | to_ist }}</p>
    </div>

    <h2 class="section-heading mt-5 mb-4">Articles by {{ user.name }} ({{ posted_articles|length }})</h2>

    <div class="row g-4">
    {% if posted_articles %}
        {% for art in posted_articles %}
        <div class="col-md-6 col-lg-4 d-flex">
            <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ (loop.index0 * 0.05)|round(2) }}s">
                {% set article_url = url_for('article_detail', article_hash_id=art.article_hash_id) %}
                <div class="article-image-container {% if not art.image_url %}img-fallback{% endif %}">
                    <a href="{{ article_url }}" tabindex="-1" aria-hidden="true">{% if art.image_url %}<img src="{{ art.image_url }}" class="article-image" alt="{{ art.title|truncate(50) }}" loading="lazy" decoding="async">{% endif %}</a>
                    <div class="img-fallback-icon" aria-hidden="true"><i class="ph ph-newspaper"></i></div>
                </div>
                <div class="article-body d-flex flex-column">
                    <h3 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                    <div class="article-meta small mb-2">
                        <span class="meta-item text-muted"><i class="ph ph-calendar-blank" aria-hidden="true"></i> {{ art.published_at | to_ist }}</span>
                    </div>
                    <p class="article-description small">{{ art.description|truncate(100) }}</p>
                    <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read more <i class="ph ph-caret-right ms-1 small" aria-hidden="true"></i></a>
                </div>
            </article>
        </div>
        {% endfor %}
    {% else %}
        <div class="col-12">
            <div class="state-card">
                <div class="state-card-icon"><i class="ph ph-note-pencil" aria-hidden="true"></i></div>
                <h2 class="state-card-title">No articles yet</h2>
                <p class="state-card-text">{{ user.name }} hasn't posted any articles yet. Check back soon.</p>
                <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Explore other articles</a></div>
            </div>
        </div>
    {% endif %}
    </div>
</div>
{% endblock %}
"""
ABOUT_US_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}About Us - BrieflyAI{% endblock %}
{% block content %}
<div class="animate-fade-in">
    <div class="page-header-static">
        <h1>About BrieflyAI</h1>
    </div>
    <div class="static-content-container">
        <p class="lead">
            Welcome to BrieflyAI, your premier destination for the latest news from India and around the world, delivered in a concise and easy-to-digest format. We leverage the power of cutting-edge AI to summarize complex news articles into key takeaways, saving you time while keeping you informed.
        </p>

        <h2><i class="icon ph ph-target" aria-hidden="true"></i>Our mission</h2>
        <p>
            In a world of information overload, our mission is to provide clarity and efficiency. We believe that everyone deserves access to accurate, unbiased news without spending hours sifting through lengthy articles. BrieflyAI cuts through the noise, offering insightful summaries that matter.
        </p>

        <h2><i class="icon ph ph-users" aria-hidden="true"></i>Community Hub</h2>
        <p>
            Beyond AI-driven news, BrieflyAI is a platform for discussion and community engagement. Our Community Hub allows users to post their own articles, share perspectives, and engage in meaningful conversations about the topics that shape our world. We are committed to fostering a respectful and intelligent environment for all our members.
        </p>

        <h2><i class="icon ph ph-cpu" aria-hidden="true"></i>Our technology</h2>
        <p>
            We use state-of-the-art Natural Language Processing (NLP) models to analyze and summarize news content from trusted sources. Our system is designed to identify the most crucial points of an article, presenting them as a quick summary and a list of key takeaways, ensuring you get the essence of the story in seconds.
        </p>
    </div>
</div>
{% endblock %}
"""
CONTACT_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Contact Us - BrieflyAI{% endblock %}
{% block content %}
<div class="animate-fade-in">
    <div class="page-header-static">
        <h1>Get in touch</h1>
    </div>
    <div class="static-content-container">
        <p class="lead text-center mb-5">
            We'd love to hear from you. Whether you have a question, feedback, or a news tip, feel free to reach out using one of the methods below.
        </p>
        <div class="row g-4">
            <div class="col-md-6">
                <div class="contact-card">
                    <div class="icon"><i class="ph ph-envelope" aria-hidden="true"></i></div>
                    <h2 class="h5">General inquiries</h2>
                    <p class="text-muted">For general questions, feedback, or support, please email us at:</p>
                    <a href="mailto:vbansal639@gmail.com" class="fw-bold">vbansal639@gmail.com</a>
                </div>
            </div>
            <div class="col-md-6">
                <div class="contact-card">
                    <div class="icon"><i class="ph ph-handshake" aria-hidden="true"></i></div>
                    <h2 class="h5">Partnerships &amp; media</h2>
                    <p class="text-muted">For partnership opportunities or media inquiries, please contact us at:</p>
                    <a href="mailto:vbansal639@gmail.com" class="fw-bold">vbansal639@gmail.com</a>
                </div>
            </div>
        </div>

        <div class="text-center mt-5">
            <h2 class="h3">Follow us</h2>
            <p class="text-muted">Stay connected with us on social media.</p>
            <div class="contact-social-links mt-3">
                <a href="#" title="Twitter" aria-label="BrieflyAI on Twitter"><i class="ph ph-x-logo" aria-hidden="true"></i></a>
                <a href="#" title="Facebook" aria-label="BrieflyAI on Facebook"><i class="ph ph-facebook-logo" aria-hidden="true"></i></a>
                <a href="#" title="LinkedIn" aria-label="BrieflyAI on LinkedIn"><i class="ph ph-linkedin-logo" aria-hidden="true"></i></a>
                <a href="#" title="Instagram" aria-label="BrieflyAI on Instagram"><i class="ph ph-instagram-logo" aria-hidden="true"></i></a>
            </div>
        </div>
    </div>
</div>
{% endblock %}
"""
PRIVACY_POLICY_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Privacy Policy - BrieflyAI{% endblock %}
{% block content %}
<div class="animate-fade-in">
    <div class="page-header-static">
        <h1>Privacy policy</h1>
    </div>
    <div class="static-content-container">
        <p class="text-muted">Last updated: November 6, 2026</p>
        <p>BrieflyAI ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you visit our website.</p>

        <h2><i class="icon ph ph-shield-check" aria-hidden="true"></i>1. Information we collect</h2>
        <p>We may collect personal information that you voluntarily provide to us when you register on the website, post articles or comments, bookmark articles, send a testimonial, or subscribe to our newsletter. This information may include your name, username, email address, and your activities on our platform such as articles posted and bookmarked.</p>

        <h2><i class="icon ph ph-list-checks" aria-hidden="true"></i>2. How we use your information</h2>
        <p>We use the information we collect to:</p>
        <ul>
            <li>Create and manage your account.</li>
            <li>Operate and maintain the website, including your profile page.</li>
            <li>Display your posted and bookmarked articles as part of your profile.</li>
            <li>Send you newsletters or promotional materials, if you have opted in.</li>
            <li>Respond to your comments and inquiries.</li>
            <li>Show testimonials on our website, only after you agree and we have reviewed them.</li>
            <li>Improve our website and services.</li>
        </ul>

        <h2><i class="icon ph ph-share-network" aria-hidden="true"></i>3. Disclosure of your information</h2>
        <p>Your username and posted articles are publicly visible. Your bookmarked articles are visible on your profile page to you when logged in. We do not sell, trade, or otherwise transfer your personally identifiable information like your email address to outside parties without your consent, except to trusted third parties who assist us in operating our website, so long as those parties agree to keep this information confidential.</p>

        <h2><i class="icon ph ph-lock" aria-hidden="true"></i>4. Security of your information</h2>
        <p>We use administrative, technical, and physical security measures to help protect your personal information. While we have taken reasonable steps to secure the personal information you provide to us, please be aware that despite our efforts, no security measures are perfect or impenetrable.</p>

        <h2><i class="icon ph ph-pencil-simple-line" aria-hidden="true"></i>5. Your choices</h2>
        <p>You can review and change your profile information by logging into your account. You may also request deletion of your account and associated data by contacting us.</p>

        {% if adsense_client %}
        <h2><i class="icon ph ph-cookie" aria-hidden="true"></i>6. Advertising and cookies</h2>
        <p>BrieflyAI is free to read because it shows ads served by Google AdSense. Third party vendors, including Google, use cookies to serve ads based on your prior visits to BrieflyAI or other websites. Google's use of advertising cookies enables it and its partners to serve ads to you based on your visits to BrieflyAI and/or other sites on the internet.</p>
        <p>You can opt out of personalised advertising in Google's <a href="https://adssettings.google.com/" target="_blank" rel="noopener noreferrer">Ads Settings</a>, or opt out of some third party vendors' use of cookies for personalised advertising at <a href="https://www.aboutads.info/choices/" target="_blank" rel="noopener noreferrer">www.aboutads.info</a>. To learn how Google uses information from sites that show its ads, see <a href="https://policies.google.com/technologies/partner-sites" target="_blank" rel="noopener noreferrer">How Google uses information from sites or apps that use our services</a>.</p>
        <p>Where the law requires it, for example in the European Economic Area, the United Kingdom and Switzerland, you are asked for consent before personalised ads are shown. BrieflyAI's own cookies only keep you signed in, protect forms from forgery and remember your light or dark mode.</p>
        {% endif %}
        {% if ga_measurement_id %}
        <p>We use Google Analytics to count visits and see which pages are read, which helps us improve the site. Google Analytics sets its own cookies and does not tell us who you are. You can block it with the <a href="https://tools.google.com/dlpage/gaoptout" target="_blank" rel="noopener noreferrer">Google Analytics opt out browser add on</a>.</p>
        {% endif %}

        <h2><i class="icon ph ph-arrows-clockwise" aria-hidden="true"></i>{% if adsense_client %}7{% else %}6{% endif %}. Changes to this privacy policy</h2>
        <p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page. You are advised to review this Privacy Policy periodically for any changes.</p>
    </div>
</div>
{% endblock %}
"""
_AD_SLOT_TEMPLATE = """{% if ads_on_page and ad_slot %}
<aside class="ad-slot ad-slot-{{ ad_name }}" aria-label="Advertisement">
    <span class="ad-slot-label">Advertisement</span>
    <ins class="adsbygoogle" style="display:block" data-ad-client="{{ adsense_client }}" data-ad-slot="{{ ad_slot }}" data-ad-format="auto" data-full-width-responsive="true"></ins>
    <script>(window.adsbygoogle = window.adsbygoogle || []).push({});</script>
</aside>
{% endif %}"""

ADMIN_STORAGE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Storage - BrieflyAI{% endblock %}
{% block content %}
<div class="animate-fade-in">
    <div class="page-header-static">
        <h1>Storage</h1>
        <p class="text-muted mb-0 mt-3">Where BrieflyAI keeps its data, how full it is, and what is cleaned up automatically.</p>
    </div>
    <div class="static-content-container">
        {% set mb = (report.size_bytes / 1048576) if report.size_bytes else None %}
        <h2><i class="icon ph ph-database" aria-hidden="true"></i>{{ report.provider }}</h2>
        {% if mb is not none %}
        <p class="mb-0">{{ '%.1f'|format(mb) }} MB used{% if report.limit_mb %} of {{ report.limit_mb }} MB{% endif %}{% if report.percent is defined %} ({{ report.percent }}%){% endif %}.</p>
        {% if report.percent is defined %}<div class="storage-meter {% if report.percent >= 80 %}is-high{% endif %}" role="img" aria-label="{{ report.percent }}% used"><span style="width: {{ report.percent }}%"></span></div>{% endif %}
        {% endif %}

        <dl class="storage-stats">
            <div class="storage-stat"><dt>Copy from the old database</dt><dd>{% if report.data_copy %}{{ report.data_copy.status|capitalize }}{% if report.data_copy.rows %}, {{ report.data_copy.rows.values()|sum }} rows{% endif %}{% else %}Not needed{% endif %}</dd></div>
            <div class="storage-stat"><dt>Last clean up</dt><dd>{% if report.cleanup %}{{ report.cleanup.at[:10] }}, {{ report.cleanup.summaries }} summaries and {{ report.cleanup.view_counts }} view counts removed{% else %}Not run yet{% endif %}</dd></div>
            <div class="storage-stat"><dt>Kept for</dt><dd>Summaries {{ report.retention.summaries }} days, view counts {{ report.retention.view_counts }} days</dd></div>
        </dl>

        <h2><i class="icon ph ph-table" aria-hidden="true"></i>Tables</h2>
        <div class="table-responsive">
            <table class="storage-table">
                <thead><tr><th scope="col">Table</th><th scope="col" class="num">Rows</th><th scope="col" class="num">Size</th></tr></thead>
                <tbody>
                {% for t in report.tables %}
                    <tr><td>{{ t.name }}</td><td class="num">{{ '{:,}'.format(t.rows) }}</td><td class="num">{% if t.bytes is not none %}{{ '%.1f'|format(t.bytes / 1024) }} KB{% else %}&ndash;{% endif %}</td></tr>
                {% endfor %}
                </tbody>
            </table>
        </div>

        <h2><i class="icon ph ph-broom" aria-hidden="true"></i>Clean up and backup</h2>
        <p>Summaries and view counts older than the periods above are removed once a day, unless a reader bookmarked or discussed the story. You can also run it now. Free databases have no automatic backups, so download a copy now and then. It contains account data, so keep it private.</p>
        <div class="d-flex flex-wrap gap-2">
            <form method="POST" action="{{ url_for('admin_storage_cleanup') }}">
                <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                <button type="submit" class="btn btn-primary"><i class="ph ph-broom me-2" aria-hidden="true"></i>Clean up now</button>
            </form>
            <a class="btn btn-outline-primary" href="{{ url_for('admin_storage_backup') }}"><i class="ph ph-download-simple me-2" aria-hidden="true"></i>Download backup</a>
        </div>
    </div>
</div>
{% endblock %}
"""

ERROR_404_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}404 Not Found - BrieflyAI{% endblock %}{% block content %}
<div class="state-card state-card-narrow animate-fade-in">
    <div class="state-card-icon"><i class="ph ph-signpost" aria-hidden="true"></i></div>
    <h1 class="state-card-title">Page not found</h1>
    <p class="state-card-text">Sorry, the page you're looking for doesn't exist or may have been moved.</p>
    <div class="state-card-actions">
        <a href="{{ url_for('index') }}" class="btn btn-primary-modal"><i class="ph ph-house me-2" aria-hidden="true"></i>Go to homepage</a>
    </div>
</div>
{% endblock %}"""

ERROR_500_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}500 Server Error - BrieflyAI{% endblock %}{% block content %}
<div class="state-card state-card-narrow state-card-danger animate-fade-in">
    <div class="state-card-icon"><i class="ph ph-hard-drives" aria-hidden="true"></i></div>
    <h1 class="state-card-title">Something went wrong</h1>
    <p class="state-card-text">We hit a snag on our end. We've been notified and are looking into it.</p>
    <div class="state-card-actions">
        <a href="{{ url_for('index') }}" class="btn btn-primary-modal"><i class="ph ph-house me-2" aria-hidden="true"></i>Go to homepage</a>
        <button type="button" class="btn btn-outline-secondary" onclick="window.location.reload()"><i class="ph ph-arrow-clockwise me-2" aria-hidden="true"></i>Try again</button>
    </div>
</div>
{% endblock %}"""
OFFLINE_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}Offline - BrieflyAI{% endblock %}{% block content %}
<div class="state-card state-card-narrow animate-fade-in">
    <div class="state-card-icon"><i class="ph ph-wifi-slash" aria-hidden="true"></i></div>
    <h1 class="state-card-title">You're offline</h1>
    <p class="state-card-text">We couldn't reach the network. Any pages you've already opened are still available, and new stories will load as soon as you're back online.</p>
    <div class="state-card-actions">
        <button type="button" class="btn btn-primary-modal" onclick="window.location.reload()"><i class="ph ph-arrow-clockwise me-2" aria-hidden="true"></i>Try again</button>
        <a href="{{ url_for('index') }}" class="btn btn-outline-secondary"><i class="ph ph-house me-2" aria-hidden="true"></i>Homepage</a>
    </div>
</div>
{% endblock %}"""

TERMS_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Terms of Service - BrieflyAI{% endblock %}
{% block meta_description %}The rules for using BrieflyAI: AI summaries, accounts, community posts and testimonials.{% endblock %}
{% block content %}
<div class="animate-fade-in">
    <div class="page-header-static">
        <h1>Terms of service</h1>
    </div>
    <div class="static-content-container">
        <p class="text-muted">Last updated: {{ terms_updated }}</p>
        <p>These terms explain the rules for using BrieflyAI. By using the site you agree to them. If you do not agree, please do not use BrieflyAI. Our <a href="{{ url_for('privacy') }}">privacy policy</a> explains how we handle your information.</p>

        <h2><i class="icon ph ph-newspaper" aria-hidden="true"></i>1. What BrieflyAI is</h2>
        <p>BrieflyAI gathers news stories from public publisher feeds and news services and uses artificial intelligence to write short summaries and key takeaways. It also hosts a Community Hub where registered readers can post articles and comments. BrieflyAI is free to use.</p>

        <h2><i class="icon ph ph-warning" aria-hidden="true"></i>2. AI summaries can be wrong</h2>
        <p>Summaries and takeaways are generated automatically. They can be incomplete, out of date or simply wrong. They are general information only and are not legal, financial, medical or any other professional advice. Read the original article before relying on a story, and do not make important decisions based on a summary alone.</p>

        <h2><i class="icon ph ph-buildings" aria-hidden="true"></i>3. News belongs to its publishers</h2>
        <p>Headlines, descriptions and articles shown or linked on BrieflyAI belong to their publishers. BrieflyAI links to the original source and does not claim ownership of publisher content. If you believe something on BrieflyAI infringes your rights, email <a href="mailto:{{ contact_email }}">{{ contact_email }}</a> with the page link and the details, and we will review it.</p>

        <h2><i class="icon ph ph-user-circle" aria-hidden="true"></i>4. Your account</h2>
        <p>You only need an account to bookmark, comment, react, post or share a testimonial. Keep your password safe, because you are responsible for what happens under your account. Do not impersonate anyone. You can export your data or delete your account at any time from your profile.</p>

        <h2><i class="icon ph ph-chats" aria-hidden="true"></i>5. What you post</h2>
        <p>You keep ownership of the articles, comments and testimonials you submit. By posting, you allow BrieflyAI to store and show that content on the site and in its feeds, such as RSS, for as long as it stays on BrieflyAI. You confirm you have the right to post it.</p>
        <p>When you send a testimonial and agree to publish it, you allow BrieflyAI to show it on the website with the name and details you gave. We review every testimonial before it appears, never change your words without asking, and remove it whenever you ask.</p>

        <h2><i class="icon ph ph-shield-check" aria-hidden="true"></i>6. Community rules</h2>
        <p>Do not post anything that:</p>
        <ul>
            <li>is unlawful, defamatory, hateful, harassing or threatening</li>
            <li>infringes someone else's copyright, trademark or privacy</li>
            <li>is spam, advertising or deliberately misleading</li>
            <li>contains malware or links to harmful sites</li>
        </ul>
        <p>Content that breaks these rules may be removed, and accounts that keep breaking them may be suspended or deleted. You can report community articles from the article page.</p>

        <h2><i class="icon ph ph-lock" aria-hidden="true"></i>7. Using the site fairly</h2>
        <p>Do not try to break, overload or get around the security of BrieflyAI, including its rate limits. Do not copy the site's content in bulk or resell it without written permission.</p>

        <h2><i class="icon ph ph-arrows-clockwise" aria-hidden="true"></i>8. Changes</h2>
        <p>BrieflyAI may change, pause or stop any feature at any time. We may also update these terms. The date at the top shows the current version, and using the site after a change means you accept the updated terms.</p>

        <h2><i class="icon ph ph-scales" aria-hidden="true"></i>9. No warranty and limits of liability</h2>
        <p>BrieflyAI is provided as is, without warranties of any kind. As far as the law allows, BrieflyAI and its operator are not liable for any loss or damage that comes from using the site, relying on a summary, or not being able to use the site.</p>

        <h2><i class="icon ph ph-globe" aria-hidden="true"></i>10. Governing law</h2>
        <p>These terms are governed by the laws of India.</p>

        <h2><i class="icon ph ph-envelope-simple" aria-hidden="true"></i>11. Contact</h2>
        <p>Questions about these terms? Email <a href="mailto:{{ contact_email }}">{{ contact_email }}</a> or use the <a href="{{ url_for('contact') }}">contact page</a>.</p>
    </div>
</div>
{% endblock %}
"""
SHARE_TESTIMONIAL_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Share your experience - BrieflyAI{% endblock %}
{% block content %}
<div class="animate-fade-in testimonial-page">
    <div class="page-header-static">
        <h1>Share your experience</h1>
        <p class="text-muted mb-0 mt-2">Tell other readers what BrieflyAI does for you. We read every note before it appears on the website.</p>
    </div>
    <div class="static-content-container testimonial-form-wrap">
        {% if mine %}
        <div class="mb-4">
            <h2 class="h6 mb-3">Your notes</h2>
            <ul class="list-unstyled ps-0 d-flex flex-column gap-2 mb-0">
                {% for t in mine %}
                <li class="testimonial-mine">
                    <p class="mb-1">{{ t.quote }}</p>
                    <span class="testimonial-status testimonial-status-{{ t.status }}">
                        {% if t.status == 'approved' %}<i class="ph ph-check-circle" aria-hidden="true"></i> Live on the website
                        {% elif t.status == 'pending' %}<i class="ph ph-hourglass-medium" aria-hidden="true"></i> Waiting for review
                        {% else %}<i class="ph ph-eye-slash" aria-hidden="true"></i> Not published{% endif %}
                    </span>
                </li>
                {% endfor %}
            </ul>
            <p class="small text-muted mt-2 mb-0">Want a note taken down? <a href="{{ url_for('contact') }}">Contact us</a> and we will remove it.</p>
        </div>
        {% endif %}

        {% if has_pending %}
        <div class="state-card state-card-narrow">
            <div class="state-card-icon"><i class="ph ph-hourglass-medium" aria-hidden="true"></i></div>
            <h2 class="state-card-title h4">Your note is waiting for review</h2>
            <p class="state-card-text">You can send another one once this one has been reviewed.</p>
        </div>
        {% else %}
        <form method="POST" action="{{ url_for('share_testimonial') }}" class="needs-validation" id="testimonialForm" novalidate>
            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
            <div class="mb-3">
                <label for="tName" class="form-label fw-medium">Name to show</label>
                <input type="text" class="form-control" id="tName" name="name" value="{{ form.name }}" maxlength="{{ limits.name }}" required autocomplete="name">
                <div class="invalid-feedback">Add the name you want shown with your note.</div>
            </div>
            <div class="mb-3">
                <label for="tAttribution" class="form-label fw-medium">About you <span class="text-muted fw-normal">(optional)</span></label>
                <input type="text" class="form-control" id="tAttribution" name="attribution" value="{{ form.attribution }}" maxlength="{{ limits.attribution }}" placeholder="For example: Student, Pune">
            </div>
            <div class="mb-3">
                <label for="tQuote" class="form-label fw-medium">Your note</label>
                <textarea class="form-control" id="tQuote" name="quote" rows="4" maxlength="{{ limits.quote }}" minlength="20" required aria-describedby="tQuoteCount">{{ form.quote }}</textarea>
                <div class="d-flex justify-content-between mt-1">
                    <div class="invalid-feedback d-block-when-invalid">Write at least 20 characters.</div>
                    <span class="small text-muted ms-auto" id="tQuoteCount" aria-live="polite">0 / {{ limits.quote }}</span>
                </div>
            </div>
            <div class="form-check mb-4">
                <input class="form-check-input" type="checkbox" value="yes" id="tConsent" name="consent" required>
                <label class="form-check-label" for="tConsent">I agree that BrieflyAI may publish this note on its website with the name and details above. I can ask for it to be removed at any time.</label>
                <div class="invalid-feedback">Please agree so we can publish your note.</div>
            </div>
            <button type="submit" class="btn btn-primary">Send for review</button>
        </form>
        {% endif %}
    </div>
</div>
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('testimonialForm');
    if (!form) { return; }
    var quote = document.getElementById('tQuote');
    var count = document.getElementById('tQuoteCount');
    var max = quote.getAttribute('maxlength');
    var update = function () { count.textContent = quote.value.length + ' / ' + max; };
    quote.addEventListener('input', update);
    update();
    form.addEventListener('submit', function (e) {
        if (!form.checkValidity()) {
            e.preventDefault();
            e.stopPropagation();
            var firstInvalid = form.querySelector(':invalid');
            if (firstInvalid) { firstInvalid.focus(); }
        }
        form.classList.add('was-validated');
    });
});
</script>
{% endblock %}
"""
ADMIN_TESTIMONIALS_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Testimonials - Admin - BrieflyAI{% endblock %}
{% block content %}
<div class="animate-fade-in testimonial-page">
    <div class="page-header-static">
        <h1>Testimonials</h1>
        <p class="text-muted mb-0 mt-2">Approved notes appear on the <a href="{{ url_for('landing') }}#readers">landing page</a>, newest first.</p>
    </div>

    <div class="static-content-container mb-4">
        <h2 class="h5 mb-3"><i class="ph ph-plus-circle me-2" aria-hidden="true"></i>Add a quote you received</h2>
        <p class="small text-muted">For notes that reached you by email or message. Only add a quote with the reader's permission, in their own words.</p>
        <form method="POST" action="{{ url_for('admin_add_testimonial') }}" class="needs-validation" id="adminTestimonialForm" novalidate>
            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
            <div class="row g-3">
                <div class="col-md-6">
                    <label for="aName" class="form-label fw-medium">Reader's name</label>
                    <input type="text" class="form-control" id="aName" name="name" maxlength="{{ limits.name }}" required>
                    <div class="invalid-feedback">Name is required.</div>
                </div>
                <div class="col-md-6">
                    <label for="aAttribution" class="form-label fw-medium">About them <span class="text-muted fw-normal">(optional)</span></label>
                    <input type="text" class="form-control" id="aAttribution" name="attribution" maxlength="{{ limits.attribution }}" placeholder="For example: Teacher, Jaipur">
                </div>
                <div class="col-12">
                    <label for="aQuote" class="form-label fw-medium">Quote</label>
                    <textarea class="form-control" id="aQuote" name="quote" rows="3" maxlength="{{ limits.quote }}" minlength="20" required></textarea>
                    <div class="invalid-feedback">Quote must be at least 20 characters.</div>
                </div>
                <div class="col-12 form-check ms-1">
                    <input class="form-check-input" type="checkbox" value="yes" id="aPermission" name="permission" required>
                    <label class="form-check-label" for="aPermission">This reader gave permission to publish this quote with their name.</label>
                    <div class="invalid-feedback">Permission is required.</div>
                </div>
            </div>
            <button type="submit" class="btn btn-primary mt-3">Add and publish</button>
        </form>
    </div>

    {% for group, label, empty in [('pending', 'Waiting for review', 'Nothing to review.'), ('approved', 'Live on the website', 'No approved notes yet.'), ('rejected', 'Not published', 'Nothing here.')] %}
    <section class="mb-4" aria-labelledby="group-{{ group }}">
        <h2 class="section-heading h5 mb-3" id="group-{{ group }}">{{ label }} <span class="text-muted fw-normal">({{ groups[group]|length }})</span></h2>
        {% if groups[group] %}
        <div class="row g-3">
            {% for t in groups[group] %}
            <div class="col-md-6 d-flex">
                <div class="contact-card text-start w-100 d-flex flex-column">
                    <p class="mb-2">{{ t.quote }}</p>
                    <p class="small text-muted mb-3">{{ t.name }}{% if t.attribution %}, {{ t.attribution }}{% endif %} · {{ 'Sent by reader @' ~ t.user.username if t.user else 'Added by admin' }} · {{ t.created_at | to_ist }}</p>
                    <div class="d-flex flex-wrap gap-2 mt-auto">
                        {% if t.status != 'approved' %}
                        <form method="POST" action="{{ url_for('admin_review_testimonial', testimonial_id=t.id, action='approve') }}"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><button class="btn btn-sm btn-primary" type="submit"><i class="ph ph-check me-1" aria-hidden="true"></i>Approve</button></form>
                        {% endif %}
                        {% if t.status != 'rejected' %}
                        <form method="POST" action="{{ url_for('admin_review_testimonial', testimonial_id=t.id, action='reject') }}"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><button class="btn btn-sm btn-outline-secondary" type="submit"><i class="ph ph-eye-slash me-1" aria-hidden="true"></i>{{ 'Unpublish' if t.status == 'approved' else 'Reject' }}</button></form>
                        {% endif %}
                        <form method="POST" action="{{ url_for('admin_review_testimonial', testimonial_id=t.id, action='delete') }}"><input type="hidden" name="csrf_token" value="{{ csrf_token() }}"><button class="btn btn-sm btn-outline-danger" type="submit"><i class="ph ph-trash me-1" aria-hidden="true"></i>Delete</button></form>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        {% else %}
        <p class="text-muted">{{ empty }}</p>
        {% endif %}
    </section>
    {% endfor %}
</div>
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('adminTestimonialForm');
    if (!form) { return; }
    form.addEventListener('submit', function (e) {
        if (!form.checkValidity()) { e.preventDefault(); e.stopPropagation(); }
        form.classList.add('was-validated');
    });
});
</script>
{% endblock %}
"""

# Landing page (/welcome). A raw string, so the CSS escapes in class names like
# .md\:heading-2 reach the browser untouched.
LANDING_HTML_TEMPLATE = r"""<!doctype html>
{#
  BrieflyAI landing page, served at "/". The news feed lives at /news.

  Standalone on purpose: it follows the landing page design system (one typeface: Geist,
  Phosphor icons, fixed spacing and type scales) instead of the Bootstrap based site shell.
  Text styles are defined once below as heading-*, title-*, body-*, button-*, caption-*
  and applied by name. Responsive variants use the md: (768px) and lg: (1024px) prefixes.

  Artwork: hand coloured Photochrom prints of India, c. 1890, from the Library of Congress
  Prints and Photographs Division (Photochrom Print Collection, no known restrictions).
#}
<html lang="en" data-theme="{{ theme }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>BrieflyAI: India's news, one minute per story</title>
    <meta name="description" content="Short AI summaries and key takeaways for India's top stories from {{ sources_short }} and more. Free to read, no account needed.">
    <link rel="canonical" href="{{ canonical_url }}">
    {% if theme == 'dark' %}<meta name="theme-color" content="#131209">{% elif theme == 'light' %}<meta name="theme-color" content="#F7F5EF">{% else %}<meta name="theme-color" content="#F7F5EF" media="(prefers-color-scheme: light)">
    <meta name="theme-color" content="#131209" media="(prefers-color-scheme: dark)">{% endif %}
    <link rel="manifest" href="{{ url_for('web_manifest') }}">
    {% if adsense_client %}<meta name="google-adsense-account" content="{{ adsense_client }}">{% endif %}
    {% if ads_on_page %}<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={{ adsense_client }}" crossorigin="anonymous"></script>{% endif %}
    <link rel="icon" href="{{ url_for('app_icon', size=192, v=2) }}" type="image/svg+xml">

    <meta property="og:site_name" content="BrieflyAI">
    <meta property="og:type" content="website">
    <meta property="og:title" content="BrieflyAI: India's news, one minute per story">
    <meta property="og:description" content="Every story gets a neutral summary of under 60 words and three or four key takeaways, with the original article one tap away. Free to read.">
    <meta property="og:url" content="{{ canonical_url }}">
    {% if og_image_url %}
    <meta property="og:image" content="{{ og_image_url }}">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:image:alt" content="BrieflyAI: India's news, one minute per story">
    <meta name="twitter:image" content="{{ og_image_url }}">
    {% endif %}
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="BrieflyAI: India's news, one minute per story">
    <meta name="twitter:description" content="Short AI summaries and key takeaways for India's top stories. Free to read, no account needed.">

    <script>document.documentElement.classList.add('js');</script>
    <script type="application/ld+json">{{ faq_schema|tojson }}</script>

    <link rel="preload" as="image" type="image/webp" fetchpriority="high"
          href="{{ url_for('static', filename='landing/hero-nirmahal-ghat-2400.webp') }}"
          imagesrcset="{{ url_for('static', filename='landing/hero-nirmahal-ghat-mobile-900.webp') }} 900w, {{ url_for('static', filename='landing/hero-nirmahal-ghat-1200.webp') }} 1200w, {{ url_for('static', filename='landing/hero-nirmahal-ghat-2400.webp') }} 2400w"
          imagesizes="100vw">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&display=swap">

    <style>
        /* ==========================================================================
           TOKENS
           ========================================================================== */
        :root {
            /* Spacing: the only values allowed for padding, margin and gap. */
            --spacing-0: 0; --spacing-25: 2px; --spacing-50: 4px; --spacing-75: 8px;
            --spacing-100: 12px; --spacing-200: 16px; --spacing-300: 24px; --spacing-400: 32px;
            --spacing-500: 40px; --spacing-600: 48px; --spacing-700: 64px; --spacing-800: 80px;
            --spacing-900: 96px;

            /* Radius: Tailwind steps. Nested shapes use outer minus gap when the gap is under 32px. */
            --rounded-md: 6px; --rounded-lg: 8px; --rounded-xl: 12px; --rounded-2xl: 16px;
            --rounded-3xl: 24px; --rounded-full: 9999px;

            /* Motion */
            --ease: cubic-bezier(0.22, 1, 0.36, 1);
            --duration-layout: 500ms;
            --duration-color: 300ms;

            --font-sans: 'Geist', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
            --grain: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0.08 0 0 0 0 0.07 0 0 0 0 0.04 0 0 0 0.9 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");

            /* Light theme: warm paper that matches the prints. */
            color-scheme: light;
            --bg: #F7F5EF;
            --surface: #EFECE4;
            --surface-raised: #FBFAF6;
            --surface-muted: #E6E2D8;
            --border: #DEDACF;
            --border-strong: #16140F;
            --text: #16140F;
            --text-muted: #5A564C;
            --text-subtle: #777266;
            --accent: #12645E;
            --logo-gist: #6CC9BD;
            --btn-bg: #16140F;
            --btn-text: #F7F5EF;
            --btn-hover: #2E2B23;
            --btn-ghost-hover: #E6E2D8;
            --nav-bg: rgba(247, 245, 239, 0.72);
            --panel-bg: rgba(247, 245, 239, 0.94);
            --hero-from: #000000;
            --hero-to: #666666;
            --card-shadow: 0 32px 64px -32px rgba(56, 44, 20, 0.35);
            --photo-filter: none;
            --photo-veil: rgba(247, 245, 239, 0);
            --on-photo: #16140F;
            --on-photo-muted: #3B372E;
        }
        :root[data-theme="dark"] {
            color-scheme: dark;
            --bg: #131209;
            --surface: #181818;
            --surface-raised: #1F1F1F;
            --surface-muted: #272727;
            --border: #313131;
            --border-strong: #F2EFE6;
            --text: #F2EFE6;
            --text-muted: #ABA697;
            --text-subtle: #8C887B;
            --accent: #6CC9BD; --logo-gist: #12645E;
            --btn-bg: #F2EFE6;
            --btn-text: #131209;
            --btn-hover: #DAD6CB;
            --btn-ghost-hover: #272727;
            --nav-bg: rgba(19, 18, 9, 0.66);
            --panel-bg: rgba(19, 18, 9, 0.92);
            --hero-from: #FFFFFF;
            --hero-to: #9B9B9B;
            --card-shadow: none;
            --photo-filter: brightness(0.5) saturate(0.85);
            --photo-veil: rgba(19, 18, 9, 0.2);
            --on-photo: #F2EFE6;
            --on-photo-muted: #D9D4C6;
        }
        @media (prefers-color-scheme: dark) {
            :root[data-theme="auto"] {
                color-scheme: dark;
                --bg: #131209;
                --surface: #181818;
                --surface-raised: #1F1F1F;
                --surface-muted: #272727;
                --border: #313131;
                --border-strong: #F2EFE6;
                --text: #F2EFE6;
                --text-muted: #ABA697;
                --text-subtle: #8C887B;
                --accent: #6CC9BD; --logo-gist: #12645E;
                --btn-bg: #F2EFE6;
                --btn-text: #131209;
                --btn-hover: #DAD6CB;
                --btn-ghost-hover: #272727;
                --nav-bg: rgba(19, 18, 9, 0.66);
                --panel-bg: rgba(19, 18, 9, 0.92);
                --hero-from: #FFFFFF;
                --hero-to: #9B9B9B;
                --card-shadow: none;
                --photo-filter: brightness(0.5) saturate(0.85);
                --photo-veil: rgba(19, 18, 9, 0.2);
                --on-photo: #F2EFE6;
                --on-photo-muted: #D9D4C6;
            }
        }

        /* ==========================================================================
           TEXT STYLES (Tailwind type scale only)
           ========================================================================== */
        .heading-1 { font-size: 3.75rem; line-height: 1; font-weight: 600; letter-spacing: -0.03em; }
        .heading-2 { font-size: 3rem; line-height: 1; font-weight: 600; letter-spacing: -0.03em; }
        .heading-3 { font-size: 2.25rem; line-height: 2.5rem; font-weight: 600; letter-spacing: -0.025em; }
        .heading-4 { font-size: 1.875rem; line-height: 2.25rem; font-weight: 600; letter-spacing: -0.02em; }
        .heading-5 { font-size: 1.5rem; line-height: 2rem; font-weight: 600; letter-spacing: -0.015em; }
        .heading-6 { font-size: 1.25rem; line-height: 1.75rem; font-weight: 600; }
        .title-1 { font-size: 1.25rem; line-height: 1.75rem; font-weight: 500; letter-spacing: -0.01em; }
        .title-2 { font-size: 1.125rem; line-height: 1.75rem; font-weight: 500; }
        .title-3 { font-size: 1rem; line-height: 1.5rem; font-weight: 500; }
        .title-4 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; }
        .title-5 { font-size: 0.75rem; line-height: 1rem; font-weight: 500; }
        .body-1 { font-size: 1rem; line-height: 1.5rem; font-weight: 600; }
        .body-2 { font-size: 1rem; line-height: 1.5rem; font-weight: 500; }
        .body-3 { font-size: 1rem; line-height: 1.5rem; font-weight: 400; }
        .body-4 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 600; }
        .body-5 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; }
        .body-6 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 400; }
        .body-7 { font-size: 1.125rem; line-height: 1.75rem; font-weight: 400; }
        .button-lg { font-size: 1.125rem; line-height: 1.75rem; font-weight: 600; }
        .button-md { font-size: 1rem; line-height: 1.5rem; font-weight: 600; }
        .button-sm { font-size: 0.875rem; line-height: 1.25rem; font-weight: 600; }
        .button-xsm { font-size: 0.75rem; line-height: 1rem; font-weight: 600; }
        .caption-1 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 500; }
        .caption-2 { font-size: 0.875rem; line-height: 1.25rem; font-weight: 400; }
        .caption-3 { font-size: 0.75rem; line-height: 1rem; font-weight: 600; letter-spacing: 0.04em; }
        .caption-4 { font-size: 0.75rem; line-height: 1rem; font-weight: 500; }
        .caption-5 { font-size: 0.75rem; line-height: 1rem; font-weight: 400; }
        @media (min-width: 768px) {
            .md\:heading-2 { font-size: 3rem; line-height: 1; font-weight: 600; letter-spacing: -0.03em; }
            .md\:heading-3 { font-size: 2.25rem; line-height: 2.5rem; font-weight: 600; letter-spacing: -0.025em; }
            .md\:body-7 { font-size: 1.125rem; line-height: 1.75rem; font-weight: 400; }
        }
        @media (min-width: 1024px) {
            .lg\:heading-1 { font-size: 3.75rem; line-height: 1; font-weight: 600; letter-spacing: -0.03em; }
            .lg\:heading-2 { font-size: 3rem; line-height: 1; font-weight: 600; letter-spacing: -0.03em; }
        }

        /* ==========================================================================
           BASE
           ========================================================================== */
        *, *::before, *::after { box-sizing: border-box; }
        html { -webkit-text-size-adjust: 100%; scroll-behavior: smooth; scroll-padding-top: var(--spacing-900); }
        body {
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font-family: var(--font-sans);
            font-size: 1rem;
            line-height: 1.5rem;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            overflow-x: clip;
        }
        h1, h2, h3, h4, h5, h6, p, ul, ol, figure, dl, dd, blockquote { margin: 0; }
        ul, ol { padding: 0; list-style: none; }
        a { color: inherit; text-decoration: none; }
        img { display: block; max-width: 100%; }
        button { font: inherit; color: inherit; background: none; border: 0; padding: 0; cursor: pointer; }
        h1, h2, h3, h4, h5, h6, [class*="heading-"], [class*="title-"] { text-wrap: balance; }
        p, li, dd, dt, figcaption { text-wrap: pretty; }
        em, i, cite, blockquote { font-style: normal; }
        ::selection { background: var(--text); color: var(--bg); }
        :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: var(--rounded-md); }

        .icon { width: 20px; height: 20px; flex: none; fill: currentColor; }
        .logo-mark { display: block; width: 24px; height: 24px; flex: none; }
        .logo-tile { fill: var(--text); }
        .logo-line { fill: var(--bg); }
        .logo-gist { fill: var(--logo-gist); }
        .icon-sm { width: 16px; height: 16px; }
        .icon-lg { width: 24px; height: 24px; }

        .container { width: 100%; max-width: 1232px; margin-inline: auto; padding-inline: var(--spacing-300); }
        @media (min-width: 768px) { .container { padding-inline: var(--spacing-400); } }
        @media (min-width: 1280px) { .container { padding-inline: var(--spacing-500); } }

        .skip-link {
            position: absolute; left: var(--spacing-200); top: var(--spacing-200); z-index: 60;
            padding: var(--spacing-75) var(--spacing-100); border-radius: var(--rounded-lg);
            background: var(--btn-bg); color: var(--btn-text); transform: translateY(-200%);
        }
        .skip-link:focus { transform: none; }

        .muted { color: var(--text-muted); }
        .subtle { color: var(--text-subtle); }
        .accent { color: var(--accent); }
        .eyebrow { display: inline-flex; align-items: center; gap: var(--spacing-75); color: var(--accent); }
        .eyebrow::before { content: ""; width: 24px; height: 1px; background: currentColor; }

        /* Photographs: one treatment everywhere (theme filter, flat veil, film grain). */
        .photo { position: relative; overflow: hidden; isolation: isolate; background: var(--surface-muted); }
        .photo img { width: 100%; height: 100%; object-fit: cover; filter: var(--photo-filter); }
        .photo::before { content: ""; position: absolute; inset: 0; z-index: 1; background: var(--photo-veil); pointer-events: none; }
        .photo::after { content: ""; position: absolute; inset: 0; z-index: 1; background-image: var(--grain); opacity: 0.14; mix-blend-mode: multiply; pointer-events: none; }
        .credit { color: var(--text-subtle); }
        .credit-on-photo {
            position: absolute; z-index: 2; display: inline-flex; align-items: center; gap: var(--spacing-50);
            padding: var(--spacing-50) var(--spacing-75); border-radius: var(--rounded-md);
            background: var(--nav-bg); color: var(--text-muted);
            -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
        }

        /* ==========================================================================
           BUTTONS
           ========================================================================== */
        .btn {
            display: inline-flex; align-items: center; justify-content: center; gap: var(--spacing-75);
            padding: var(--spacing-75) var(--spacing-100);
            border-radius: var(--rounded-lg); border: 1px solid transparent; white-space: nowrap;
            transition: background-color var(--duration-color) var(--ease), color var(--duration-color) var(--ease),
                        border-color var(--duration-color) var(--ease), transform var(--duration-layout) var(--ease);
        }
        .btn .icon { transition: transform var(--duration-layout) var(--ease); }
        .btn:active { transform: scale(0.98); }
        .btn-primary { background: var(--btn-bg); color: var(--btn-text); }
        .btn-primary:hover { background: var(--btn-hover); }
        .btn-primary:hover .icon-arrow { transform: translateX(var(--spacing-25)); }
        .btn-ghost { color: var(--text); border-color: var(--border); background: var(--surface-raised); }
        .btn-ghost:hover { background: var(--btn-ghost-hover); }

        .text-link {
            display: inline-flex; align-items: center; gap: var(--spacing-50);
            color: var(--text); text-decoration: underline; text-decoration-color: var(--border);
            text-underline-offset: 4px;
            transition: text-decoration-color var(--duration-color) var(--ease), color var(--duration-color) var(--ease);
        }
        .text-link:hover { text-decoration-color: currentColor; }
        .text-link .icon { transition: transform var(--duration-layout) var(--ease); }
        .text-link:hover .icon { transform: translate(var(--spacing-25), calc(var(--spacing-25) * -1)); }

        /* ==========================================================================
           NAVIGATION
           ========================================================================== */
        .site-nav {
            position: fixed; top: var(--spacing-200); left: 50%; z-index: 50; transform: translateX(-50%);
            display: flex; align-items: center; gap: var(--spacing-75);
            width: max-content; max-width: calc(100% - var(--spacing-400));
            padding: var(--spacing-75); border-radius: var(--rounded-2xl);
            border: 1px solid var(--border); background: var(--nav-bg);
            -webkit-backdrop-filter: blur(16px) saturate(140%); backdrop-filter: blur(16px) saturate(140%);
        }
        .nav-brand {
            display: inline-flex; align-items: center; gap: var(--spacing-75);
            padding: var(--spacing-75) var(--spacing-100); border-radius: var(--rounded-lg); /* 16 minus 8 */
            transition: background-color var(--duration-color) var(--ease);
        }
        .nav-brand:hover { background: var(--btn-ghost-hover); }
        .nav-brand .icon { color: var(--accent); }
        .nav-links { display: none; align-items: center; gap: var(--spacing-25); }
        .nav-links a {
            display: block; padding: var(--spacing-75) var(--spacing-100); border-radius: var(--rounded-lg); color: var(--text-muted);
            transition: background-color var(--duration-color) var(--ease), color var(--duration-color) var(--ease);
        }
        .nav-links a:hover { color: var(--text); background: var(--btn-ghost-hover); }
        .nav-links a.is-current { color: var(--text); background: var(--surface-muted); }
        .nav-cta { display: none; }
        .nav-toggle {
            position: relative; display: inline-flex; align-items: center; justify-content: center;
            width: 40px; height: 40px; border-radius: var(--rounded-lg);
            transition: background-color var(--duration-color) var(--ease);
        }
        .nav-toggle:hover { background: var(--btn-ghost-hover); }
        .nav-toggle-line {
            position: absolute; left: 12px; right: 12px; height: 1.5px; border-radius: var(--rounded-full);
            background: currentColor; transition: transform var(--duration-layout) var(--ease);
        }
        .nav-toggle-line:first-child { transform: translateY(-4px); }
        .nav-toggle-line:last-child { transform: translateY(4px); }
        .nav-toggle[aria-expanded="true"] .nav-toggle-line:first-child { transform: rotate(45deg); }
        .nav-toggle[aria-expanded="true"] .nav-toggle-line:last-child { transform: rotate(-45deg); }
        @media (min-width: 900px) {
            .nav-links { display: flex; }
            .nav-cta { display: inline-flex; }
            .nav-toggle { display: none; }
        }

        .mobile-menu {
            position: fixed; inset: 0; z-index: 40;
            display: flex; flex-direction: column; justify-content: space-between; gap: var(--spacing-400);
            padding: var(--spacing-900) var(--spacing-300) var(--spacing-400);
            background: var(--panel-bg); -webkit-backdrop-filter: blur(24px); backdrop-filter: blur(24px);
            opacity: 0; transition: opacity var(--duration-color) var(--ease); overflow-y: auto;
        }
        .mobile-menu[hidden] { display: none; }
        .mobile-menu.is-open { opacity: 1; }
        .mobile-menu-list { display: flex; flex-direction: column; gap: var(--spacing-75); }
        .mobile-menu-list a { display: block; padding: var(--spacing-75) 0; }
        .menu-item { opacity: 0; transform: translateY(var(--spacing-300)); transition: opacity var(--duration-layout) var(--ease), transform var(--duration-layout) var(--ease); }
        .mobile-menu.is-open .menu-item { opacity: 1; transform: none; }
        .mobile-menu .btn { width: 100%; }
        html.menu-open { overflow: hidden; }

        /* ==========================================================================
           HERO: a full bleed print, headline on the sky, a real summary overlapping the edge
           ========================================================================== */
        .hero { position: relative; isolation: isolate; min-height: max(640px, min(100svh, 960px)); color: var(--on-photo); }
        .hero-media { position: absolute; inset: 0; z-index: -1; }
        .hero-media img { object-position: 50% 0%; transform: scale(1.04); transition: transform 1400ms var(--ease); }
        .is-loaded .hero-media img { transform: none; }
        .hero-content { position: relative; padding-top: var(--spacing-900); padding-bottom: var(--spacing-900); }
        @media (min-width: 768px) { .hero-copy { padding-top: var(--spacing-600); } }
        .hero-copy { max-width: 680px; }
        .js .hero-copy > * { opacity: 0; transform: translateY(var(--spacing-200)); transition: opacity 700ms var(--ease), transform 700ms var(--ease); }
        .is-loaded .hero-copy > * { opacity: 1; transform: none; }
        .is-loaded .hero-copy > :nth-child(2) { transition-delay: 80ms; }
        .is-loaded .hero-copy > :nth-child(3) { transition-delay: 160ms; }
        .is-loaded .hero-copy > :nth-child(4) { transition-delay: 240ms; }
        .is-loaded .hero-copy > :nth-child(5) { transition-delay: 320ms; }
        .hero-badge {
            display: inline-flex; align-items: center; gap: var(--spacing-75);
            padding: var(--spacing-50) var(--spacing-100); border-radius: var(--rounded-full);
            background: var(--nav-bg); color: var(--text);
            -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
        }
        .live-dot { width: 8px; height: 8px; border-radius: var(--rounded-full); background: var(--accent); animation: pulse 2400ms var(--ease) infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
        .hero-title { margin-top: var(--spacing-300); }
        .hero-title-text {
            background: linear-gradient(90deg, var(--hero-from), var(--hero-to));
            -webkit-box-decoration-break: clone; box-decoration-break: clone;
            -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent;
        }
        .hero-sub { margin-top: var(--spacing-300); max-width: 560px; color: var(--on-photo-muted); }
        .hero-actions { margin-top: var(--spacing-400); display: flex; flex-wrap: wrap; align-items: center; gap: var(--spacing-100) var(--spacing-300); }
        /* Frosted backing wherever small text could cross the bridge or the boats. */
        .frost { background: var(--nav-bg); -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px); }
        .hero-note { padding: var(--spacing-50) var(--spacing-75); border-radius: var(--rounded-md); color: var(--text-muted); }
        .hero-proof { display: inline-block; margin-top: var(--spacing-400); padding: var(--spacing-100) var(--spacing-200); border-radius: var(--rounded-xl); color: var(--text-muted); }
        .hero-proof-short { display: none; }
        @media (max-width: 767px) { .hero-proof-short { display: block; color: var(--text); } }
        .hero-sources { margin-top: var(--spacing-75); display: flex; flex-wrap: wrap; gap: var(--spacing-75) var(--spacing-200); color: var(--text); }
        .hero-credit { left: var(--spacing-300); bottom: calc(var(--spacing-900) + var(--spacing-200)); }
        @media (min-width: 768px) { .hero-credit { left: var(--spacing-400); bottom: var(--spacing-300); } }
        /* Small screens: the text crosses the river, so it sits on a frosted panel. */
        @media (max-width: 767px) {
            .hero-copy { padding: var(--spacing-300); border-radius: var(--rounded-2xl); background: var(--nav-bg); -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px); }
            .hero-copy .frost { background: none; -webkit-backdrop-filter: none; backdrop-filter: none; padding-inline: 0; }
            .hero-proof { margin-top: var(--spacing-200); padding-block: 0; }
            .hero-proof:has(.hero-proof-short) .hero-proof-label, .hero-proof:has(.hero-proof-short) .hero-sources { display: none; }
        }

        /* Story card: real product UI that overlaps the photograph. */
        .hero-card-wrap { position: relative; z-index: 3; margin-top: calc(var(--spacing-900) * -1); }
        .hero-card-wrap .story-card { max-width: 480px; margin-left: auto; }
        .js .hero-card-wrap .story-card { opacity: 0; transition: opacity 800ms var(--ease) 400ms; }
        .is-loaded .hero-card-wrap .story-card { opacity: 1; }
        /* Wide screens: the card hangs off the bottom right corner of the photograph. */
        @media (min-width: 1280px) {
            .hero-card-wrap { height: 0; margin-top: 0; }
            .hero-card-wrap .story-card { max-width: 440px; transform: translateY(calc(-100% + var(--spacing-900))); }
        }
        .story-card {
            position: relative; text-align: left; padding: var(--spacing-300);
            border-radius: var(--rounded-2xl); border: 1px solid var(--border);
            background: var(--surface-raised); color: var(--text); box-shadow: var(--card-shadow);
        }
        .story-card-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--spacing-75); }
        .chip {
            display: inline-flex; align-items: center; gap: var(--spacing-50);
            padding: var(--spacing-50) var(--spacing-75); border-radius: var(--rounded-md);
            background: var(--surface-muted); color: var(--accent);
        }
        .story-title { margin-top: var(--spacing-200); }
        .story-block { margin-top: var(--spacing-300); }
        .story-block-label { color: var(--text-subtle); margin-bottom: var(--spacing-75); }
        .takeaways { display: flex; flex-direction: column; gap: var(--spacing-100); }
        .takeaways li { display: flex; gap: var(--spacing-75); align-items: flex-start; }
        .takeaways .icon { color: var(--accent); }
        .story-foot { margin-top: var(--spacing-300); }

        /* ==========================================================================
           SECTIONS
           ========================================================================== */
        .section { padding-block: var(--spacing-800) var(--spacing-900); }
        @media (min-width: 768px) { .section { padding-block: var(--spacing-900); } }
        .section-head { max-width: 680px; margin-bottom: var(--spacing-600); }
        .section-head > * + * { margin-top: var(--spacing-200); }
        .after-hero { padding-top: var(--spacing-800); }
        .section-flush { padding-top: 0; }
        @media (min-width: 768px) { .section-flush { padding-top: 0; } }
        @media (min-width: 1280px) { .after-hero { padding-top: var(--spacing-900); } .after-hero .container { padding-top: var(--spacing-900); } }

        /* --- Why: a print of a crowded street against the promise --- */
        .why { display: grid; gap: var(--spacing-600); align-items: center; }
        @media (min-width: 1024px) { .why { grid-template-columns: 5fr 6fr; gap: var(--spacing-800); } }
        .why-figure .photo { aspect-ratio: 6 / 5; border-radius: var(--rounded-3xl); }
        .why-figure figcaption { margin-top: var(--spacing-100); }
        .why .section-head { margin-bottom: 0; }
        .compare { margin-top: var(--spacing-500); display: grid; gap: var(--spacing-400); }
        @media (min-width: 640px) { .compare { grid-template-columns: 1fr 1fr; gap: var(--spacing-300); } }
        .compare-col h3 { padding-bottom: var(--spacing-100); margin-bottom: var(--spacing-200); border-bottom: 1px solid var(--border); }
        .compare-col.is-strong h3 { border-bottom-color: var(--border-strong); }
        .compare-list { display: flex; flex-direction: column; gap: var(--spacing-200); }
        .compare-list li { display: flex; gap: var(--spacing-100); align-items: flex-start; }
        .compare-list .icon { margin-top: var(--spacing-25); }
        .compare-col:not(.is-strong) .icon { color: var(--text-subtle); }
        .compare-col.is-strong .icon { color: var(--accent); }

        /* --- Features: asymmetric bento with small pieces of real interface --- */
        .bento { display: grid; gap: var(--spacing-300); }
        @media (min-width: 768px) { .bento { grid-template-columns: repeat(12, 1fr); } .b-a { grid-column: span 7; } .b-b { grid-column: span 5; } .b-c { grid-column: span 5; } .b-d { grid-column: span 7; } .b-e { grid-column: span 12; } }
        .tile {
            display: flex; flex-direction: column; gap: var(--spacing-300);
            padding: var(--spacing-300); border-radius: var(--rounded-3xl);
            border: 1px solid var(--border); background: var(--surface);
        }
        @media (min-width: 768px) { .tile { padding: var(--spacing-400); } }
        .tile-text h3 + p { margin-top: var(--spacing-75); }
        .tile-text p { color: var(--text-muted); max-width: 520px; }
        /* Inner panels sit 32px inside a 24px radius tile, so they keep their own radius. */
        .ui { margin-top: auto; border-radius: var(--rounded-2xl); border: 1px solid var(--border); background: var(--surface-raised); padding: var(--spacing-200); }
        .spec-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--spacing-200); }
        .spec-row dt { display: block; }
        .spec-row dd { display: block; margin-top: var(--spacing-50); color: var(--text-muted); }
        .skel { display: block; height: 8px; border-radius: var(--rounded-full); background: var(--surface-muted); }
        .skel + .skel { margin-top: var(--spacing-75); }
        .skel-90 { width: 90%; } .skel-75 { width: 75%; } .skel-50 { width: 50%; }
        .ui .link-row { margin-top: var(--spacing-200); }
        .ui-label { display: flex; align-items: center; gap: var(--spacing-75); color: var(--accent); margin-bottom: var(--spacing-200); }
        .ui-chips { display: flex; flex-wrap: wrap; gap: var(--spacing-75); margin-top: var(--spacing-200); }
        .ui-chip { display: inline-flex; align-items: center; height: 28px; padding: 0 var(--spacing-100); border-radius: var(--rounded-md); border: 1px solid var(--border); background: var(--bg); }
        .ui-chip .skel { width: 40px; height: 6px; }
        .ui-chip .skel-chip-lg { width: 56px; } .ui-chip .skel-chip-sm { width: 32px; }
        .source-stack { display: flex; flex-direction: column; }
        .source-stack li { display: flex; justify-content: space-between; align-items: center; gap: var(--spacing-200); padding-block: var(--spacing-100); }
        .source-stack li + li { border-top: 1px solid var(--border); }
        .link-row { display: flex; align-items: center; justify-content: space-between; gap: var(--spacing-200); padding: var(--spacing-100) var(--spacing-200); border-radius: var(--rounded-lg); background: var(--bg); border: 1px solid var(--border); }
        .link-row span { display: inline-flex; align-items: center; gap: var(--spacing-75); }
        .b-e { gap: var(--spacing-400); }
        @media (min-width: 768px) { .b-e { flex-direction: row; align-items: center; justify-content: space-between; } .b-e .ui { margin-top: 0; } }
        .action-row { display: flex; flex-wrap: wrap; gap: var(--spacing-100); }
        .action { display: inline-flex; align-items: center; gap: var(--spacing-75); padding: var(--spacing-75) var(--spacing-100); border-radius: var(--rounded-lg); background: var(--bg); border: 1px solid var(--border); }
        .action .icon { color: var(--accent); }

        /* --- Tagline: full bleed mountains, words light up one at a time --- */
        .tagline-section { position: relative; isolation: isolate; color: #F2EFE6; display: flex; align-items: center; min-height: max(560px, 86svh); padding-block: var(--spacing-900); }
        .tagline-section .photo { position: absolute; inset: 0; z-index: -1; }
        .tagline-section .photo img { object-position: 50% 35%; filter: brightness(0.82); }
        .tagline-section .photo::before { background: rgba(19, 18, 9, 0.3); }
        .tagline { max-width: 760px; }
        .tagline .word { transition: opacity var(--duration-layout) var(--ease); }
        .js .tagline .word { opacity: 0.3; }
        .js .tagline .word.is-on { opacity: 1; }
        .tagline-credit { right: var(--spacing-300); bottom: var(--spacing-300); background: rgba(19, 18, 9, 0.5); color: #D9D4C6; }

        /* --- How it works: a numbered line, not three boxes --- */
        .steps { display: grid; gap: var(--spacing-500); }
        @media (min-width: 768px) { .steps { grid-template-columns: repeat(3, 1fr); gap: var(--spacing-400); } }
        .step-num { display: flex; align-items: center; gap: var(--spacing-200); color: var(--accent); }
        .step-num::after { content: ""; flex: 1; height: 1px; background: var(--border); }
        .step h3 { margin-top: var(--spacing-300); }
        .step p { margin-top: var(--spacing-75); color: var(--text-muted); }

        /* --- Readers --- */
        .quotes { columns: 1; column-gap: var(--spacing-300); }
        @media (min-width: 768px) { .quotes { columns: 2; } }
        @media (min-width: 1024px) { .quotes { columns: 3; } }
        .quote-card { break-inside: avoid; margin: 0 0 var(--spacing-300); display: flex; flex-direction: column; gap: var(--spacing-300); padding: var(--spacing-300); border-radius: var(--rounded-2xl); border: 1px solid var(--border); background: var(--surface); }
        .quote-person { display: flex; align-items: center; gap: var(--spacing-100); }
        .quote-name, .quote-detail { display: block; }
        .quote-detail { margin-top: var(--spacing-25); }
        .avatar { display: inline-flex; align-items: center; justify-content: center; flex: none; width: 40px; height: 40px; border-radius: var(--rounded-xl); background: var(--surface-muted); }
        .quotes-foot { margin-top: var(--spacing-200); }
        .invite { display: grid; gap: var(--spacing-300); align-items: center; padding: var(--spacing-400); border-radius: var(--rounded-3xl); border: 1px solid var(--border); background: var(--surface); }
        @media (min-width: 768px) { .invite { grid-template-columns: auto 1fr auto; padding: var(--spacing-500); } }
        .invite .icon-lg { color: var(--accent); }
        .invite p { margin-top: var(--spacing-75); color: var(--text-muted); max-width: 560px; }

        /* --- Sources: an editorial index --- */
        .sources { display: grid; gap: var(--spacing-600); }
        @media (min-width: 1024px) { .sources { grid-template-columns: 5fr 6fr; gap: var(--spacing-800); } }
        .sources .section-head { margin-bottom: 0; }
        .proof-note { display: flex; gap: var(--spacing-100); align-items: flex-start; margin-top: var(--spacing-300); color: var(--text-muted); }
        .proof-note .icon { color: var(--accent); }
        .index-list li { display: flex; align-items: baseline; justify-content: space-between; gap: var(--spacing-200); padding-block: var(--spacing-200); border-top: 1px solid var(--border); }
        .index-list li:last-child { border-bottom: 1px solid var(--border); }
        .index-list .is-muted { color: var(--text-subtle); }
        .stat { display: flex; align-items: baseline; gap: var(--spacing-200); margin-bottom: var(--spacing-400); }

        /* --- FAQ: every answer visible, side by side --- */
        .faq-grid { display: grid; gap: var(--spacing-500) var(--spacing-600); }
        @media (min-width: 768px) { .faq-grid { grid-template-columns: 1fr 1fr; } }
        .faq-grid dt { padding-top: var(--spacing-200); border-top: 1px solid var(--border); }
        .faq-grid dd { margin-top: var(--spacing-75); color: var(--text-muted); }

        /* --- Footer: the last print, with the final call to action on its sky --- */
        .site-footer { position: relative; isolation: isolate; color: var(--on-photo); padding-top: var(--spacing-900); }
        .site-footer > .photo { position: absolute; inset: 0; z-index: -1; }
        .site-footer > .photo img { object-position: 50% 0%; }
        .final-cta { max-width: 680px; }
        .final-cta p { margin-top: var(--spacing-200); color: var(--on-photo-muted); }
        .final-cta .hero-actions { margin-top: var(--spacing-400); }
        .final-cta .text-link { color: var(--on-photo); }
        .footer-panel {
            margin-top: var(--spacing-900); margin-bottom: var(--spacing-300);
            padding: var(--spacing-400) var(--spacing-300); border-radius: var(--rounded-3xl);
            border: 1px solid var(--border); background: var(--panel-bg); color: var(--text);
            -webkit-backdrop-filter: blur(16px); backdrop-filter: blur(16px);
        }
        @media (min-width: 768px) { .footer-panel { padding: var(--spacing-500); } }
        .footer-top { display: grid; gap: var(--spacing-400); }
        @media (min-width: 768px) { .footer-top { grid-template-columns: 2fr 1fr 1fr; } }
        .footer-brand { display: inline-flex; align-items: center; gap: var(--spacing-75); }
        .footer-brand .icon { color: var(--accent); }
        .footer-blurb { margin-top: var(--spacing-100); max-width: 320px; color: var(--text-muted); }
        .footer-col h2 { margin-bottom: var(--spacing-100); }
        .footer-col ul { display: flex; flex-direction: column; gap: var(--spacing-75); }
        .footer-col a { color: var(--text-muted); transition: color var(--duration-color) var(--ease); }
        .footer-col a:hover { color: var(--text); }
        .footer-bottom { margin-top: var(--spacing-400); padding-top: var(--spacing-200); border-top: 1px solid var(--border); display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--spacing-100); color: var(--text-subtle); }

        /* ==========================================================================
           SCROLL REVEAL (only when JS runs, so content never hides without it)
           ========================================================================== */
        .js [data-reveal] { opacity: 0; transform: translateY(var(--spacing-400)); transition: opacity 700ms var(--ease), transform 700ms var(--ease); }
        .js [data-reveal].is-visible { opacity: 1; transform: none; }

        @media (prefers-reduced-motion: reduce) {
            html { scroll-behavior: auto; }
            .js [data-reveal], .js .hero-copy > * { transform: none; }
            .hero-media img { transform: none; transition: none; }
            .menu-item, .mobile-menu.is-open .menu-item { transform: none; }
            .nav-toggle-line, .btn, .btn .icon, .text-link .icon { transition-property: opacity, background-color, color, border-color; }
            .btn:active { transform: none; }
            .live-dot { animation: none; }
            .js .tagline .word { opacity: 1; }
        }
    </style>
</head>
<body>
    {# Phosphor icons (MIT), inlined as a sprite so the page needs no icon font. #}
    <svg xmlns="http://www.w3.org/2000/svg" width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false">
    <symbol id="ph-arrow-right" viewBox="0 0 256 256"><path d="M221.66,133.66l-72,72a8,8,0,0,1-11.32-11.32L196.69,136H40a8,8,0,0,1,0-16H196.69L138.34,61.66a8,8,0,0,1,11.32-11.32l72,72A8,8,0,0,1,221.66,133.66Z"/></symbol>
    <symbol id="ph-arrow-up-right" viewBox="0 0 256 256"><path d="M200,64V168a8,8,0,0,1-16,0V83.31L69.66,197.66a8,8,0,0,1-11.32-11.32L172.69,72H88a8,8,0,0,1,0-16H192A8,8,0,0,1,200,64Z"/></symbol>
    <symbol id="ph-check-circle" viewBox="0 0 256 256"><path d="M173.66,98.34a8,8,0,0,1,0,11.32l-56,56a8,8,0,0,1-11.32,0l-24-24a8,8,0,0,1,11.32-11.32L112,148.69l50.34-50.35A8,8,0,0,1,173.66,98.34ZM232,128A104,104,0,1,1,128,24,104.11,104.11,0,0,1,232,128Zm-16,0a88,88,0,1,0-88,88A88.1,88.1,0,0,0,216,128Z"/></symbol>
    <symbol id="ph-x-circle" viewBox="0 0 256 256"><path d="M165.66,101.66,139.31,128l26.35,26.34a8,8,0,0,1-11.32,11.32L128,139.31l-26.34,26.35a8,8,0,0,1-11.32-11.32L116.69,128,90.34,101.66a8,8,0,0,1,11.32-11.32L128,116.69l26.34-26.35a8,8,0,0,1,11.32,11.32ZM232,128A104,104,0,1,1,128,24,104.11,104.11,0,0,1,232,128Zm-16,0a88,88,0,1,0-88,88A88.1,88.1,0,0,0,216,128Z"/></symbol>
    <symbol id="ph-timer" viewBox="0 0 256 256"><path d="M128,40a96,96,0,1,0,96,96A96.11,96.11,0,0,0,128,40Zm0,176a80,80,0,1,1,80-80A80.09,80.09,0,0,1,128,216ZM173.66,90.34a8,8,0,0,1,0,11.32l-40,40a8,8,0,0,1-11.32-11.32l40-40A8,8,0,0,1,173.66,90.34ZM96,16a8,8,0,0,1,8-8h48a8,8,0,0,1,0,16H104A8,8,0,0,1,96,16Z"/></symbol>
    <symbol id="ph-sparkle" viewBox="0 0 256 256"><path d="M197.58,129.06,146,110l-19-51.62a15.92,15.92,0,0,0-29.88,0L78,110l-51.62,19a15.92,15.92,0,0,0,0,29.88L78,178l19,51.62a15.92,15.92,0,0,0,29.88,0L146,178l51.62-19a15.92,15.92,0,0,0,0-29.88ZM137,164.22a8,8,0,0,0-4.74,4.74L112,223.85,91.78,169A8,8,0,0,0,87,164.22L32.15,144,87,123.78A8,8,0,0,0,91.78,119L112,64.15,132.22,119a8,8,0,0,0,4.74,4.74L191.85,144ZM144,40a8,8,0,0,1,8-8h16V16a8,8,0,0,1,16,0V32h16a8,8,0,0,1,0,16H184V64a8,8,0,0,1-16,0V48H152A8,8,0,0,1,144,40ZM248,88a8,8,0,0,1-8,8h-8v8a8,8,0,0,1-16,0V96h-8a8,8,0,0,1,0-16h8V72a8,8,0,0,1,16,0v8h8A8,8,0,0,1,248,88Z"/></symbol>
    <symbol id="ph-stack" viewBox="0 0 256 256"><path d="M230.91,172A8,8,0,0,1,228,182.91l-96,56a8,8,0,0,1-8.06,0l-96-56A8,8,0,0,1,36,169.09l92,53.65,92-53.65A8,8,0,0,1,230.91,172ZM220,121.09l-92,53.65L36,121.09A8,8,0,0,0,28,134.91l96,56a8,8,0,0,0,8.06,0l96-56A8,8,0,1,0,220,121.09ZM24,80a8,8,0,0,1,4-6.91l96-56a8,8,0,0,1,8.06,0l96,56a8,8,0,0,1,0,13.82l-96,56a8,8,0,0,1-8.06,0l-96-56A8,8,0,0,1,24,80Zm23.88,0L128,126.74,208.12,80,128,33.26Z"/></symbol>
    <symbol id="ph-link-simple" viewBox="0 0 256 256"><path d="M165.66,90.34a8,8,0,0,1,0,11.32l-64,64a8,8,0,0,1-11.32-11.32l64-64A8,8,0,0,1,165.66,90.34ZM215.6,40.4a56,56,0,0,0-79.2,0L106.34,70.45a8,8,0,0,0,11.32,11.32l30.06-30a40,40,0,0,1,56.57,56.56l-30.07,30.06a8,8,0,0,0,11.31,11.32L215.6,119.6a56,56,0,0,0,0-79.2ZM138.34,174.22l-30.06,30.06a40,40,0,1,1-56.56-56.57l30.05-30.05a8,8,0,0,0-11.32-11.32L40.4,136.4a56,56,0,0,0,79.2,79.2l30.06-30.07a8,8,0,0,0-11.32-11.31Z"/></symbol>
    <symbol id="ph-bookmark-simple" viewBox="0 0 256 256"><path d="M184,32H72A16,16,0,0,0,56,48V224a8,8,0,0,0,12.24,6.78L128,193.43l59.77,37.35A8,8,0,0,0,200,224V48A16,16,0,0,0,184,32Zm0,177.57-51.77-32.35a8,8,0,0,0-8.48,0L72,209.57V48H184Z"/></symbol>
    <symbol id="ph-newspaper" viewBox="0 0 256 256"><path d="M88,112a8,8,0,0,1,8-8h80a8,8,0,0,1,0,16H96A8,8,0,0,1,88,112Zm8,40h80a8,8,0,0,0,0-16H96a8,8,0,0,0,0,16ZM232,64V184a24,24,0,0,1-24,24H32A24,24,0,0,1,8,184.11V88a8,8,0,0,1,16,0v96a8,8,0,0,0,16,0V64A16,16,0,0,1,56,48H216A16,16,0,0,1,232,64Zm-16,0H56V184a23.84,23.84,0,0,1-1.37,8H208a8,8,0,0,0,8-8Z"/></symbol>
    <symbol id="ph-scissors" viewBox="0 0 256 256"><path d="M157.73,113.13A8,8,0,0,1,159.82,102L227.48,55.7a8,8,0,0,1,9,13.21l-67.67,46.3a7.92,7.92,0,0,1-4.51,1.4A8,8,0,0,1,157.73,113.13Zm80.87,85.09a8,8,0,0,1-11.12,2.08L136,137.7,93.49,166.78a36,36,0,1,1-9-13.19L121.83,128,84.44,102.41a35.86,35.86,0,1,1,9-13.19l143,97.87A8,8,0,0,1,238.6,198.22ZM80,180a20,20,0,1,0-5.86,14.14A19.85,19.85,0,0,0,80,180ZM74.14,90.13a20,20,0,1,0-28.28,0A19.85,19.85,0,0,0,74.14,90.13Z"/></symbol>
    <symbol id="ph-book-open" viewBox="0 0 256 256"><path d="M232,48H160a40,40,0,0,0-32,16A40,40,0,0,0,96,48H24a8,8,0,0,0-8,8V200a8,8,0,0,0,8,8H96a24,24,0,0,1,24,24,8,8,0,0,0,16,0,24,24,0,0,1,24-24h72a8,8,0,0,0,8-8V56A8,8,0,0,0,232,48ZM96,192H32V64H96a24,24,0,0,1,24,24V200A39.81,39.81,0,0,0,96,192Zm128,0H160a39.81,39.81,0,0,0-24,8V88a24,24,0,0,1,24-24h64Z"/></symbol>
    <symbol id="ph-plus" viewBox="0 0 256 256"><path d="M224,128a8,8,0,0,1-8,8H136v80a8,8,0,0,1-16,0V136H40a8,8,0,0,1,0-16h80V40a8,8,0,0,1,16,0v80h80A8,8,0,0,1,224,128Z"/></symbol>
    <symbol id="ph-shield-check" viewBox="0 0 256 256"><path d="M208,40H48A16,16,0,0,0,32,56v56c0,52.72,25.52,84.67,46.93,102.19,23.06,18.86,46,25.26,47,25.53a8,8,0,0,0,4.2,0c1-.27,23.91-6.67,47-25.53C198.48,196.67,224,164.72,224,112V56A16,16,0,0,0,208,40Zm0,72c0,37.07-13.66,67.16-40.6,89.42A129.3,129.3,0,0,1,128,223.62a128.25,128.25,0,0,1-38.92-21.81C61.82,179.51,48,149.3,48,112l0-56,160,0ZM82.34,141.66a8,8,0,0,1,11.32-11.32L112,148.69l50.34-50.35a8,8,0,0,1,11.32,11.32l-56,56a8,8,0,0,1-11.32,0Z"/></symbol>
    <symbol id="ph-rss-simple" viewBox="0 0 256 256"><path d="M224,192a8,8,0,0,1-16,0c0-79.4-64.6-144-144-144a8,8,0,0,1,0-16C152.22,32,224,103.78,224,192ZM64,104a8,8,0,0,0,0,16,72.08,72.08,0,0,1,72,72,8,8,0,0,0,16,0A88.1,88.1,0,0,0,64,104Zm4,72a12,12,0,1,0,12,12A12,12,0,0,0,68,176Z"/></symbol>
    <symbol id="ph-quotes-fill" viewBox="0 0 256 256"><path d="M116,72v88a48.05,48.05,0,0,1-48,48,8,8,0,0,1,0-16,32,32,0,0,0,32-32v-8H40a16,16,0,0,1-16-16V72A16,16,0,0,1,40,56h60A16,16,0,0,1,116,72ZM216,56H156a16,16,0,0,0-16,16v64a16,16,0,0,0,16,16h60v8a32,32,0,0,1-32,32,8,8,0,0,0,0,16,48.05,48.05,0,0,0,48-48V72A16,16,0,0,0,216,56Z"/></symbol>
    <symbol id="ph-chat-circle-text" viewBox="0 0 256 256"><path d="M168,112a8,8,0,0,1-8,8H96a8,8,0,0,1,0-16h64A8,8,0,0,1,168,112Zm-8,24H96a8,8,0,0,0,0,16h64a8,8,0,0,0,0-16Zm72-8A104,104,0,0,1,79.12,219.82L45.07,231.17a16,16,0,0,1-20.24-20.24l11.35-34.05A104,104,0,1,1,232,128Zm-16,0A88,88,0,1,0,51.81,172.06a8,8,0,0,1,.66,6.54L40,216,77.4,203.53a7.85,7.85,0,0,1,2.53-.42,8,8,0,0,1,4,1.08A88,88,0,0,0,216,128Z"/></symbol>
    <symbol id="ph-note-pencil" viewBox="0 0 256 256"><path d="M229.66,58.34l-32-32a8,8,0,0,0-11.32,0l-96,96A8,8,0,0,0,88,128v32a8,8,0,0,0,8,8h32a8,8,0,0,0,5.66-2.34l96-96A8,8,0,0,0,229.66,58.34ZM124.69,152H104V131.31l64-64L188.69,88ZM200,76.69,179.31,56,192,43.31,212.69,64ZM224,128v80a16,16,0,0,1-16,16H48a16,16,0,0,1-16-16V48A16,16,0,0,1,48,32h80a8,8,0,0,1,0,16H48V208H208V128a8,8,0,0,1,16,0Z"/></symbol>
    <symbol id="ph-image" viewBox="0 0 256 256"><path d="M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40Zm0,16V158.75l-26.07-26.06a16,16,0,0,0-22.63,0l-20,20-44-44a16,16,0,0,0-22.62,0L40,149.37V56ZM40,172l52-52,80,80H40Zm176,28H194.63l-36-36,20-20L216,181.38V200ZM144,100a12,12,0,1,1,12,12A12,12,0,0,1,144,100Z"/></symbol>
</svg>

    <a class="skip-link button-sm" href="#main">Skip to content</a>

    <header>
        <nav class="site-nav" aria-label="Main">
            <a class="nav-brand title-3" href="{{ url_for('landing') }}" aria-label="BrieflyAI, back to top">
                <svg class="logo-mark" viewBox="0 0 32 32" width="24" height="24" aria-hidden="true" focusable="false"><rect class="logo-tile" x="1" y="1" width="30" height="30" rx="9"/><rect class="logo-line" x="8" y="8.5" width="16" height="3" rx="1.5"/><rect class="logo-line" x="8" y="14.5" width="11" height="3" rx="1.5"/><rect class="logo-gist" x="8" y="20.5" width="6" height="3" rx="1.5"/></svg>
                <span>BrieflyAI</span>
            </a>
            <ul class="nav-links title-4">
                <li><a href="#why" data-spy>Why BrieflyAI</a></li>
                <li><a href="#features" data-spy>Features</a></li>
                <li><a href="#how" data-spy>How it works</a></li>
                <li><a href="#faq" data-spy>FAQ</a></li>
            </ul>
            <a class="btn btn-primary button-sm nav-cta" href="{{ url_for('index') }}">Read today's briefing</a>
            <button class="nav-toggle" id="navToggle" type="button" aria-expanded="false" aria-controls="mobileMenu" aria-label="Open menu">
                <span class="nav-toggle-line" aria-hidden="true"></span>
                <span class="nav-toggle-line" aria-hidden="true"></span>
            </button>
        </nav>
    </header>

    <div class="mobile-menu" id="mobileMenu" hidden>
        <nav aria-label="Mobile">
            <ul class="mobile-menu-list heading-5">
                <li class="menu-item"><a href="#why" data-spy>Why BrieflyAI</a></li>
                <li class="menu-item"><a href="#features" data-spy>Features</a></li>
                <li class="menu-item"><a href="#how" data-spy>How it works</a></li>
                <li class="menu-item"><a href="#faq" data-spy>FAQ</a></li>
            </ul>
        </nav>
        <div class="menu-item">
            <a class="btn btn-primary button-md" href="{{ url_for('index') }}">Read today's briefing<svg class="icon icon-arrow" aria-hidden="true"><use href="#ph-arrow-right"></use></svg></a>
        </div>
    </div>

    <main id="main">
        {# ================= HERO ================= #}
        <section class="hero" id="top" aria-labelledby="hero-title">
            <div class="hero-media photo">
                <picture>
                    <source media="(max-width: 767px)" srcset="{{ url_for('static', filename='landing/hero-nirmahal-ghat-mobile-900.webp') }}" type="image/webp">
                    <img src="{{ url_for('static', filename='landing/hero-nirmahal-ghat-2400.webp') }}"
                         srcset="{{ url_for('static', filename='landing/hero-nirmahal-ghat-1200.webp') }} 1200w, {{ url_for('static', filename='landing/hero-nirmahal-ghat-2400.webp') }} 2400w"
                         sizes="100vw" width="2400" height="1258" fetchpriority="high" decoding="async"
                         alt="Boats moored on the Hooghly river at Nirmahal Ghat in Calcutta around 1890, with the old pontoon bridge on the horizon">
                </picture>
            </div>
            <div class="container hero-content">
                <div class="hero-copy">
                    <p class="hero-badge caption-4"><span class="live-dot" aria-hidden="true"></span>Free AI news briefings for India</p>
                    <h1 class="hero-title heading-3 md:heading-2 lg:heading-1" id="hero-title"><span class="hero-title-text">India's news,<br> one minute per story</span></h1>
                    <p class="hero-sub body-3 md:body-7">Know what happened today without opening ten tabs. Every story gets a neutral summary of under 60 words and three or four key takeaways, with the original article one tap away.</p>
                    <div class="hero-actions">
                        <a class="btn btn-primary button-md" href="{{ url_for('index') }}">Read today's briefing<svg class="icon icon-arrow" aria-hidden="true"><use href="#ph-arrow-right"></use></svg></a>
                        <p class="hero-note frost caption-2">Free to read. No account or card needed.</p>
                    </div>
                    <div class="hero-proof frost">
                        <p class="caption-4 hero-proof-label">Main feed from</p>
                        <ul class="hero-sources title-4" aria-label="News sources in the main feed">
                            {% for name in sources %}<li>{{ name }}</li>{% endfor %}
                        </ul>
                        {% if sources|length > 2 %}<p class="hero-proof-short caption-1">From {{ sources[0] }}, {{ sources[1] }} and {{ sources|length - 2 }} more newsrooms</p>{% endif %}
                    </div>
                </div>
            </div>
            <p class="credit-on-photo hero-credit caption-5"><svg class="icon icon-sm" aria-hidden="true"><use href="#ph-image"></use></svg>Nirmahal Ghat, Calcutta, c. 1890</p>

        </section>

        <div class="hero-card-wrap">
            <div class="container">
                {% if sample_story %}
                <article class="story-card" aria-label="A real BrieflyAI summary">
                    <div class="story-card-head">
                        <span class="chip caption-3"><span class="live-dot" aria-hidden="true"></span>Real summary from BrieflyAI</span>
                        <span class="caption-5 subtle">{{ sample_story.source }}{% if sample_story.published %} · {{ sample_story.published }}{% endif %}</span>
                    </div>
                    <p class="story-title title-1">{{ sample_story.title }}</p>
                    <div class="story-block">
                        <p class="story-block-label caption-3">AI summary</p>
                        <p class="body-6">{{ sample_story.summary }}</p>
                    </div>
                    <div class="story-block">
                        <p class="story-block-label caption-3">Key takeaways</p>
                        <ul class="takeaways">
                            {% for point in sample_story.takeaways %}
                            <li><svg class="icon" aria-hidden="true"><use href="#ph-check-circle"></use></svg><span class="body-6">{{ point }}</span></li>
                            {% endfor %}
                        </ul>
                    </div>
                    <p class="story-foot"><a class="text-link button-sm" href="{{ sample_story.url }}">Open this story<svg class="icon icon-sm" aria-hidden="true"><use href="#ph-arrow-up-right"></use></svg></a></p>
                </article>
                {% else %}
                <article class="story-card" aria-label="What every BrieflyAI summary contains">
                    <div class="story-card-head">
                        <span class="chip caption-3">How every summary looks</span>
                        <span class="caption-5 subtle">Publisher · Time in IST</span>
                    </div>
                    <p class="story-title title-1">The publisher's headline, exactly as they wrote it</p>
                    <div class="story-block">
                        <p class="story-block-label caption-3">AI summary</p>
                        <p class="body-6">A neutral summary of under 60 words. It opens with the single most important fact, so you know what changed before you decide to read further.</p>
                    </div>
                    <div class="story-block">
                        <p class="story-block-label caption-3">Key takeaways</p>
                        <ul class="takeaways">
                            <li><svg class="icon" aria-hidden="true"><use href="#ph-check-circle"></use></svg><span class="body-6">Three or four takeaways for every story.</span></li>
                            <li><svg class="icon" aria-hidden="true"><use href="#ph-check-circle"></use></svg><span class="body-6">Each one is a single sentence.</span></li>
                            <li><svg class="icon" aria-hidden="true"><use href="#ph-check-circle"></use></svg><span class="body-6">Each stays under 20 words, so you can scan them.</span></li>
                        </ul>
                    </div>
                    <p class="story-foot"><a class="text-link button-sm" href="{{ url_for('index') }}">See today's stories<svg class="icon icon-sm" aria-hidden="true"><use href="#ph-arrow-up-right"></use></svg></a></p>
                </article>
                {% endif %}
            </div>
        </div>

        {# ================= WHY ================= #}
        <section class="section after-hero" id="why" aria-labelledby="why-title">
            <div class="container why">
                <figure class="why-figure" data-reveal>
                    <div class="photo">
                        <img src="{{ url_for('static', filename='landing/chitpore-road-1400.webp') }}"
                             srcset="{{ url_for('static', filename='landing/chitpore-road-800.webp') }} 800w, {{ url_for('static', filename='landing/chitpore-road-1400.webp') }} 1400w"
                             sizes="(min-width: 1024px) 520px, 100vw" width="1400" height="1151" loading="lazy" decoding="async"
                             alt="A crowded corner of Chitpore Road in Calcutta around 1890, with carts, shopfronts and people crossing the street">
                    </div>
                    <figcaption class="credit caption-5">Chitpore Road, Calcutta, c. 1890. Everything at once.</figcaption>
                </figure>
                <div data-reveal>
                    <div class="section-head">
                        <p class="eyebrow caption-3">Why BrieflyAI</p>
                        <h2 class="heading-4 md:heading-2" id="why-title">The news should not take your whole morning</h2>
                        <p class="body-3 md:body-7 muted">Most of us skim the same story on three sites, click into a long article, and still have to hunt for the one fact that changed. BrieflyAI puts that fact first.</p>
                    </div>
                    <div class="compare">
                        <div class="compare-col">
                            <h3 class="title-3 muted">The usual routine</h3>
                            <ul class="compare-list body-6 muted">
                                <li><svg class="icon" aria-hidden="true"><use href="#ph-x-circle"></use></svg>Several news sites to cover one day</li>
                                <li><svg class="icon" aria-hidden="true"><use href="#ph-x-circle"></use></svg>The same story read more than once</li>
                                <li><svg class="icon" aria-hidden="true"><use href="#ph-x-circle"></use></svg>Background to scroll past before what changed</li>
                            </ul>
                        </div>
                        <div class="compare-col is-strong">
                            <h3 class="title-3">With BrieflyAI</h3>
                            <ul class="compare-list body-6">
                                <li><svg class="icon" aria-hidden="true"><use href="#ph-check-circle"></use></svg>One feed built from {% if source_count_word %}{{ source_count_word }} {% endif %}major newsrooms</li>
                                <li><svg class="icon" aria-hidden="true"><use href="#ph-check-circle"></use></svg>The main fact in the first sentence</li>
                                <li><svg class="icon" aria-hidden="true"><use href="#ph-check-circle"></use></svg>The original article one tap away</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        {# ================= FEATURES ================= #}
        <section class="section" id="features" aria-labelledby="features-title">
            <div class="container">
                <div class="section-head" data-reveal>
                    <p class="eyebrow caption-3">What you get</p>
                    <h2 class="heading-4 md:heading-2" id="features-title">Built for a quick read, not a long scroll</h2>
                </div>
                <div class="bento">
                    <div class="tile b-a" data-reveal>
                        <div class="tile-text">
                            <h3 class="title-1">The gist in under a minute</h3>
                            <p class="body-3">Every story is cut to a summary that leads with the most important fact, followed by three or four one sentence takeaways.</p>
                        </div>
                        <dl class="ui spec-row">
                            <div><dt class="heading-4">60</dt><dd class="caption-2">word limit for each summary</dd></div>
                            <div><dt class="heading-4">3 to 4</dt><dd class="caption-2">takeaways for every story</dd></div>
                            <div><dt class="heading-4">1 tap</dt><dd class="caption-2">to the original article</dd></div>
                        </dl>
                    </div>
                    <div class="tile b-b" data-reveal>
                        <div class="tile-text">
                            <h3 class="title-1">The big picture first</h3>
                            <p class="body-3">A two sentence briefing at the top of the feed connects the day's main themes, with topic tags you can tap to search.</p>
                        </div>
                        <div class="ui" aria-hidden="true">
                            <p class="ui-label caption-3"><svg class="icon icon-sm"><use href="#ph-sparkle"></use></svg>Today's briefing</p>
                            <span class="skel skel-90"></span>
                            <span class="skel skel-75"></span>
                            <span class="skel skel-50"></span>
                            <div class="ui-chips"><span class="ui-chip"><span class="skel"></span></span><span class="ui-chip"><span class="skel skel-chip-lg"></span></span><span class="ui-chip"><span class="skel skel-chip-sm"></span></span></div>
                        </div>
                    </div>
                    <div class="tile b-c" data-reveal>
                        <div class="tile-text">
                            <h3 class="title-1">Major newsrooms, one feed</h3>
                            <p class="body-3">Stories land in one place, alongside the week's popular stories and yesterday's headlines.</p>
                        </div>
                        <ul class="ui source-stack body-5">
                            {% for name in sources %}<li><span>{{ name }}</span><svg class="icon icon-sm accent" aria-hidden="true"><use href="#ph-check-circle"></use></svg></li>{% endfor %}
                        </ul>
                    </div>
                    <div class="tile b-d" data-reveal>
                        <div class="tile-text">
                            <h3 class="title-1">Check anything at the source</h3>
                            <p class="body-3">Each summary links to the publisher's original article, so you can read the full reporting whenever a story matters to you.</p>
                        </div>
                        <div class="ui" aria-hidden="true">
                            <span class="skel skel-75"></span>
                            <span class="skel skel-90"></span>
                            <div class="link-row body-5">
                                <span><svg class="icon icon-sm accent"><use href="#ph-newspaper"></use></svg>Read the original{% if sources %} at {{ sources[0] }}{% endif %}</span>
                                <svg class="icon icon-sm"><use href="#ph-arrow-up-right"></use></svg>
                            </div>
                        </div>
                    </div>
                    <div class="tile b-e" data-reveal>
                        <div class="tile-text">
                            <h3 class="title-1">Save stories and join in</h3>
                            <p class="body-3">A free account lets you bookmark stories, react and reply in the comments, and post your own articles to the Community Hub.</p>
                        </div>
                        <div class="ui action-row body-5" aria-hidden="true">
                            <span class="action"><svg class="icon icon-sm"><use href="#ph-bookmark-simple"></use></svg>Bookmark</span>
                            <span class="action"><svg class="icon icon-sm"><use href="#ph-chat-circle-text"></use></svg>Reply</span>
                            <span class="action"><svg class="icon icon-sm"><use href="#ph-note-pencil"></use></svg>Post to Community Hub</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        {# ================= TAGLINE ================= #}
        <section class="tagline-section" aria-label="What BrieflyAI is for">
            <div class="photo">
                <img src="{{ url_for('static', filename='landing/darjeeling-2400.webp') }}"
                     srcset="{{ url_for('static', filename='landing/darjeeling-1200.webp') }} 1200w, {{ url_for('static', filename='landing/darjeeling-2400.webp') }} 2400w"
                     sizes="100vw" width="2400" height="1584" loading="lazy" decoding="async"
                     alt="The Kanchenjunga range seen across layered hills from Darjeeling around 1890">
            </div>
            <div class="container">
                <p class="tagline heading-3 lg:heading-2" data-tagline>{% for word in tagline.split() %}<span class="word">{{ word }}</span>{% if not loop.last %} {% endif %}{% endfor %}</p>
            </div>
            <p class="credit-on-photo tagline-credit caption-5"><svg class="icon icon-sm" aria-hidden="true"><use href="#ph-image"></use></svg>Kanchenjunga from Darjeeling, c. 1890</p>
        </section>

        {# ================= HOW IT WORKS ================= #}
        <section class="section" id="how" aria-labelledby="how-title">
            <div class="container">
                <div class="section-head" data-reveal>
                    <p class="eyebrow caption-3">How it works</p>
                    <h2 class="heading-4 md:heading-2" id="how-title">From newsroom to takeaways in three steps</h2>
                </div>
                <ol class="steps">
                    <li class="step" data-reveal>
                        <p class="step-num heading-4">01</p>
                        <h3 class="title-1">We collect the day's stories</h3>
                        <p class="body-3">New stories arrive through the day from the public news feeds of major Indian newsrooms, gathered into one feed.</p>
                    </li>
                    <li class="step" data-reveal>
                        <p class="step-num heading-4">02</p>
                        <h3 class="title-1">AI condenses each one</h3>
                        <p class="body-3">When a story is opened, a language model writes a neutral summary and the key takeaways. It is saved, so the next reader gets it instantly.</p>
                    </li>
                    <li class="step" data-reveal>
                        <p class="step-num heading-4">03</p>
                        <h3 class="title-1">You read, then go deeper</h3>
                        <p class="body-3">Skim the takeaways, open the original when you want the full report, or bookmark the story for later.</p>
                    </li>
                </ol>
            </div>
        </section>

        {# ================= READERS ================= #}
        <section class="section section-flush" id="readers" aria-labelledby="readers-title">
            <div class="container">
                {% if testimonials %}
                <div class="section-head" data-reveal>
                    <p class="eyebrow caption-3">Readers</p>
                    <h2 class="heading-4 md:heading-2" id="readers-title">What readers say</h2>
                </div>
                <div class="quotes">
                    {% for t in testimonials %}
                    <figure class="quote-card" data-reveal>
                        <svg class="icon icon-lg accent" aria-hidden="true"><use href="#ph-quotes-fill"></use></svg>
                        <blockquote class="body-3 md:body-7">{{ t.quote }}</blockquote>
                        <figcaption class="quote-person">
                            <span class="avatar title-4" aria-hidden="true">{{ t.initials }}</span>
                            <span><span class="quote-name title-4">{{ t.name }}</span>{% if t.detail %}<span class="quote-detail caption-5 subtle">{{ t.detail }}</span>{% endif %}</span>
                        </figcaption>
                    </figure>
                    {% endfor %}
                </div>
                <p class="quotes-foot body-6 muted" data-reveal>Read BrieflyAI often? <a class="text-link" href="{{ url_for('share_testimonial') }}">Share your experience</a></p>
                {% else %}
                <div class="invite" data-reveal>
                    <svg class="icon icon-lg" aria-hidden="true"><use href="#ph-quotes-fill"></use></svg>
                    <div>
                        <h2 class="title-1" id="readers-title">Reader notes will appear here</h2>
                        <p class="body-3">We only publish quotes from real readers who agree to be featured. Read BrieflyAI often? Tell other readers what it does for you.</p>
                    </div>
                    <a class="btn btn-ghost button-md" href="{{ url_for('share_testimonial') }}">Share your experience</a>
                </div>
                {% endif %}
            </div>
        </section>

        {# ================= SOURCES ================= #}
        <section class="section section-flush" id="sources" aria-labelledby="sources-title">
            <div class="container sources">
                <div class="section-head" data-reveal>
                    <p class="eyebrow caption-3">Sources</p>
                    <h2 class="heading-4 md:heading-2" id="sources-title">Built on reporting from newsrooms you know</h2>
                    <p class="body-3 md:body-7 muted">BrieflyAI never writes the news. Every summary is generated from a published article and links straight back to it.</p>
                    <p class="proof-note body-6"><svg class="icon" aria-hidden="true"><use href="#ph-shield-check"></use></svg><span>AI can miss nuance. When a story matters to you, the full original is always one tap away.</span></p>
                </div>
                <div data-reveal>
                    {% if weekly_summaries %}
                    <p class="stat"><span class="heading-2">{{ "{:,}".format(weekly_summaries) }}</span><span class="caption-2 muted">stories summarized on BrieflyAI in the last 7 days</span></p>
                    {% endif %}
                    <ul class="index-list">
                        {% for name in sources %}<li><span class="heading-6">{{ name }}</span><span class="caption-5 subtle">Main feed</span></li>{% endfor %}
                        <li><span class="heading-6 is-muted">Other outlets</span><span class="caption-5 subtle">Popular and yesterday's headlines</span></li>
                    </ul>
                </div>
            </div>
        </section>

        {# ================= FAQ ================= #}
        <section class="section section-flush" id="faq" aria-labelledby="faq-title">
            <div class="container">
                <div class="section-head" data-reveal>
                    <p class="eyebrow caption-3">FAQ</p>
                    <h2 class="heading-4 md:heading-2" id="faq-title">Questions, answered</h2>
                    <p class="body-3 muted">Something else on your mind? <a class="text-link" href="{{ url_for('contact') }}">Get in touch</a>.</p>
                </div>
                <dl class="faq-grid">
                    {% for item in faqs %}
                    <div data-reveal>
                        <dt class="title-2">{{ item.q }}</dt>
                        <dd class="body-3">{{ item.a }}</dd>
                    </div>
                    {% endfor %}
                </dl>
            </div>
        </section>
    </main>

    {# ================= FINAL CTA + FOOTER ================= #}
    <footer class="site-footer" aria-labelledby="start-title">
        <div class="photo">
            <img src="{{ url_for('static', filename='landing/bombay-harbour-2400.webp') }}"
                 srcset="{{ url_for('static', filename='landing/bombay-harbour-1200.webp') }} 1200w, {{ url_for('static', filename='landing/bombay-harbour-2400.webp') }} 2400w"
                 sizes="100vw" width="2400" height="1241" loading="lazy" decoding="async"
                 alt="Bombay harbour around 1890, with steamers at anchor and the pier pavilion">
        </div>
        <div class="container">
            <div class="final-cta" data-reveal>
                <h2 class="heading-4 md:heading-2" id="start-title">Start with today's briefing</h2>
                <p class="body-3 md:body-7">Free to read, with no account and no card. Create a free account later if you want to save stories or join the discussion.</p>
                <div class="hero-actions">
                    <a class="btn btn-primary button-md" href="{{ url_for('index') }}">Read today's briefing<svg class="icon icon-arrow" aria-hidden="true"><use href="#ph-arrow-right"></use></svg></a>
                    {% if not session.user_id %}<a class="text-link caption-2" href="{{ url_for('register') }}">Create a free account</a>{% endif %}
                </div>
            </div>

            <div class="footer-panel">
                <div class="footer-top">
                    <div>
                        <a class="footer-brand title-3" href="{{ url_for('landing') }}"><svg class="logo-mark" viewBox="0 0 32 32" width="24" height="24" aria-hidden="true" focusable="false"><rect class="logo-tile" x="1" y="1" width="30" height="30" rx="9"/><rect class="logo-line" x="8" y="8.5" width="16" height="3" rx="1.5"/><rect class="logo-line" x="8" y="14.5" width="11" height="3" rx="1.5"/><rect class="logo-gist" x="8" y="20.5" width="6" height="3" rx="1.5"/></svg><span>BrieflyAI</span></a>
                        <p class="footer-blurb body-6">Short AI summaries and key takeaways for India's top stories.</p>
                    </div>
                    <nav class="footer-col" aria-label="Read">
                        <h2 class="title-4">Read</h2>
                        <ul class="body-6">
                            <li><a href="{{ url_for('index') }}">Today's briefing</a></li>
                            <li><a href="{{ url_for('index', category_name='Popular Stories', page=1) }}">Popular stories</a></li>
                            <li><a href="{{ url_for('index', category_name='Community Hub', page=1) }}">Community Hub</a></li>
                            <li><a href="{{ url_for('rss_feed') }}">RSS feed</a></li>
                        </ul>
                    </nav>
                    <nav class="footer-col" aria-label="Company">
                        <h2 class="title-4">BrieflyAI</h2>
                        <ul class="body-6">
                            <li><a href="{{ url_for('about') }}">About</a></li>
                            <li><a href="{{ url_for('contact') }}">Contact</a></li>
                            <li><a href="{{ url_for('privacy') }}">Privacy policy</a></li>
                            <li><a href="{{ url_for('terms') }}">Terms of service</a></li>
                        </ul>
                    </nav>
                </div>
                <div class="footer-bottom caption-5">
                    <span>&copy; {{ current_year }} BrieflyAI. Made in India. Summaries are AI generated; always check the original.</span>
                    <span>Photochrom prints of India, c. 1890, Library of Congress. Footer: Bombay harbour.</span>
                </div>
            </div>
        </div>
    </footer>

    {% if ga_measurement_id %}
    <script>
    /* Analytics loads once the page is idle, the same way as on every other page. */
    window.addEventListener('load', function () {
        var go = function () {
            window.dataLayer = window.dataLayer || [];
            window.gtag = function () { window.dataLayer.push(arguments); };
            window.gtag('js', new Date());
            window.gtag('config', '{{ ga_measurement_id }}');
            var s = document.createElement('script');
            s.async = true; s.src = 'https://www.googletagmanager.com/gtag/js?id={{ ga_measurement_id }}';
            document.head.appendChild(s);
        };
        if ('requestIdleCallback' in window) { requestIdleCallback(go, { timeout: 3000 }); } else { setTimeout(go, 1800); }
    });
    </script>
    {% endif %}
    <script>
    (function () {
        var root = document.documentElement;
        root.classList.add('js');
        var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        var heroImg = document.querySelector('.hero-media img');
        var markLoaded = function () { requestAnimationFrame(function () { root.classList.add('is-loaded'); }); };
        if (!heroImg || heroImg.complete) { markLoaded(); } else { heroImg.addEventListener('load', markLoaded); heroImg.addEventListener('error', markLoaded); setTimeout(markLoaded, 1200); }

        /* --- Mobile menu ------------------------------------------------------ */
        var toggle = document.getElementById('navToggle');
        var menu = document.getElementById('mobileMenu');
        var page = [document.getElementById('main'), document.querySelector('.site-footer')];
        var items = menu ? Array.prototype.slice.call(menu.querySelectorAll('.menu-item')) : [];
        var hideTimer = null;
        function setInert(on) { page.forEach(function (el) { if (el) { if (on) { el.setAttribute('inert', ''); } else { el.removeAttribute('inert'); } } }); }
        function openMenu() {
            clearTimeout(hideTimer);
            menu.hidden = false;
            void menu.offsetWidth; /* commit the hidden state so the entrance animates */
            items.forEach(function (el, i) { el.style.transitionDelay = Math.min(i * 60, 400) + 'ms'; });
            menu.classList.add('is-open');
            toggle.setAttribute('aria-expanded', 'true'); toggle.setAttribute('aria-label', 'Close menu');
            root.classList.add('menu-open'); setInert(true);
        }
        function closeMenu(returnFocus) {
            items.forEach(function (el) { el.style.transitionDelay = '0ms'; });
            menu.classList.remove('is-open');
            toggle.setAttribute('aria-expanded', 'false'); toggle.setAttribute('aria-label', 'Open menu');
            root.classList.remove('menu-open'); setInert(false);
            hideTimer = setTimeout(function () { menu.hidden = true; }, 300);
            if (returnFocus) { toggle.focus(); }
        }
        if (toggle && menu) {
            toggle.addEventListener('click', function () { if (menu.classList.contains('is-open')) { closeMenu(false); } else { openMenu(); } });
            menu.addEventListener('click', function (e) { if (e.target.closest('a')) { closeMenu(false); } });
            document.addEventListener('keydown', function (e) {
                if (!menu.classList.contains('is-open')) { return; }
                if (e.key === 'Escape') { closeMenu(true); return; }
                if (e.key === 'Tab') {
                    var focusables = [toggle].concat(Array.prototype.slice.call(menu.querySelectorAll('a')));
                    var first = focusables[0], last = focusables[focusables.length - 1];
                    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
                    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
                }
            });
            var desktop = window.matchMedia('(min-width: 900px)');
            if (desktop.addEventListener) { desktop.addEventListener('change', function (e) { if (e.matches && menu.classList.contains('is-open')) { closeMenu(false); } }); }
        }

        if (!('IntersectionObserver' in window)) {
            document.querySelectorAll('[data-reveal]').forEach(function (el) { el.classList.add('is-visible'); });
            document.querySelectorAll('.tagline .word').forEach(function (el) { el.classList.add('is-on'); });
            return;
        }

        /* --- Scroll reveals: each section once, never replayed ----------------- */
        var revealObserver = new IntersectionObserver(function (entries, obs) {
            entries.forEach(function (entry) { if (entry.isIntersecting) { entry.target.classList.add('is-visible'); obs.unobserve(entry.target); } });
        }, { rootMargin: '0px 0px -10% 0px', threshold: 0 });
        document.querySelectorAll('[data-reveal]').forEach(function (el) { revealObserver.observe(el); });

        /* --- Current section in the nav ---------------------------------------- */
        var spyLinks = Array.prototype.slice.call(document.querySelectorAll('[data-spy]'));
        var sectionIds = [];
        spyLinks.forEach(function (a) { var id = a.getAttribute('href').slice(1); if (sectionIds.indexOf(id) === -1) { sectionIds.push(id); } });
        function setCurrent(id) {
            spyLinks.forEach(function (a) {
                var on = a.getAttribute('href') === '#' + id;
                a.classList.toggle('is-current', on);
                if (on) { a.setAttribute('aria-current', 'true'); } else { a.removeAttribute('aria-current'); }
            });
        }
        var spyObserver = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) { if (entry.isIntersecting) { setCurrent(entry.target.id); } });
        }, { rootMargin: '-45% 0px -50% 0px' });
        ['top', 'readers', 'sources'].concat(sectionIds).forEach(function (id) { var el = document.getElementById(id); if (el) { spyObserver.observe(el); } });

        /* --- Tagline: words light up one at a time, in reading order ------------- */
        var tagline = document.querySelector('[data-tagline]');
        if (tagline) {
            var words = tagline.querySelectorAll('.word');
            if (reduceMotion) {
                Array.prototype.forEach.call(words, function (w) { w.classList.add('is-on'); });
            } else {
                var shown = 0, ticking = false;
                var update = function () {
                    ticking = false;
                    var rect = tagline.getBoundingClientRect();
                    var vh = window.innerHeight;
                    var start = vh * 0.85, end = vh * 0.45;
                    var progress = (start - rect.top) / (rect.height + (start - end));
                    progress = Math.max(0, Math.min(1, progress));
                    var target = Math.round(progress * words.length);
                    var batch = 0;
                    while (shown < target) { words[shown].style.transitionDelay = (batch * 40) + 'ms'; words[shown].classList.add('is-on'); shown++; batch++; }
                    if (shown >= words.length) { window.removeEventListener('scroll', onScroll); window.removeEventListener('resize', onScroll); }
                };
                var onScroll = function () { if (!ticking) { ticking = true; window.requestAnimationFrame(update); } };
                window.addEventListener('scroll', onScroll, { passive: true });
                window.addEventListener('resize', onScroll);
                update();
            }
        }
    })();

    if ('serviceWorker' in navigator) {
        window.addEventListener('load', function () { navigator.serviceWorker.register('/sw.js').catch(function () {}); });
    }
    </script>
</body>
</html>
"""

# ==============================================================================
# --- 8. Add all templates to the template_storage dictionary ---
# ==============================================================================
template_storage['BASE_HTML_TEMPLATE'] = BASE_HTML_TEMPLATE
template_storage['INDEX_HTML_TEMPLATE'] = INDEX_HTML_TEMPLATE
template_storage['ARTICLE_HTML_TEMPLATE'] = ARTICLE_HTML_TEMPLATE
template_storage['LOGIN_HTML_TEMPLATE'] = LOGIN_HTML_TEMPLATE
template_storage['REGISTER_HTML_TEMPLATE'] = REGISTER_HTML_TEMPLATE
template_storage['PROFILE_HTML_TEMPLATE'] = PROFILE_HTML_TEMPLATE
template_storage['ABOUT_US_HTML_TEMPLATE'] = ABOUT_US_HTML_TEMPLATE
template_storage['CONTACT_HTML_TEMPLATE'] = CONTACT_HTML_TEMPLATE
template_storage['PRIVACY_POLICY_HTML_TEMPLATE'] = PRIVACY_POLICY_HTML_TEMPLATE
template_storage['404_TEMPLATE'] = ERROR_404_TEMPLATE
template_storage['500_TEMPLATE'] = ERROR_500_TEMPLATE
template_storage['_COMMENT_TEMPLATE'] = _COMMENT_TEMPLATE
template_storage['PUBLIC_PROFILE_HTML_TEMPLATE'] = PUBLIC_PROFILE_HTML_TEMPLATE
template_storage['OFFLINE_TEMPLATE'] = OFFLINE_TEMPLATE
template_storage['LANDING_HTML_TEMPLATE'] = LANDING_HTML_TEMPLATE
template_storage['TERMS_HTML_TEMPLATE'] = TERMS_HTML_TEMPLATE
template_storage['SHARE_TESTIMONIAL_HTML_TEMPLATE'] = SHARE_TESTIMONIAL_HTML_TEMPLATE
template_storage['ADMIN_TESTIMONIALS_HTML_TEMPLATE'] = ADMIN_TESTIMONIALS_HTML_TEMPLATE
template_storage['ADMIN_STORAGE_HTML_TEMPLATE'] = ADMIN_STORAGE_HTML_TEMPLATE
template_storage['_AD_SLOT_TEMPLATE'] = _AD_SLOT_TEMPLATE


# ==============================================================================
# --- 9. App Context & Main Execution Block ---
# ==============================================================================
with app.app_context():
    init_db()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.logger.info(f"Starting Flask app in {'debug' if debug_mode else 'production'} mode on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
