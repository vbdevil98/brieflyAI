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
database_url = os.environ.get('DATABASE_URL') # Using 'database_url' for the value from environment

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

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    # Managed Postgres drops idle connections; without these, the first query after an
    # idle period fails with "server closed the connection unexpectedly".
    'pool_pre_ping': True,
    'pool_recycle': 280,
}
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

@app.route('/')
@app.route('/page/<int:page>')
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

@app.route('/ads.txt')
def ads_txt():
    # Ensure this is your correct AdSense Publisher ID
    ads_content = "google.com, pub-6975904325280886, DIRECT, f08c47fec0942fa0"
    # If you have other ad partners, add their lines here, each on a new line.
    # e.g., ads_content += "\notheradsystem.com, theirPubId, DIRECT, theirTagId"
    return Response(ads_content, mimetype='text/plain')


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

    add(url_for('index', _external=True), 'hourly', '1.0')
    for endpoint in ('about', 'contact', 'privacy'):
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
        "description": "AI-summarized, India-centric news.",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#F6F5F2",
        "theme_color": "#0B0C10",
        "icons": [{
            "src": url_for('app_icon', size=size),
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
        '<rect width="512" height="512" rx="96" fill="#0B0C10"/>'
        '<path d="M286 60 154 292h84l-24 160 148-236h-88z" fill="#14B8A6"/>'
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
const VERSION = 'brieflyai-v1';
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
  if (/^\\/(login|logout|register|profile|add_comment|vote_comment|delete_comment|edit_comment|toggle_bookmark|post_article|report_article)/.test(url.pathname)) {
    return;
  }

  // Static assets from our allowed CDNs: cache-first, they are versioned URLs.
  if (url.origin !== self.location.origin) {
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


def prune_old_data(article_days=None, analysis_days=None, stat_days=None):
    """
    Delete rows the app no longer needs, to keep a small database within its quota.

    Community articles and their comments are user-generated and are NEVER deleted here;
    only derived/cached rows are pruned.
    """
    article_days = article_days or int(os.environ.get('RETAIN_ANALYSIS_DAYS', '45'))
    analysis_days = analysis_days or article_days
    stat_days = stat_days or int(os.environ.get('RETAIN_STATS_DAYS', '90'))
    removed = {}
    try:
        analysis_cutoff = datetime.now(timezone.utc) - timedelta(days=analysis_days)
        removed['analysis'] = (ArticleAnalysis.query
                               .filter(ArticleAnalysis.created_at < analysis_cutoff)
                               .delete(synchronize_session=False))
        stat_cutoff = datetime.now(timezone.utc) - timedelta(days=stat_days)
        removed['stats'] = (ArticleStat.query
                            .filter(ArticleStat.last_viewed_at < stat_cutoff)
                            .delete(synchronize_session=False))
        # Bookmarks pointing at API articles that aged out are already shown as
        # "stale" in the UI; they are tiny, so they are kept deliberately.
        db.session.commit()
        app.logger.info(f"Pruned old rows: {removed}")
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Pruning failed: {e}", exc_info=True)
        removed['error'] = str(e)
    return removed


@app.route('/admin/maintenance', methods=['POST'])
@login_required
@rate_limit(6, 3600, scope='maintenance')
def run_maintenance():
    """Manual prune, so a small database can be kept under quota without shell access."""
    if not session.get('is_admin') or session.get('username') != ADMIN_USERNAME:
        abort(403)
    result = prune_old_data()
    return jsonify({"success": 'error' not in result, "removed": result})


@app.route('/admin/storage')
@login_required
def storage_report():
    """Row counts per table, so you can see what is actually consuming the quota."""
    if not session.get('is_admin') or session.get('username') != ADMIN_USERNAME:
        abort(403)
    try:
        report = {
            "community_articles": CommunityArticle.query.count(),
            "comments": Comment.query.count(),
            "users": User.query.count(),
            "bookmarks": BookmarkedArticle.query.count(),
            "ai_analysis_rows": ArticleAnalysis.query.count(),
            "article_stats": ArticleStat.query.count(),
            "subscribers": Subscriber.query.count(),
            "reports": ReportedArticle.query.count(),
        }
        return jsonify({"success": True, "counts": report})
    except Exception as e:
        app.logger.error(f"Storage report failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": "Could not build the report."}), 500


@app.route('/healthz')
def health_check():
    """Liveness/readiness probe for the host platform."""
    status = {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
    code = 200
    try:
        db.session.execute(db.text('SELECT 1'))
        status["database"] = "ok"
    except Exception as e:
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
    return jsonify(status), code

# ==============================================================================
# --- 7. HTML Templates (Stored in memory) ---
# ==============================================================================
BASE_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#0B0C10">
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <link rel="manifest" href="{{ url_for('web_manifest') }}">
    <link rel="icon" href="{{ url_for('app_icon', size=192) }}" type="image/svg+xml">
    <link rel="alternate" type="application/rss+xml" title="BrieflyAI - Community Stories" href="{{ url_for('rss_feed') }}">
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

    {# Two families only, one request, display=swap so text paints immediately in a fallback. #}
    <link rel="preload" as="style" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap">

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

    {# Font Awesome is decorative only -- load it without blocking first paint. #}
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" media="print" onload="this.media='all';this.onload=null;">
    <noscript><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"></noscript>

    <style>
        /* ==========================================================================
           DESIGN TOKENS
           ========================================================================== */
        :root {
            --primary-color: #4338CA; --primary-light: #6D5FF7; --primary-dark: #2D278A;
            --secondary-color: #0D9488; --secondary-light: #14B8A6; --accent-color: #EA580C;
            --text-color: #0B0C0F; --text-muted-color: #5B5F6A;
            --light-bg: #F6F5F2; --card-bg: #FFFFFF; --card-border-color: #E4E2DB;
            --footer-bg: #0A0A0D; --footer-text: #A9ACB6; --footer-link-hover: var(--secondary-light);
            --primary-color-rgb: 67, 56, 202; --secondary-color-rgb: 13, 148, 136; --accent-color-rgb: 234, 88, 12; --text-muted-color-rgb: 91, 95, 106;
            --card-bg-rgb: 255, 255, 255;
            --bookmark-active-color: var(--accent-color);

            --glass-bg: rgba(11, 12, 16, 0.72);
            --glass-border: rgba(255, 255, 255, 0.10);
            --glass-blur: 16px;
            --surface-gradient: linear-gradient(160deg, rgba(var(--primary-color-rgb), 0.05), rgba(var(--secondary-color-rgb), 0.03));

            --shadow-sm: 0 1px 2px 0 rgb(10 10 12 / 0.05);
            --shadow-md: 0 4px 14px -3px rgb(10 10 12 / 0.10), 0 2px 5px -2px rgb(10 10 12 / 0.05);
            --shadow-lg: 0 18px 38px -10px rgb(10 10 12 / 0.18), 0 8px 14px -6px rgb(10 10 12 / 0.08);
            --shadow-glow: 0 8px 24px -6px rgba(var(--primary-color-rgb), 0.45);

            --border-radius-xs: 0.375rem; --border-radius-sm: 0.55rem; --border-radius-md: 0.85rem;
            --border-radius-lg: 1.15rem; --border-radius-pill: 999px;

            --font-display: 'Instrument Serif', 'Iowan Old Style', Georgia, serif;
            --font-sans: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
            --font-body: var(--font-sans);
            --font-mono: ui-monospace, 'SF Mono', SFMono-Regular, Menlo, Consolas, monospace;

            --text-xs: 0.72rem; --text-sm: 0.82rem; --text-base: 1rem; --text-md: 1.06rem;
            --text-lg: 1.16rem; --text-xl: 1.4rem; --text-2xl: 1.8rem;
            --text-3xl: clamp(2.1rem, 4.6vw, 3rem); --text-4xl: clamp(2.5rem, 6vw, 3.75rem);

            --space-1: 0.25rem; --space-2: 0.5rem; --space-3: 0.75rem; --space-4: 1rem; --space-5: 1.5rem; --space-6: 2rem; --space-7: 3rem;

            --ease-standard: cubic-bezier(0.4, 0, 0.2, 1);
            --ease-premium: cubic-bezier(0.16, 1, 0.3, 1);
            --ease-spring: cubic-bezier(0.34, 1.4, 0.64, 1);
            --duration-fast: 150ms; --duration-base: 240ms; --duration-slow: 420ms;

            --skeleton-base: #ECEAE5; --skeleton-sheen: #F8F7F4;
        }
        body.dark-mode {
            --primary-color: #8B82FF; --primary-light: #ADA4FF; --primary-dark: #6D5FF7;
            --secondary-color: #2DD4BF; --secondary-light: #5EEAD4; --accent-color: #FB923C;
            --text-color: #F3F3F5; --text-muted-color: #9AA0AC;
            --light-bg: #08080B; --card-bg: #131317; --card-border-color: #26262E;
            --footer-bg: #000000; --footer-text: #9AA0AC;
            --primary-color-rgb: 139, 130, 255; --secondary-color-rgb: 45, 212, 191; --accent-color-rgb: 251, 146, 60; --text-muted-color-rgb: 154, 160, 172;
            --card-bg-rgb: 19, 19, 23;
            --glass-bg: rgba(8, 8, 11, 0.72);
            --glass-border: rgba(255, 255, 255, 0.08);
            --bookmark-active-color: var(--accent-color);
            --skeleton-base: #1C1C22; --skeleton-sheen: #292930;
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
            padding-top: 84px; margin: 0; font-family: var(--font-sans); font-size: 1rem; line-height: 1.65;
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

        h1, h2, h3, h4, h5 { font-family: var(--font-sans); font-weight: 700; letter-spacing: -0.022em; line-height: 1.22; }

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
            font-weight: 400;
            letter-spacing: -0.01em;
            line-height: 1.12;
        }

        .eyebrow { font-family: var(--font-sans); font-size: var(--text-xs); font-weight: 700; text-transform: uppercase; letter-spacing: 0.13em; color: var(--primary-color); }
        .section-heading { font-size: var(--text-2xl); font-weight: 700; margin: 0; letter-spacing: -0.025em; }

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
        .btn { border-radius: var(--border-radius-sm); font-weight: 600; font-size: 0.9rem; letter-spacing: -0.005em; transition: background-color var(--duration-base) var(--ease-standard), border-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring), color var(--duration-base) var(--ease-standard); }
        .btn:hover { transform: translateY(-1px); }
        .btn:active { transform: translateY(0) scale(0.975); }
        .btn-primary { --bs-btn-bg: var(--primary-color); --bs-btn-border-color: var(--primary-color); --bs-btn-hover-bg: var(--primary-dark); --bs-btn-hover-border-color: var(--primary-dark); --bs-btn-active-bg: var(--primary-dark); --bs-btn-active-border-color: var(--primary-dark); --bs-btn-focus-shadow-rgb: var(--primary-color-rgb); }
        .btn-primary:hover, .btn-primary:focus { box-shadow: var(--shadow-glow); }
        .btn-primary-modal { background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); border: none; color: #fff; border-radius: var(--border-radius-sm); font-weight: 600; padding: 0.65rem 1.5rem; transition: box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring), filter var(--duration-base) var(--ease-standard); }
        .btn-primary-modal:hover { color: #fff; filter: brightness(1.07); box-shadow: var(--shadow-glow); transform: translateY(-1px); }
        .btn-primary-modal:active { transform: translateY(0) scale(0.975); }
        .btn-outline-primary { --bs-btn-color: var(--primary-color); --bs-btn-border-color: var(--primary-color); --bs-btn-hover-bg: var(--primary-color); --bs-btn-hover-border-color: var(--primary-color); --bs-btn-active-bg: var(--primary-color); --bs-btn-active-border-color: var(--primary-color); }
        .btn-outline-secondary { --bs-btn-color: var(--text-muted-color); --bs-btn-border-color: var(--card-border-color); --bs-btn-hover-bg: var(--text-color); --bs-btn-hover-border-color: var(--text-color); }
        .btn-danger { --bs-btn-bg: #DC2626; --bs-btn-border-color: #DC2626; --bs-btn-hover-bg: #B91C1C; --bs-btn-hover-border-color: #B91C1C; }
        .btn .spinner-border { vertical-align: -0.15em; }
        /* Arrow nudge on hover. */
        .btn i.fa-arrow-right, .btn i.fa-chevron-right, .read-more i, .read-more-btn i { transition: transform var(--duration-base) var(--ease-spring); }
        .btn:hover i.fa-arrow-right, .btn:hover i.fa-chevron-right, .read-more:hover i, .read-more-btn:hover i { transform: translateX(4px); }

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
        #alert-placeholder .alert .alert-icon { font-size: 1.05rem; margin-top: 0.15rem; flex-shrink: 0; }
        #alert-placeholder .alert.alert-success .alert-icon { color: var(--secondary-color); }
        #alert-placeholder .alert.alert-danger .alert-icon { color: #DC2626; }
        #alert-placeholder .alert.alert-warning .alert-icon { color: #D97706; }
        #alert-placeholder .alert.alert-info .alert-icon { color: var(--primary-color); }
        #alert-placeholder .alert .btn-close { position: absolute; top: 0.85rem; right: 0.85rem; font-size: 0.75rem; }
        body.dark-mode #alert-placeholder .alert .btn-close { filter: invert(1) grayscale(100%) brightness(200%); }

        /* ==========================================================================
           NAVBAR (glass)
           ========================================================================== */
        .navbar-main {
            position: fixed; top: 0; width: 100%; z-index: 1040; padding: 0.75rem 0;
            background: var(--glass-bg);
            -webkit-backdrop-filter: saturate(180%) blur(var(--glass-blur));
            backdrop-filter: saturate(180%) blur(var(--glass-blur));
            border-bottom: 1px solid var(--glass-border);
            transition: padding var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard), background-color var(--duration-base) var(--ease-standard);
        }
        @supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
            .navbar-main { background: #0B0C10; }
        }
        .navbar-main.is-scrolled { padding: 0.5rem 0; box-shadow: 0 6px 24px -10px rgba(0,0,0,0.55); }
        .navbar-content-wrapper { display: flex; align-items: center; justify-content: space-between; gap: 1rem; width: 100%; }
        .navbar-left { flex-shrink: 0; }
        .navbar-center { flex-grow: 1; min-width: 150px; max-width: 540px; }
        .navbar-right { flex-shrink: 0; }
        .navbar-brand-custom { color: #fff !important; font-weight: 800; font-size: 1.45rem; font-family: var(--font-sans); display: flex; align-items: center; gap: 9px; text-decoration: none !important; letter-spacing: -0.035em; }
        .navbar-brand-custom .brand-icon { color: var(--secondary-light); font-size: 1.2rem; transition: transform var(--duration-base) var(--ease-spring); }
        .navbar-brand-custom:hover .brand-icon { transform: rotate(-12deg) scale(1.12); }

        .search-container { position: relative; width: 100%; }
        .navbar-search { width: 100%; border-radius: var(--border-radius-pill); padding: 0.58rem 2.6rem 0.58rem 2.75rem; border: 1px solid rgba(255,255,255,0.13); font-size: 0.9rem; transition: background-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard), border-color var(--duration-base) var(--ease-standard); background: rgba(255,255,255,0.07); color: #fff; }
        .navbar-search::placeholder { color: rgba(255,255,255,0.5); }
        .navbar-search:focus { background: rgba(255,255,255,0.12); box-shadow: 0 0 0 3.5px rgba(var(--primary-color-rgb), 0.4); border-color: var(--primary-light); outline: none; color: #fff; }
        .search-icon { color: rgba(255,255,255,0.55); transition: color var(--duration-base) var(--ease-standard); left: 1.05rem; position: absolute; top: 50%; transform: translateY(-50%); font-size: 0.85rem; pointer-events: none; }
        .search-container:focus-within .search-icon { color: var(--primary-light); }
        .search-clear-btn { position: absolute; right: 0.45rem; top: 50%; transform: translateY(-50%); background: none; border: none; color: rgba(255,255,255,0.5); width: 28px; height: 28px; border-radius: 50%; display: none; align-items: center; justify-content: center; font-size: 0.8rem; transition: color var(--duration-fast) var(--ease-standard), background-color var(--duration-fast) var(--ease-standard); }
        .search-clear-btn:hover { color: #fff; background: rgba(255,255,255,0.14); }
        .search-container.has-value .search-clear-btn { display: flex; }

        .header-controls { display: flex; gap: 0.8rem; align-items: center; }
        .header-btn { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); padding: 0.5rem 1rem; border-radius: var(--border-radius-pill); color: #fff; font-weight: 600; transition: background-color var(--duration-base) var(--ease-standard), border-color var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); display: flex; align-items: center; gap: 0.5rem; cursor: pointer; text-decoration: none; font-size: 0.86rem; }
        .header-btn:hover { background: var(--primary-color); border-color: var(--primary-color); color: #fff; transform: translateY(-1px); }
        .header-btn:active { transform: scale(0.96); }

        /* ==========================================================================
           OFFCANVAS
           ========================================================================== */
        .offcanvas { background-color: #101116; color: var(--footer-text); z-index: 1045; border-left: 1px solid rgba(255,255,255,0.08); width: min(360px, 88vw); }
        body.dark-mode .offcanvas { background-color: #0B0B0F; }
        .offcanvas-header { border-bottom-color: rgba(255,255,255,0.08) !important; padding: 1.25rem 1.5rem; }
        .offcanvas-title { font-family: var(--font-sans); font-weight: 700; letter-spacing: -0.02em; }
        .offcanvas-body { padding: 1.5rem; }
        .offcanvas-header .btn-close { filter: invert(1) grayscale(100%) brightness(200%); }
        .sidebar-section { margin-bottom: 1.5rem; }
        .sidebar-heading { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.13em; color: var(--text-muted-color); margin-bottom: 0.75rem; font-weight: 700; }
        .sidebar-btn { display: flex; align-items: center; padding: 0.75rem 1rem; width: 100%; text-align: left; background-color: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #fff; text-decoration: none; border-radius: var(--border-radius-md); transition: background-color var(--duration-base) var(--ease-standard), border-color var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); }
        .sidebar-btn:hover { background-color: rgba(255,255,255,0.09); border-color: var(--primary-light); color: #fff; transform: translateX(2px); }
        .sidebar-avatar { width: 40px; height: 40px; border-radius: 50%; background-image: linear-gradient(135deg, var(--primary-light), var(--secondary-color)); color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.15rem; flex-shrink: 0; }
        .offcanvas .dropdown-toggle, .offcanvas .dropdown-toggle::after { color: #fff; }
        .offcanvas .dropdown-menu { background-color: #1C1D25; border-color: #2E2F3A; border-radius: var(--border-radius-md); }
        .offcanvas .dropdown-item { color: var(--footer-text); border-radius: var(--border-radius-sm); transition: background-color var(--duration-fast) var(--ease-standard), color var(--duration-fast) var(--ease-standard); }
        .offcanvas .dropdown-item:hover { background-color: var(--primary-color); color: #fff; }
        .offcanvas .nav-link { padding: 0.65rem 1rem; font-weight: 600; border-radius: var(--border-radius-sm); transition: background-color var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); color: var(--footer-text) !important; }
        .offcanvas .nav-link.active { background-image: linear-gradient(135deg, var(--primary-light), var(--primary-dark)); color: #fff !important; box-shadow: var(--shadow-md); }
        .offcanvas .nav-link:not(.active):hover { background-color: rgba(255,255,255,0.08); color: #fff !important; transform: translateX(3px); }
        .offcanvas-search .navbar-search, .offcanvas-search .search-clear-btn { color: #fff; }
        #dateFilterForm .form-control { background-color: #1C1D25; border-color: #2E2F3A; color: #fff; border-radius: var(--border-radius-sm); }
        #dateFilterForm .form-control:focus { background-color: #1C1D25; border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(var(--primary-color-rgb),0.25); color: #fff; }
        #dateFilterForm .btn-primary { border-radius: var(--border-radius-sm); }

        /* ==========================================================================
           TABS
           ========================================================================== */
        .nav-tabs { border-bottom: 1px solid var(--card-border-color); gap: 0.4rem; }
        .nav-tabs .nav-link { border: none; background: transparent; color: var(--text-muted-color); font-weight: 700; font-size: 0.8rem; letter-spacing: 0.06em; text-transform: uppercase; padding: 0.9rem 0.6rem; border-radius: 0; position: relative; transition: color var(--duration-base) var(--ease-standard); }
        .nav-tabs .nav-link::after { content: ''; position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; background-image: linear-gradient(90deg, var(--primary-color), var(--secondary-color)); transform: scaleX(0); transition: transform var(--duration-base) var(--ease-premium); }
        .nav-tabs .nav-link.active { color: var(--text-color); background: transparent; }
        .nav-tabs .nav-link.active::after { transform: scaleX(1); }
        .nav-tabs .nav-link:not(.active):hover { color: var(--text-color); }
        .nav-tabs .nav-link:not(.active):hover::after { transform: scaleX(0.35); }

        /* ==========================================================================
           CARDS
           ========================================================================== */
        .article-card, .article-full-content-wrapper { background: var(--card-bg); border-radius: var(--border-radius-lg); border: 1px solid var(--card-border-color); box-shadow: var(--shadow-sm); transition: transform var(--duration-base) var(--ease-premium), box-shadow var(--duration-base) var(--ease-premium), border-color var(--duration-base) var(--ease-standard); }
        .article-card { position: relative; overflow: hidden; }
        .article-card::after { content: ''; position: absolute; inset: 0; border-radius: inherit; background: var(--surface-gradient); opacity: 0; transition: opacity var(--duration-base) var(--ease-standard); pointer-events: none; }
        .article-card:hover { transform: translateY(-6px); box-shadow: var(--shadow-lg); border-color: rgba(var(--primary-color-rgb), 0.35); }
        .article-card:hover::after { opacity: 1; }

        .article-image-container { height: 200px; overflow: hidden; position: relative; border-top-left-radius: var(--border-radius-lg); border-top-right-radius: var(--border-radius-lg); background: var(--skeleton-base); background-image: linear-gradient(100deg, var(--skeleton-base) 30%, var(--skeleton-sheen) 50%, var(--skeleton-base) 70%); background-size: 200% 100%; animation: skeletonShimmer 1.5s ease-in-out infinite; }
        .article-image-container.is-loaded, .article-image-container.img-fallback { animation: none; background-image: none; }
        .article-image-container.img-fallback { background: var(--light-bg); }
        .article-image { width: 100%; height: 100%; object-fit: cover; opacity: 0; transition: transform 0.6s var(--ease-premium), opacity var(--duration-slow) var(--ease-standard); }
        .article-image.is-loaded { opacity: 1; }
        .img-fallback .article-image { display: none; }
        .article-card:hover .article-image { transform: scale(1.06); }
        .img-fallback-icon { display: none; position: absolute; inset: 0; align-items: center; justify-content: center; font-size: 1.75rem; color: var(--card-border-color); }
        .img-fallback .img-fallback-icon { display: flex; }

        .article-body { padding: 1.35rem 1.4rem 1.4rem; flex-grow: 1; display: flex; flex-direction: column; position: relative; z-index: 1; }
        .article-title { font-family: var(--font-sans); font-weight: 700; line-height: 1.27; margin-bottom: 0.55rem; font-size: var(--text-lg); letter-spacing: -0.025em; }
        .article-title a { color: var(--text-color); text-decoration: none; transition: color var(--duration-fast) var(--ease-standard); }
        .article-card:hover .article-title a { color: var(--primary-color) !important; }
        .article-meta { display: flex; align-items: center; margin-bottom: 0.85rem; flex-wrap: wrap; gap: 0.35rem 0.45rem; }
        .meta-item { display: flex; align-items: center; font-family: var(--font-sans); font-size: 0.66rem; color: var(--text-muted-color); background: var(--light-bg); border: 1px solid var(--card-border-color); padding: 0.22rem 0.6rem; border-radius: var(--border-radius-pill); font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }
        .meta-item i { font-size: 0.75rem; margin-right: 0.35rem; color: var(--secondary-color); }
        .article-description { color: var(--text-muted-color); margin-bottom: 1.15rem; font-size: 0.9rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .read-more { margin-top: auto; background: var(--text-color); color: var(--card-bg) !important; border: none; padding: 0.62rem 0; border-radius: var(--border-radius-sm); font-weight: 600; font-size: 0.84rem; transition: background-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); width: 100%; text-align: center; text-decoration: none; display: inline-block; }
        .read-more:hover { background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); color: #fff !important; box-shadow: var(--shadow-glow); }
        .read-more:active { transform: scale(0.98); }

        /* Long card grids below the fold: let the browser skip offscreen layout work. */
        .tab-pane .row.g-4 { content-visibility: auto; contain-intrinsic-size: 1px 900px; }

        /* ==========================================================================
           PAGINATION
           ========================================================================== */
        .pagination { flex-wrap: wrap; }
        .page-item .page-link { border-radius: var(--border-radius-sm); width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; color: var(--text-muted-color); background-color: var(--card-bg); border: 1px solid var(--card-border-color); font-weight: 700; transition: color var(--duration-fast) var(--ease-standard), border-color var(--duration-fast) var(--ease-standard), background-color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); font-size: 0.86rem; margin: 0 0.2rem; }
        .page-item .page-link:hover { border-color: var(--primary-color); color: var(--primary-color); transform: translateY(-2px); }
        .page-item.active .page-link { background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); border-color: transparent; color: #fff; }
        .page-item.disabled .page-link { color: var(--text-muted-color); pointer-events: none; background-color: var(--light-bg); opacity: 0.55; }
        .page-link-prev-next .page-link { width: auto; padding-left: 1.1rem; padding-right: 1.1rem; }

        /* ==========================================================================
           FOOTER
           ========================================================================== */
        footer { background: var(--footer-bg); color: var(--footer-text); margin-top: auto; padding: 3.5rem 0 1.5rem; font-size: 0.9rem; position: relative; }
        footer::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(var(--secondary-color-rgb), 0.55), rgba(var(--primary-color-rgb), 0.55), transparent); }
        .footer-content.row { display: flex; flex-wrap: wrap; }
        .footer-section h5 { color: #fff; margin-bottom: 1.2rem; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.13em; }
        .footer-links { display: flex; flex-direction: column; gap: 0.8rem; }
        .footer-links a { color: var(--footer-text); text-decoration: none; transition: color var(--duration-base) var(--ease-standard), padding-left var(--duration-base) var(--ease-premium); }
        .footer-links a:hover { color: var(--footer-link-hover); padding-left: 6px; }
        .social-links { display: flex; gap: 0.6rem; margin-top: 0.5rem; }
        .social-links a { color: var(--footer-text); font-size: 1rem; transition: color var(--duration-base) var(--ease-standard), background-color var(--duration-base) var(--ease-standard), transform var(--duration-base) var(--ease-spring); width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; border-radius: var(--border-radius-sm); background: rgba(255,255,255,0.06); }
        .social-links a:hover { color: #fff; background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); transform: translateY(-3px); }
        .footer-newsletter-input { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.14); color: #fff; }
        .footer-newsletter-input::placeholder { color: rgba(255,255,255,0.5); }
        .footer-newsletter-input:focus { background: rgba(255,255,255,0.1); border-color: var(--primary-light); box-shadow: 0 0 0 3px rgba(var(--primary-color-rgb), 0.4); color: #fff; }
        .copyright { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 0.4rem 1.25rem; text-align: center; padding-top: 2rem; margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.08); font-size: 0.8rem; color: var(--text-muted-color); width: 100%; }
        .back-to-top-link { color: var(--footer-text); background: none; border: none; font-size: 0.8rem; display: inline-flex; align-items: center; gap: 0.35rem; transition: color var(--duration-base) var(--ease-standard); }
        .back-to-top-link:hover { color: var(--secondary-light); }

        /* ==========================================================================
           MODALS
           ========================================================================== */
        .modal-content { border-radius: var(--border-radius-lg); border: 1px solid var(--card-border-color); background-color: var(--card-bg); color: var(--text-color); box-shadow: var(--shadow-lg); }
        body.dark-mode .modal-header .btn-close { filter: invert(1) grayscale(100%) brightness(200%); }
        .admin-controls { position: fixed; bottom: 25px; right: 25px; z-index: 1030; }
        .add-article-btn { width: 56px; height: 56px; border-radius: var(--border-radius-md); color: #fff; border: none; display: flex; align-items: center; justify-content: center; font-size: 20px; cursor: pointer; background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); box-shadow: var(--shadow-lg); transition: box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-base) var(--ease-spring), filter var(--duration-base) var(--ease-standard); }
        .add-article-btn:hover { filter: brightness(1.08); transform: translateY(-4px) scale(1.04); box-shadow: var(--shadow-glow); }
        .add-article-btn:active { transform: translateY(-1px) scale(0.96); }

        /* ==========================================================================
           STATIC PAGES
           ========================================================================== */
        .page-header-static { background-color: var(--card-bg); background-image: var(--surface-gradient); border-radius: var(--border-radius-lg); padding: clamp(2rem, 5vw, 3.25rem); margin-bottom: 2rem; text-align: center; border: 1px solid var(--card-border-color); position: relative; overflow: hidden; }
        .page-header-static::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 3px; background-image: linear-gradient(90deg, var(--primary-color), var(--secondary-color)); }
        .page-header-static h1 { color: var(--text-color); font-size: var(--text-4xl); margin: 0; }
        .static-content-container { background-color: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: clamp(1.5rem, 5vw, 3rem); font-size: 1.03rem; line-height: 1.8; box-shadow: var(--shadow-sm); }
        .static-content-container h2 { font-family: var(--font-sans); font-weight: 700; color: var(--text-color); border-bottom: 2px solid var(--secondary-color); padding-bottom: 0.5rem; margin-top: 2.5rem; margin-bottom: 1.5rem; display: inline-block; font-size: var(--text-xl); }
        .static-content-container h2 .icon { margin-right: 0.75rem; color: var(--primary-color); }
        .static-content-container p, .static-content-container li { max-width: 68ch; }
        .static-content-container p.lead { font-family: var(--font-display); font-weight: 400; font-size: 1.4rem; line-height: 1.4; color: var(--text-color); max-width: 40ch; }
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
        .auth-header { padding: 2.25rem 2rem 2rem; background-image: linear-gradient(135deg, var(--primary-light), var(--primary-dark)); text-align: center; }
        .auth-header .icon { font-size: 2rem; color: #fff; width: 64px; height: 64px; margin: 0 auto; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.16); border-radius: 50%; }
        .auth-header h1 { font-size: var(--text-2xl); color: #fff; margin-top: 1rem; margin-bottom: 0; }
        .auth-body { padding: 2rem 2.25rem; }
        .input-group-icon { position: relative; }
        .input-group-icon .form-control { padding-left: 2.5rem; }
        .input-group-icon .input-icon { position: absolute; left: 0.8rem; top: 50%; transform: translateY(-50%); color: var(--text-muted-color); transition: color var(--duration-base) var(--ease-standard); }
        .input-group-icon:focus-within .input-icon { color: var(--primary-color); }
        .auth-body .btn { padding: 0.75rem; font-weight: 700; font-size: 1rem; }
        .auth-footer { padding: 1.25rem 2.25rem; text-align: center; border-top: 1px solid var(--card-border-color); background: var(--light-bg); }

        /* ==========================================================================
           PROFILE
           ========================================================================== */
        .profile-header-card { background: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: 2rem; box-shadow: var(--shadow-sm); display: flex; flex-direction: column; align-items: center; text-align: center; position: relative; overflow: hidden; }
        .profile-header-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 96px; background-image: linear-gradient(135deg, rgba(var(--primary-color-rgb), 0.16), rgba(var(--secondary-color-rgb), 0.10)); }
        .profile-avatar-wrapper { position: relative; margin-bottom: 1rem; z-index: 1; }
        .profile-avatar { width: 116px; height: 116px; border-radius: 50%; background-image: linear-gradient(150deg, var(--primary-light), var(--primary-dark)); color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 3.2rem; font-family: var(--font-sans); border: 5px solid var(--card-bg); box-shadow: var(--shadow-md); }
        .profile-header-card h1 { margin-bottom: 0.25rem; font-size: var(--text-3xl); position: relative; z-index: 1; }
        .profile-header-card .username { color: var(--text-muted-color); font-weight: 600; margin-bottom: 1rem; }
        .profile-stats { display: flex; gap: 1.5rem; margin-top: 1.5rem; border-top: 1px solid var(--card-border-color); padding-top: 1.5rem; width: 100%; justify-content: center; }
        .stat-item { text-align: center; background: none; border: none; padding: 0.4rem 1rem; border-radius: var(--border-radius-md); transition: background-color var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); }
        button.stat-item { cursor: pointer; }
        button.stat-item:hover { background-color: var(--light-bg); transform: translateY(-2px); }
        .stat-item .icon { font-size: 1.4rem; color: var(--secondary-color); margin-bottom: 0.5rem; }
        .stat-item .count { font-size: 1.3rem; font-weight: 800; color: var(--text-color); letter-spacing: -0.03em; }
        .stat-item .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; color: var(--text-muted-color); }
        .profile-tabs .nav-link { padding: 0.75rem 1rem; }

        /* ==========================================================================
           EMPTY / ERROR STATES
           ========================================================================== */
        .state-card { background-color: var(--card-bg); background-image: var(--surface-gradient); border-radius: var(--border-radius-lg); text-align: center; padding: clamp(2.25rem, 6vw, 3.5rem) 2rem; border: 1px dashed var(--card-border-color); }
        .state-card.state-card-narrow { max-width: 600px; margin-left: auto; margin-right: auto; }
        .state-card.state-card-solid { border-style: solid; box-shadow: var(--shadow-sm); }
        .state-card-icon { width: 72px; height: 72px; border-radius: 50%; background-image: linear-gradient(140deg, rgba(var(--primary-color-rgb), 0.16), rgba(var(--secondary-color-rgb), 0.10)); color: var(--primary-color); display: inline-flex; align-items: center; justify-content: center; font-size: 1.6rem; margin-bottom: 1.25rem; }
        .state-card-icon.state-card-icon-sm { width: 52px; height: 52px; font-size: 1.1rem; margin-bottom: 0.85rem; }
        .state-card.state-card-danger .state-card-icon { background-image: linear-gradient(140deg, rgba(220, 38, 38, 0.18), rgba(220, 38, 38, 0.08)); color: #DC2626; }
        .state-card-title { font-size: var(--text-2xl); margin-bottom: 0.5rem; }
        .state-card-text { color: var(--text-muted-color); max-width: 46ch; margin: 0 auto; }
        .state-card-actions { margin-top: 1.5rem; display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; }

        /* ==========================================================================
           BOOKMARK
           ========================================================================== */
        .bookmark-btn { background: none; border: none; font-size: 1.5rem; color: var(--text-muted-color); cursor: pointer; padding: 0.25rem 0.5rem; transition: color var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); vertical-align: middle; border-radius: 50%; }
        .bookmark-btn.active { color: var(--bookmark-active-color); transform: scale(1.1); }
        .bookmark-btn:hover { color: var(--secondary-light); transform: scale(1.12); }
        .article-card .bookmark-btn { font-size: 1.2rem; position: relative; z-index: 2; }

        /* ==========================================================================
           AI SUMMARY / TAKEAWAYS / SKELETON
           ========================================================================== */
        .summary-box, .takeaways-box { background-color: var(--card-bg); background-image: var(--surface-gradient); border: 1px solid rgba(var(--primary-color-rgb), 0.16); border-radius: var(--border-radius-md); margin: 1.6rem 0; padding: 1.5rem 1.65rem; }
        .summary-box h2, .takeaways-box h2 { font-family: var(--font-sans); font-weight: 800; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.13em; color: var(--primary-color); margin-bottom: 0.75rem; }
        .summary-box p { font-size: 1.12rem; line-height: 1.6; max-width: 62ch; margin: 0; }
        .takeaways-box { border-left: 3px solid var(--secondary-color); }
        .takeaways-box h2 { color: var(--secondary-color); }
        .takeaways-box ul { margin: 0; padding-left: 1.15rem; }
        .takeaways-box li { margin-bottom: 0.45rem; }
        .takeaways-box li:last-child { margin-bottom: 0; }
        .content-text { white-space: pre-wrap; line-height: 1.8; font-size: 1.07rem; color: var(--text-color); max-width: 70ch; }
        .content-divider-heading { font-size: var(--text-xl); margin: 0 0 1rem; }
        .comment-login-prompt { text-align: center; margin-top: 1.5rem; padding: 1.75rem; background: var(--light-bg); border-radius: var(--border-radius-md); border: 1px solid var(--card-border-color); }
        .comment-login-prompt i { font-size: 1.5rem; color: var(--text-muted-color); margin-bottom: 0.5rem; display: block; }

        .ai-skeleton-box { border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); padding: 1.25rem 1.4rem; margin-bottom: 1.25rem; }
        .ai-skeleton-label { height: 0.72rem; width: 120px; border-radius: var(--border-radius-xs); margin-bottom: 0.7rem; }
        .ai-skeleton-line { height: 0.82rem; border-radius: var(--border-radius-xs); margin-bottom: 0.6rem; }
        .ai-skeleton-line:last-child { margin-bottom: 0; }
        .ai-skeleton-line.w-100 { width: 100%; } .ai-skeleton-line.w-95 { width: 95%; } .ai-skeleton-line.w-90 { width: 90%; } .ai-skeleton-line.w-80 { width: 80%; } .ai-skeleton-line.w-70 { width: 70%; }
        .ai-skeleton-label, .ai-skeleton-line { background: linear-gradient(100deg, var(--skeleton-base) 30%, var(--skeleton-sheen) 50%, var(--skeleton-base) 70%); background-size: 200% 100%; animation: skeletonShimmer 1.5s ease-in-out infinite; }
        .ai-skeleton-caption { color: var(--text-muted-color); font-size: var(--text-sm); text-align: center; margin: 0.25rem 0 0; }
        .ai-skeleton-caption i { color: var(--primary-color); }

        .hero-image-wrap { position: relative; overflow: hidden; border-radius: var(--border-radius-md); box-shadow: var(--shadow-md); margin: 1.25rem 0; aspect-ratio: 16 / 8; max-height: 460px; background: var(--skeleton-base); background-image: linear-gradient(100deg, var(--skeleton-base) 30%, var(--skeleton-sheen) 50%, var(--skeleton-base) 70%); background-size: 200% 100%; animation: skeletonShimmer 1.5s ease-in-out infinite; }
        .hero-image-wrap.is-loaded, .hero-image-wrap.img-fallback { animation: none; background-image: none; }
        .hero-image-wrap.img-fallback { background: var(--light-bg); }
        .hero-image-wrap .hero-image { width: 100%; height: 100%; object-fit: cover; display: block; opacity: 0; transition: opacity var(--duration-slow) var(--ease-standard); }
        .hero-image-wrap .hero-image.is-loaded { opacity: 1; }
        .hero-image-wrap.img-fallback .hero-image { display: none; }
        .hero-image-wrap .img-fallback-icon { font-size: 2.25rem; }
        .article-full-content-wrapper { padding: clamp(1.25rem, 4vw, 3rem); margin: 1rem auto 2rem; max-width: 860px; box-shadow: var(--shadow-md); }
        .article-title-main { font-size: clamp(2.1rem, 5vw, 3.1rem); color: var(--text-color); margin-bottom: 0.75rem; }
        .article-meta-detailed .meta-item { font-size: 0.68rem; }

        /* ==========================================================================
           COMMENTS
           ========================================================================== */
        .comment-section h2 { padding-bottom: 0.75rem; border-bottom: 1px solid var(--card-border-color); font-size: var(--text-xl); }
        .comment-form-heading { font-size: var(--text-lg); font-weight: 700; margin-bottom: 1rem; }
        .comment-thread { position: relative; }
        #comments-list > .comment-thread + .comment-thread { margin-top: 1.75rem; padding-top: 1.75rem; border-top: 1px solid var(--card-border-color); }
        .comment-container { display: flex; gap: 1rem; align-items: flex-start; }
        .comment-replies { margin-left: 3.5rem; padding-left: 1.25rem; margin-top: 1.25rem; border-left: 2px solid var(--card-border-color); }
        .comment-replies.comment-replies-flat { margin-left: 1.5rem; padding-left: 1rem; }
        .comment-replies:empty { display: none; margin: 0; padding: 0; border: 0; }
        .comment-replies > .comment-thread + .comment-thread { margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px dashed var(--card-border-color); }
        .comment-avatar { width: 44px; height: 44px; border-radius: 50%; background-image: linear-gradient(140deg, var(--primary-light), var(--primary-dark)); color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700; flex-shrink: 0; box-shadow: var(--shadow-sm); }
        .comment-replies .comment-avatar { width: 38px; height: 38px; font-size: 0.9rem; }
        .comment-body { flex-grow: 1; min-width: 0; }
        .comment-header { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.25rem; }
        .comment-author { font-weight: 700; color: var(--text-color); }
        .comment-date { font-size: 0.78rem; color: var(--text-muted-color); }
        .comment-content { word-wrap: break-word; }
        .comment-actions { position: relative; display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; margin-top: 0.5rem; }
        .comment-actions button { background: none; border: none; color: var(--text-muted-color); padding: 0.28rem 0.6rem; border-radius: var(--border-radius-pill); font-size: 0.82rem; font-weight: 600; display: flex; align-items: center; gap: 0.3rem; transition: color var(--duration-fast) var(--ease-standard), background-color var(--duration-fast) var(--ease-standard); }
        .comment-actions button:hover { color: var(--primary-color); background-color: rgba(var(--primary-color-rgb), 0.1); }
        .react-btn { position: relative; }
        .reaction-box { display: none; position: absolute; bottom: 100%; left: 0; margin-bottom: 8px; background-color: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-pill); padding: 4px 8px; box-shadow: var(--shadow-lg); z-index: 10; white-space: nowrap; animation: fadeInUp 0.22s var(--ease-spring); }
        .reaction-box.show { display: flex; gap: 2px; }
        .reaction-emoji { font-size: 1.3rem; cursor: pointer; transition: transform 0.18s var(--ease-spring), background-color var(--duration-fast) var(--ease-standard); padding: 4px; background: none; border: none; border-radius: 50%; line-height: 1; }
        .reaction-emoji:hover { transform: scale(1.3) translateY(-2px); }
        .reaction-emoji.is-selected { background-color: rgba(var(--primary-color-rgb), 0.16); box-shadow: inset 0 0 0 1.5px rgba(var(--primary-color-rgb), 0.5); }
        .reaction-summary { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
        .reaction-pill { display: flex; align-items: center; background-color: rgba(var(--primary-color-rgb), 0.08); border: 1px solid transparent; border-radius: var(--border-radius-pill); padding: 2px 9px; font-size: 0.78rem; font-weight: 600; cursor: default; transition: background-color var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard); }
        .reaction-pill.user-reacted { background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); color: #fff; border-color: transparent; }
        .reaction-pill .emoji { font-size: 0.9rem; margin-right: 4px; }
        .reply-form-container, .edit-form-container { padding: 1rem; border-radius: var(--border-radius-md); margin-top: 0.75rem; background-color: var(--light-bg); border: 1px solid var(--card-border-color); display: none; }

        /* ==========================================================================
           COMMUNITY HUB + LIST HEADERS
           ========================================================================== */
        .community-hub-header { display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap; background-color: var(--card-bg); background-image: linear-gradient(135deg, rgba(var(--primary-color-rgb),0.09), rgba(var(--secondary-color-rgb),0.06)); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: clamp(1.5rem, 4vw, 2.5rem); margin-bottom: 2rem; }
        .community-hub-header-text { max-width: 640px; }
        .community-hub-header h1 { font-family: var(--font-display); font-weight: 400; font-size: var(--text-3xl); margin: 0.4rem 0 0.6rem; }
        .community-hub-sub { color: var(--text-muted-color); margin: 0; max-width: 58ch; }
        .community-hub-cta { flex-shrink: 0; white-space: nowrap; }
        .list-page-header { margin-bottom: 1.75rem; padding-bottom: 1.25rem; border-bottom: 1px solid var(--card-border-color); }
        .list-page-header .eyebrow { display: block; margin-bottom: 0.45rem; }
        .list-page-header h1 { margin: 0; font-family: var(--font-display); font-weight: 400; font-size: var(--text-3xl); }

        /* ==========================================================================
           FEATURED STORY (parallax layer) + AI SYNTHESIS
           ========================================================================== */
        .featured-story { display: flex; background-color: var(--card-bg); border-radius: var(--border-radius-lg); box-shadow: var(--shadow-lg); margin-bottom: 2rem; overflow: hidden; border: 1px solid var(--card-border-color); position: relative; }
        .featured-story-image { flex: 0 0 55%; background-size: cover; background-position: center; background-color: var(--skeleton-base); min-height: 440px; position: relative; will-change: transform; transform: translate3d(0, var(--parallax-y, 0px), 0) scale(1.08); transition: transform 80ms linear; }
        .featured-story-image::after { content: ''; position: absolute; inset: 0; background: linear-gradient(100deg, rgba(0,0,0,0) 55%, rgba(var(--card-bg-rgb), 0.35) 100%); }
        .featured-story-content { flex: 0 0 45%; padding: clamp(1.5rem, 4vw, 3rem); display: flex; flex-direction: column; justify-content: center; position: relative; }
        .featured-story-content .meta-item:first-child { background-image: linear-gradient(135deg, rgba(var(--accent-color-rgb), 0.2), rgba(var(--accent-color-rgb), 0.1)); color: var(--accent-color); border-color: transparent; font-weight: 700; }
        .featured-story-content .meta-item:first-child i { color: var(--accent-color); }
        .featured-story-content h2 { font-size: var(--text-3xl); margin: 1rem 0; }
        .featured-story-content h2 a { color: var(--text-color); text-decoration: none; transition: color var(--duration-base) var(--ease-standard); }
        .featured-story-content h2 a:hover { color: var(--primary-color); }
        .featured-story-content .description { font-size: 1.02rem; color: var(--text-muted-color); margin-bottom: 2rem; }
        .featured-story-content .read-more-btn { background: var(--text-color); color: var(--card-bg); padding: 0.8rem 1.6rem; text-decoration: none; border-radius: var(--border-radius-sm); font-weight: 600; font-size: 0.9rem; transition: background-color var(--duration-base) var(--ease-standard), box-shadow var(--duration-base) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); align-self: flex-start; }
        .featured-story-content .read-more-btn:hover { background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); color: #fff; box-shadow: var(--shadow-glow); transform: translateY(-2px); }

        .ai-synthesis-card { background-color: var(--card-bg); background-image: var(--surface-gradient); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg); padding: clamp(1.5rem, 4vw, 2.5rem); margin-bottom: 2rem; box-shadow: var(--shadow-sm); position: relative; overflow: hidden; }
        .ai-synthesis-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background-image: linear-gradient(180deg, var(--primary-color), var(--secondary-color)); }
        .synthesis-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.1rem; }
        .synthesis-header i { font-size: 0.95rem; color: #fff; background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center; border-radius: var(--border-radius-sm); flex-shrink: 0; }
        .synthesis-header h2 { font-size: 0.72rem; margin: 0; font-weight: 800; text-transform: uppercase; letter-spacing: 0.13em; color: var(--text-muted-color); }
        .synthesis-text { font-size: clamp(1.35rem, 2.6vw, 1.75rem); line-height: 1.35; text-align: left; color: var(--text-color); position: relative; max-width: 40ch; margin: 0; }
        .synthesis-keywords { margin-top: 1.5rem; padding-top: 1.25rem; border-top: 1px solid var(--card-border-color); text-align: left; position: relative; }
        .synthesis-keywords .keyword-tag { display: inline-block; background-color: var(--light-bg); border: 1px solid var(--card-border-color); color: var(--text-color); padding: 0.35rem 0.95rem; border-radius: var(--border-radius-pill); margin: 0.2rem 0.3rem 0.2rem 0; font-size: 0.78rem; font-weight: 600; text-decoration: none; transition: background-color var(--duration-fast) var(--ease-standard), color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); }
        .synthesis-keywords .keyword-tag:hover { background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); color: #fff; border-color: transparent; transform: translateY(-2px); }

        /* ==========================================================================
           READING PROGRESS BAR
           ========================================================================== */
        .reading-progress { position: fixed; top: 0; left: 0; height: 3px; width: 100%; z-index: 1050; background: transparent; pointer-events: none; }
        .reading-progress__fill { height: 100%; width: 0%; background-image: linear-gradient(90deg, var(--primary-light), var(--secondary-color)); transform-origin: left; transition: width 80ms linear; }

        /* ==========================================================================
           ARTICLE TOOLBAR (read time, listen, text size, share)
           ========================================================================== */
        .article-toolbar { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; padding: 0.85rem 0; margin: 1.25rem 0; border-top: 1px solid var(--card-border-color); border-bottom: 1px solid var(--card-border-color); }
        .toolbar-btn { display: inline-flex; align-items: center; gap: 0.4rem; background: none; border: 1px solid var(--card-border-color); color: var(--text-muted-color); border-radius: var(--border-radius-pill); padding: 0.35rem 0.85rem; font-size: 0.8rem; font-weight: 600; cursor: pointer; transition: color var(--duration-fast) var(--ease-standard), border-color var(--duration-fast) var(--ease-standard), background-color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-spring); white-space: nowrap; }
        .toolbar-btn:hover { color: var(--primary-color); border-color: var(--primary-color); transform: translateY(-1px); }
        .toolbar-btn.is-active { background-image: linear-gradient(135deg, var(--primary-light), var(--primary-color)); color: #fff; border-color: transparent; }
        .read-time-badge { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.8rem; font-weight: 600; color: var(--text-muted-color); margin-right: auto; }
        .read-time-badge i { color: var(--secondary-color); }
        .toolbar-spacer { margin-left: auto; }

        /* Share menu */
        .share-wrap { position: relative; }
        .share-menu { display: none; position: absolute; right: 0; bottom: calc(100% + 8px); background: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); box-shadow: var(--shadow-lg); padding: 0.4rem; min-width: 190px; z-index: 20; animation: fadeInUp 0.22s var(--ease-spring); }
        .share-menu.show { display: block; }
        .share-menu button, .share-menu a { display: flex; align-items: center; gap: 0.65rem; width: 100%; background: none; border: none; color: var(--text-color); text-decoration: none; padding: 0.55rem 0.75rem; border-radius: var(--border-radius-sm); font-size: 0.86rem; font-weight: 500; text-align: left; transition: background-color var(--duration-fast) var(--ease-standard); }
        .share-menu button:hover, .share-menu a:hover { background-color: var(--light-bg); color: var(--text-color); }
        .share-menu i { width: 1.1rem; text-align: center; }
        .share-menu .i-whatsapp { color: #25D366; } .share-menu .i-x { color: var(--text-color); }
        .share-menu .i-facebook { color: #1877F2; } .share-menu .i-linkedin { color: #0A66C2; }
        .share-menu .i-link { color: var(--primary-color); }

        /* Text-size presets, applied to the article body */
        .article-full-content-wrapper[data-text-size="large"] .content-text,
        .article-full-content-wrapper[data-text-size="large"] .summary-box p,
        .article-full-content-wrapper[data-text-size="large"] .takeaways-box li { font-size: 1.22rem; line-height: 1.85; }
        .article-full-content-wrapper[data-text-size="xlarge"] .content-text,
        .article-full-content-wrapper[data-text-size="xlarge"] .summary-box p,
        .article-full-content-wrapper[data-text-size="xlarge"] .takeaways-box li { font-size: 1.4rem; line-height: 1.9; }
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
        .recent-item__title { font-size: 0.88rem; font-weight: 700; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; margin: 0 0 0.35rem; letter-spacing: -0.02em; }
        .recent-item__source { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; color: var(--text-muted-color); }
        .link-btn { background: none; border: none; color: var(--text-muted-color); font-size: 0.78rem; font-weight: 600; text-decoration: underline; text-underline-offset: 0.2em; padding: 0; }
        .link-btn:hover { color: var(--primary-color); }

        /* ==========================================================================
           KEYBOARD SHORTCUTS
           ========================================================================== */
        .kbd-list { display: grid; grid-template-columns: 1fr auto; gap: 0.6rem 1.5rem; align-items: center; }
        .kbd-list dt { color: var(--text-color); font-size: 0.9rem; }
        .kbd-list dd { margin: 0; text-align: right; }
        kbd { font-family: var(--font-mono); font-size: 0.75rem; background: var(--light-bg); border: 1px solid var(--card-border-color); border-bottom-width: 2px; border-radius: var(--border-radius-xs); padding: 0.15rem 0.45rem; color: var(--text-color); }

        /* Comment sort control */
        .comment-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
        .sort-select { font-size: 0.82rem; font-weight: 600; padding: 0.35rem 2rem 0.35rem 0.75rem; border-radius: var(--border-radius-pill); border: 1px solid var(--card-border-color); background-color: var(--card-bg); color: var(--text-color); width: auto; }

        /* ==========================================================================
           PRINT
           ========================================================================== */
        @media print {
            .navbar-main, .offcanvas, footer, .admin-controls, .article-toolbar, .reading-progress,
            #alert-placeholder, .comment-section, .recent-strip, .modal, .offcanvas-backdrop,
            .bookmark-btn, .read-more, .pagination, .btn { display: none !important; }
            body { padding-top: 0; background: #fff; color: #000; font-size: 12pt; }
            .article-full-content-wrapper { box-shadow: none; border: none; padding: 0; max-width: 100%; }
            .summary-box, .takeaways-box { border: 1px solid #ccc; background: none !important; break-inside: avoid; }
            .content-text, p { max-width: none; }
            a[href^="http"]::after { content: " (" attr(href) ")"; font-size: 9pt; color: #555; word-break: break-all; }
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
            .footer-section { text-align: center; }
            .footer-links { align-items: center; }
            .social-links { justify-content: center; }
            .footer-links a:hover { padding-left: 0; }
            .copyright { flex-direction: column; gap: 0.5rem; }
        }
        @media (max-width: 575.98px) {
            body { font-size: 0.95rem; padding-top: 76px; }
            .navbar-brand-custom { font-size: 1.3rem; }
            .comment-replies { margin-left: 1.25rem; padding-left: 1rem; }
            .comment-replies.comment-replies-flat { margin-left: 0.85rem; padding-left: 0.75rem; }
            .comment-container { gap: 0.75rem; }
            .comment-avatar { width: 38px; height: 38px; }
            .comment-replies .comment-avatar { width: 33px; height: 33px; }
            .profile-avatar { width: 96px; height: 96px; font-size: 2.7rem; }
            .profile-stats { gap: 0.5rem; flex-wrap: wrap; }
            .auth-card { margin: 1rem auto; border: none; box-shadow: none; }
            .auth-body, .article-body { padding: 1.4rem; }
            #alert-placeholder { top: 84px; width: 94vw; }
        }
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body class="{{ request.cookies.get('darkMode', 'disabled') }}{% block body_class %}{% endblock %}">

    <a class="visually-hidden-focusable" href="#main-content">Skip to main content</a>

    <header>
        <nav class="navbar-main" id="mainNavbar">
            <div class="container">
                <div class="navbar-content-wrapper">
                    <div class="navbar-left">
                        <a class="navbar-brand-custom" href="{{ url_for('index') }}">
                            <i class="fas fa-bolt-lightning brand-icon" aria-hidden="true"></i>
                            <span>BrieflyAI</span>
                        </a>
                    </div>
                    <div class="navbar-center">
                        <form action="{{ url_for('search_results') }}" method="GET" class="search-container" id="navbarSearchForm">
                            <label for="navbarSearchInput" class="visually-hidden">Search news articles</label>
                            <input type="search" name="query" id="navbarSearchInput" class="form-control navbar-search" placeholder="Search news articles..." value="{{ request.args.get('query', '') }}" autocomplete="off">
                            <i class="fas fa-search search-icon" aria-hidden="true"></i>
                            <button type="button" class="search-clear-btn" aria-label="Clear search"><i class="fas fa-xmark" aria-hidden="true"></i></button>
                        </form>
                    </div>
                    <div class="navbar-right">
                        <div class="header-controls">
                            <button class="header-btn" type="button" data-bs-toggle="offcanvas" data-bs-target="#mainOffcanvas" aria-controls="mainOffcanvas" aria-label="Open menu">
                                <i class="fas fa-bars" aria-hidden="true"></i>
                                <span class="d-none d-sm-inline ms-1">Menu</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    </header>

    <div class="offcanvas offcanvas-end" tabindex="-1" id="mainOffcanvas" aria-labelledby="mainOffcanvasLabel">
        <div class="offcanvas-header">
            <h2 class="offcanvas-title h5" id="mainOffcanvasLabel">BrieflyAI Menu</h2>
            <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close menu"></button>
        </div>
        <div class="offcanvas-body d-flex flex-column">

            <div class="sidebar-section d-md-none">
                <h6 class="sidebar-heading">Search</h6>
                <form action="{{ url_for('search_results') }}" method="GET" class="search-container offcanvas-search">
                    <label for="offcanvasSearchInput" class="visually-hidden">Search news articles</label>
                    <input type="search" name="query" id="offcanvasSearchInput" class="form-control navbar-search" placeholder="Search news articles..." value="{{ request.args.get('query', '') }}" autocomplete="off">
                    <i class="fas fa-search search-icon" aria-hidden="true"></i>
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
                            <li><a class="dropdown-item" href="{{ url_for('profile') }}"><i class="fas fa-id-card fa-fw me-2" aria-hidden="true"></i>Profile</a></li>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item" href="{{ url_for('logout') }}"><i class="fas fa-sign-out-alt fa-fw me-2" aria-hidden="true"></i>Logout</a></li>
                        </ul>
                    </div>
                {% else %}
                    <a href="{{ url_for('login') }}" class="sidebar-btn">
                        <i class="fas fa-sign-in-alt fa-fw me-2" aria-hidden="true"></i> Login / Register
                    </a>
                {% endif %}
            </div>

            <div class="sidebar-section">
                 <button class="sidebar-btn dark-mode-toggle w-100" type="button" aria-pressed="false">
                    <i class="fas fa-moon fa-fw me-2" aria-hidden="true"></i> <span class="theme-text">Dark Mode</span>
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
                                <i class="fas fa-fw fa-{% if cat_item == 'All Articles' %}globe-americas{% elif cat_item == 'Popular Stories' %}fire-alt{% elif cat_item == "Yesterday's Headlines" %}history{% elif cat_item == 'Community Hub' %}users{% endif %} me-2" aria-hidden="true"></i>
                                {{ cat_item }}
                            </a>
                        </li>
                    {% endfor %}
                </ul>
            </div>

            <hr class="my-2">

            <div class="sidebar-section">
                <h6 class="sidebar-heading">Filter by Date</h6>
                <form id="dateFilterForm" class="mt-2">
                    <label for="articleDateFilter" class="visually-hidden">Filter articles by date</label>
                    <div class="input-group">
                        <input type="date" id="articleDateFilter" class="form-control" title="Filter 'All Articles' by date" value="{{ current_filter_date | default('', true) }}">
                        <button class="btn btn-primary" type="submit" title="Apply Date Filter">Go</button>
                    </div>
                     {% if current_filter_date %}
                        <button class="btn btn-sm btn-outline-danger mt-2 w-100" type="button" id="clearDateFilter">Clear Date Filter</button>
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
                    <h2 class="modal-title h5" id="shortcutsModalLabel"><i class="fas fa-keyboard me-2" aria-hidden="true"></i>Keyboard shortcuts</h2>
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
                    <i class="fas {{ {'success': 'fa-circle-check', 'danger': 'fa-circle-exclamation', 'warning': 'fa-triangle-exclamation'}.get(category, 'fa-circle-info') }} alert-icon" aria-hidden="true"></i>
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
            <i class="fas fa-pen-to-square" aria-hidden="true"></i>
        </button>
    </div>
    <div class="modal fade" id="addArticleModal" tabindex="-1" aria-hidden="true" aria-labelledby="addArticleModalLabel">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content p-4">
                <div class="modal-header border-0 pb-0">
                    <h2 class="modal-title h4" id="addArticleModalLabel">Post New Article</h2>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <p class="small text-muted d-flex align-items-center gap-2 mb-3" id="draftStatus" hidden>
                        <i class="fas fa-cloud-arrow-up" aria-hidden="true"></i><span id="draftStatusText">Draft saved</span>
                        <button type="button" class="link-btn ms-auto" id="discardDraftBtn">Discard draft</button>
                    </p>
                    <form id="addArticleForm" action="{{ url_for('post_article') }}" method="POST">
                        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                        <div class="mb-3"><label for="articleTitle" class="form-label">Article Title</label><input type="text" id="articleTitle" name="title" class="form-control" required></div>
                        <div class="mb-3"><label for="articleDescription" class="form-label">Short Description</label><textarea id="articleDescription" name="description" class="form-control" rows="3" required></textarea></div>
                        <div class="mb-3"><label for="articleSource" class="form-label">Source Name</label><input type="text" id="articleSource" name="sourceName" class="form-control" value="Community Post" required></div>
                        <div class="mb-3"><label for="articleImage" class="form-label">Image URL (Optional)</label><input type="url" id="articleImage" name="imageUrl" class="form-control"></div>
                        <div class="mb-3"><label for="articleContent" class="form-label">Full Article Content</label><textarea id="articleContent" name="content" class="form-control" rows="7" required></textarea></div>
                        <div class="d-flex justify-content-end gap-2 mt-4"><button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">Cancel</button><button type="submit" class="btn btn-primary">Post Article</button></div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <footer class="mt-auto">
        <div class="container">
            <div class="footer-content row">
                <div class="footer-section col-lg-4 col-md-6 mb-4">
                    <div class="d-flex align-items-center mb-2">
                        <i class="fas fa-bolt-lightning me-2" style="color:var(--secondary-light); font-size: 1.4rem;" aria-hidden="true"></i>
                        <span class="h5 mb-0" style="color:#fff; font-weight:800; letter-spacing:-0.03em;">BrieflyAI</span>
                    </div>
                    <p class="small text-light">Your premier source for AI summarized, India-centric news.</p>
                    <div class="social-links">
                        <a href="#" title="Twitter" aria-label="BrieflyAI on Twitter"><i class="fab fa-twitter" aria-hidden="true"></i></a><a href="#" title="Facebook" aria-label="BrieflyAI on Facebook"><i class="fab fa-facebook-f" aria-hidden="true"></i></a><a href="#" title="LinkedIn" aria-label="BrieflyAI on LinkedIn"><i class="fab fa-linkedin-in" aria-hidden="true"></i></a><a href="#" title="Instagram" aria-label="BrieflyAI on Instagram"><i class="fab fa-instagram" aria-hidden="true"></i></a>
                    </div>
                </div>
                <div class="footer-section col-lg-2 col-md-6 mb-4">
                    <h5>Quick Links</h5>
                    <div class="footer-links">
                        <a href="{{ url_for('index') }}">Home</a>
                        <a href="{{ url_for('about') }}">About Us</a>
                        <a href="{{ url_for('contact') }}">Contact</a>
                        <a href="{{ url_for('privacy') }}">Privacy Policy</a>
                        <a href="{{ url_for('rss_feed') }}">RSS Feed</a>
                        {% if session.user_id %}<a href="{{ url_for('profile') }}">My Profile</a>{% endif %}
                    </div>
                </div>
                <div class="footer-section col-lg-2 col-md-6 mb-4">
                    <h5>Categories</h5>
                    <div class="footer-links">
                        {% for cat_item in categories %}<a href="{{ url_for('index', category_name=cat_item, page=1) }}">{{ cat_item }}</a>{% endfor %}
                    </div>
                </div>
                <div class="footer-section col-lg-4 col-md-6 mb-4">
                    <h5>Newsletter</h5>
                    <p class="small text-light">Subscribe for weekly updates!</p>
                    <form action="{{ url_for('subscribe') }}" method="POST" class="mt-3">
                        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                        <label for="footerNewsletterEmail" class="visually-hidden">Your email</label>
                        <div class="input-group">
                            <input type="email" id="footerNewsletterEmail" name="email" class="form-control form-control-sm footer-newsletter-input" placeholder="Your Email" aria-label="Your Email" required>
                            <button class="btn btn-sm btn-primary" type="submit">Subscribe</button>
                        </div>
                    </form>
                </div>
            </div>
            <div class="copyright">
                <span>&copy; {{ current_year }} BrieflyAI. All rights reserved. Made with <i class="fas fa-heart text-danger" aria-hidden="true"></i> in India.</span>
                <button type="button" class="back-to-top-link" id="backToTopBtn"><i class="fas fa-arrow-up" aria-hidden="true"></i> Back to top</button>
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
        var icons = { success: 'fa-circle-check', danger: 'fa-circle-exclamation', warning: 'fa-triangle-exclamation', info: 'fa-circle-info' };
        var wrap = document.createElement('div');
        wrap.className = 'alert alert-' + type + ' alert-dismissible fade show alert-top';
        wrap.setAttribute('role', 'alert');
        var icon = document.createElement('i');
        icon.className = 'fas ' + (icons[type] || icons.info) + ' alert-icon';
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

    /* Third-party tags (ads, analytics) are deferred until the page is idle so they
       never compete with content for bandwidth or main-thread time during load. */
    BrieflyAI.loadDeferredScripts = function () {
        if (BrieflyAI._deferredLoaded) { return; }
        BrieflyAI._deferredLoaded = true;
        window.dataLayer = window.dataLayer || [];
        window.gtag = function () { window.dataLayer.push(arguments); };
        window.gtag('js', new Date());
        window.gtag('config', 'G-CV5LWJ7NQ7');
        [
            'https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-6975904325280886',
            'https://www.googletagmanager.com/gtag/js?id=G-CV5LWJ7NQ7'
        ].forEach(function (src) {
            var s = document.createElement('script');
            s.src = src; s.async = true; s.crossOrigin = 'anonymous';
            document.head.appendChild(s);
        });
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
                    if (themeIcon) { themeIcon.className = isDarkMode ? 'fas fa-sun fa-fw me-2' : 'fas fa-moon fa-fw me-2'; }
                    if (themeText) { themeText.textContent = isDarkMode ? 'Light Mode' : 'Dark Mode'; }
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
                        if (icon) { icon.className = 'fas fa-circle-notch fa-spin search-icon'; }
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
                        <button type="submit" class="btn btn-sm btn-primary">Save Changes</button>
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
                <button type="button" class="react-btn" data-comment-id="{{ comment.id }}" title="React" aria-haspopup="true" aria-expanded="false" aria-controls="reaction-box-{{ comment.id }}"><i class="far fa-smile" aria-hidden="true"></i> React</button>
                <button type="button" class="reply-btn" data-comment-id="{{ comment.id }}" title="Reply"><i class="fas fa-reply" aria-hidden="true"></i> Reply</button>

                {% if session.user_id == comment.user_id %}
                    <button type="button" class="edit-btn" data-comment-id="{{ comment.id }}" title="Edit"><i class="fas fa-pencil-alt" aria-hidden="true"></i> Edit</button>
                    <button type="button" class="delete-btn" data-comment-id="{{ comment.id }}" title="Delete"><i class="fas fa-trash-alt" aria-hidden="true"></i> Delete</button>
                {% endif %}
            </div>
            <div class="reply-form-container" id="reply-form-container-{{ comment.id }}">
                <form class="reply-form">
                    <input type="hidden" name="parent_id" value="{{ comment.id }}">
                    <label class="visually-hidden" for="reply-content-{{ comment.id }}">Write a reply to {{ comment.author.name if comment.author else 'this comment' }}</label>
                    <div class="mb-2"><textarea class="form-control form-control-sm" id="reply-content-{{ comment.id }}" name="content" rows="2" placeholder="Write a reply..." required></textarea></div>
                    <div class="d-flex justify-content-end gap-2">
                        <button type="button" class="btn btn-sm btn-outline-secondary cancel-reply-btn">Cancel</button>
                        <button type="submit" class="btn btn-sm btn-primary">Post Reply</button>
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
        <div class="ai-synthesis-card">
            <div class="synthesis-header">
                <i class="fas fa-brain" aria-hidden="true"></i>
                <h2>Today's Briefing: The Big Picture</h2>
            </div>
            <p class="synthesis-text">&ldquo;{{ synthesis }}&rdquo;</p>
            {% if keywords %}
            <div class="synthesis-keywords">
                {% for keyword in keywords %}
                    <a href="{{ url_for('search_results', query=keyword) }}" class="keyword-tag">{{ keyword }}</a>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        {% endif %}

        {% if featured_article %}
        <article class="featured-story">
            <div class="featured-story-image" style="background-image: url('{{ featured_article.urlToImage }}')" role="img" aria-label="{{ featured_article.title|truncate(80) }}"></div>
            <div class="featured-story-content">
                <div class="article-meta">
                    <span class="meta-item"><i class="fas fa-fire-alt" aria-hidden="true"></i> Top Story</span>
                    <span class="meta-item"><i class="fas fa-building" aria-hidden="true"></i> {{ featured_article.source.name|truncate(20) }}</span>
                </div>
                <h2><a href="{{ url_for('article_detail', article_hash_id=featured_article.id) }}">{{ featured_article.title }}</a></h2>
                <p class="description">{{ featured_article.description|truncate(150) }}</p>
                <a href="{{ url_for('article_detail', article_hash_id=featured_article.id) }}" class="read-more-btn">Read Full Story <i class="fas fa-arrow-right ms-1" aria-hidden="true"></i></a>
            </div>
        </article>
        {% endif %}

        <section class="recent-strip" id="trendingStrip" hidden aria-labelledby="trendingHeading">
            <div class="recent-strip__head">
                <h2 class="section-heading h5 mb-0" id="trendingHeading"><i class="fas fa-arrow-trend-up me-2" aria-hidden="true"></i>Trending this week</h2>
            </div>
            <div class="recent-strip__list" id="trendingList"></div>
        </section>

        <section class="recent-strip" id="recentStrip" hidden aria-labelledby="recentStripHeading">
            <div class="recent-strip__head">
                <h2 class="section-heading h5 mb-0" id="recentStripHeading"><i class="fas fa-clock-rotate-left me-2" aria-hidden="true"></i>Pick up where you left off</h2>
                <button type="button" class="link-btn" id="clearRecentBtn">Clear</button>
            </div>
            <div class="recent-strip__list" id="recentStripList"></div>
        </section>

        <ul class="nav nav-tabs nav-fill mb-3" id="newsTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="popular-tab" data-bs-toggle="tab" data-bs-target="#popular-tab-pane" type="button" role="tab" aria-controls="popular-tab-pane" aria-selected="true">
                    <i class="fas fa-fire-alt me-1" aria-hidden="true"></i> POPULAR STORIES
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="yesterday-tab" data-bs-toggle="tab" data-bs-target="#yesterday-tab-pane" type="button" role="tab" aria-controls="yesterday-tab-pane" aria-selected="false">
                    <i class="fas fa-history me-1" aria-hidden="true"></i> YESTERDAY'S HEADLINES
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
                                        <div class="img-fallback-icon" aria-hidden="true"><i class="fas fa-newspaper"></i></div>
                                    </div>
                                    <div class="article-body d-flex flex-column">
                                        <div class="d-flex justify-content-between align-items-start">
                                            <h3 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                                            {% if session.user_id %}<button type="button" class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" aria-label="{% if art.is_bookmarked %}Remove bookmark{% else %}Add bookmark{% endif %} for {{ art.title|truncate(50) }}" data-article-hash-id="{{ art.id }}" data-is-community="false" data-title="{{ art.title|e }}" data-source-name="{{ art.source.name|e }}" data-image-url="{{ art.urlToImage|e }}" data-description="{{ (art.description if art.description else '')|e }}" data-published-at="{{ (art.publishedAt if art.publishedAt else '')|e }}"><i class="fa-solid fa-bookmark" aria-hidden="true"></i></button>{% endif %}
                                        </div>
                                        <div class="article-meta small mb-2">
                                            <span class="meta-item text-muted"><i class="fas fa-building" aria-hidden="true"></i> {{ art.source.name|truncate(20) }}</span>
                                            <span class="meta-item text-muted"><i class="far fa-calendar-alt" aria-hidden="true"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                                        </div>
                                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small" aria-hidden="true"></i></a>
                                    </div>
                                </article>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="col-12">
                            <div class="state-card state-card-solid py-4">
                                <div class="state-card-icon state-card-icon-sm"><i class="fas fa-satellite-dish" aria-hidden="true"></i></div>
                                <p class="state-card-text mb-0">More popular stories are on their way &mdash; please check back shortly.</p>
                            </div>
                        </div>
                    {% endif %}
                </div>
                {% if popular_articles %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('index', category_name='Popular Stories') }}" class="btn btn-outline-primary">View All Popular Stories <i class="fas fa-arrow-right ms-1" aria-hidden="true"></i></a>
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
                                        <div class="img-fallback-icon" aria-hidden="true"><i class="fas fa-newspaper"></i></div>
                                    </div>
                                    <div class="article-body d-flex flex-column">
                                        <div class="d-flex justify-content-between align-items-start">
                                            <h3 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                                            {% if session.user_id %}<button type="button" class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" aria-label="{% if art.is_bookmarked %}Remove bookmark{% else %}Add bookmark{% endif %} for {{ art.title|truncate(50) }}" data-article-hash-id="{{ art.id }}" data-is-community="false" data-title="{{ art.title|e }}" data-source-name="{{ art.source.name|e }}" data-image-url="{{ art.urlToImage|e }}" data-description="{{ (art.description if art.description else '')|e }}" data-published-at="{{ (art.publishedAt if art.publishedAt else '')|e }}"><i class="fa-solid fa-bookmark" aria-hidden="true"></i></button>{% endif %}
                                        </div>
                                        <div class="article-meta small mb-2">
                                            <span class="meta-item text-muted"><i class="fas fa-building" aria-hidden="true"></i> {{ art.source.name|truncate(20) }}</span>
                                            <span class="meta-item text-muted"><i class="far fa-calendar-alt" aria-hidden="true"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                                        </div>
                                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small" aria-hidden="true"></i></a>
                                    </div>
                                </article>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="col-12">
                            <div class="state-card state-card-solid py-4">
                                <div class="state-card-icon state-card-icon-sm"><i class="fas fa-satellite-dish" aria-hidden="true"></i></div>
                                <p class="state-card-text mb-0">Yesterday's headlines couldn't be loaded right now &mdash; please check back shortly.</p>
                            </div>
                        </div>
                    {% endif %}
                </div>
                {% if latest_yesterday_articles %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('index', category_name="Yesterday's Headlines") }}" class="btn btn-outline-primary">View All of Yesterday's Headlines <i class="fas fa-arrow-right ms-1" aria-hidden="true"></i></a>
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
            <span class="eyebrow"><i class="far fa-calendar-alt me-1" aria-hidden="true"></i> Filtered by date</span>
            <h1>{{ current_filter_date }}</h1>
        </div>
    {% elif selected_category == 'Community Hub' %}
        <div class="community-hub-header">
            <div class="community-hub-header-text">
                <span class="eyebrow"><i class="fas fa-users me-1" aria-hidden="true"></i> Community Hub</span>
                <h1>Stories from our readers</h1>
                <p class="community-hub-sub">Real perspectives, posted by the BrieflyAI community. Have something to say about today's news?</p>
            </div>
            {% if session.user_id %}
                <button type="button" class="btn btn-primary-modal community-hub-cta" data-bs-toggle="modal" data-bs-target="#addArticleModal"><i class="fas fa-pen-to-square me-2" aria-hidden="true"></i>Share Your Story</button>
            {% else %}
                <a href="{{ url_for('login', next=request.full_path) }}" class="btn btn-outline-primary community-hub-cta"><i class="fas fa-sign-in-alt me-2" aria-hidden="true"></i>Log In to Post</a>
            {% endif %}
        </div>
    {% elif query %}
        <div class="list-page-header">
            <span class="eyebrow"><i class="fas fa-magnifying-glass me-1" aria-hidden="true"></i> Search results</span>
            <h1>&ldquo;{{ query|truncate(40) }}&rdquo;</h1>
        </div>
    {% elif selected_category != 'All Articles' %}
        <div class="list-page-header">
            <span class="eyebrow"><i class="fas fa-{% if selected_category == 'Popular Stories' %}fire-alt{% elif selected_category == "Yesterday's Headlines" %}history{% else %}layer-group{% endif %} me-1" aria-hidden="true"></i> Category</span>
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
                        <div class="img-fallback-icon" aria-hidden="true"><i class="fas fa-newspaper"></i></div>
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
                                    data-image-url="{{ (art.image_url if art.is_community_article else art.urlToImage)|e }}"
                                    data-description="{{ (art.description if art.description else '')|e }}"
                                    data-published-at="{{ (art.published_at.isoformat() if art.is_community_article and art.published_at else (art.publishedAt if not art.is_community_article and art.publishedAt else ''))|e }}">
                                <i class="fa-solid fa-bookmark" aria-hidden="true"></i>
                            </button>
                            {% endif %}
                        </div>
                        <div class="article-meta small mb-2">
                            <span class="meta-item text-muted"><i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}" aria-hidden="true"></i> {% if art.is_community_article and art.author %}<a href="{{ url_for('public_profile', username=art.author.username) }}" class="text-muted text-decoration-none">{{ art.author.name|truncate(20) }}</a>{% else %}{{ art.source.name|truncate(20) }}{% endif %}</span>
                            <span class="meta-item text-muted"><i class="far fa-calendar-alt" aria-hidden="true"></i> {{ (art.published_at | to_ist if art.is_community_article else (art.publishedAt | to_ist if art.publishedAt else 'N/A')) }}</span>
                        </div>
                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small" aria-hidden="true"></i></a>
                    </div>
                </article>
            </div>
            {% endfor %}
        </div>
    {% elif not articles %}
        {% if query %}
        <div class="state-card">
            <div class="state-card-icon"><i class="fas fa-magnifying-glass" aria-hidden="true"></i></div>
            <h2 class="state-card-title">No results for &ldquo;{{ query|truncate(40) }}&rdquo;</h2>
            <p class="state-card-text">Try a different search term, or browse by category instead.</p>
            <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Browse All Articles</a></div>
        </div>
        {% else %}
        <div class="state-card">
            <div class="state-card-icon"><i class="fas fa-newspaper" aria-hidden="true"></i></div>
            <h2 class="state-card-title">No articles here yet</h2>
            <p class="state-card-text">Try a different category, or check back again soon.</p>
            <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Back to Homepage</a></div>
        </div>
        {% endif %}
    {% endif %}

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
        <div class="state-card-icon"><i class="fas fa-newspaper" aria-hidden="true"></i></div>
        <h1 class="state-card-title">Article Not Found</h1>
        <p class="state-card-text">The article you're looking for may have been removed, or the link may be incorrect.</p>
        <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Go to Homepage</a></div>
    </div>
{% else %}
<div class="reading-progress" aria-hidden="true"><div class="reading-progress__fill" id="readingProgressFill"></div></div>
<article class="article-full-content-wrapper animate-fade-in" id="articleWrapper" data-text-size="normal">
    <div class="mb-3 d-flex justify-content-between align-items-center flex-wrap gap-2">
        <a href="{{ previous_list_page }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-arrow-left me-2" aria-hidden="true"></i>Back to List</a>

        <div class="d-flex align-items-center gap-2">

            {% if session.get('is_admin') and is_community_article %}
            <button type="button" id="adminDeleteBtn" class="btn btn-sm btn-danger" title="Admin: Delete Post">
                <i class="fas fa-trash-alt" aria-hidden="true"></i> Delete Post
            </button>
            {% endif %}

            {% if session.user_id %}
                {% if is_community_article %}
                <button type="button" id="reportBtn" class="btn btn-sm btn-outline-warning" title="Report this article for review">
                    <i class="fas fa-flag" aria-hidden="true"></i> Report
                </button>
                {% endif %}

                <button type="button" id="bookmarkBtn" class="bookmark-btn {% if is_bookmarked %}active{% endif %}" title="{% if is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" aria-label="{% if is_bookmarked %}Remove bookmark{% else %}Add bookmark{% endif %} for this article" data-article-hash-id="{{ article.article_hash_id if is_community_article else article.id }}" data-is-community="{{ 'true' if is_community_article else 'false' }}" data-title="{{ article.title|e }}" data-source-name="{{ (article.author.name if is_community_article and article.author else article.source.name)|e }}" data-image-url="{{ (article.image_url if is_community_article else article.urlToImage)|e }}" data-description="{{ (article.description if article.description else '')|e }}" data-published-at="{{ (article.published_at.isoformat() if is_community_article and article.published_at else (article.publishedAt if not is_community_article and article.publishedAt else ''))|e }}"><i class="fa-solid fa-bookmark" aria-hidden="true"></i></button>
            {% endif %}
        </div>
    </div>

    <h1 class="mb-2 article-title-main">{{ article.title }}</h1>
    <div class="article-meta-detailed d-flex align-items-center flex-wrap gap-2 text-muted small">
        <span class="meta-item" title="Source"><i class="fas fa-{{ 'user-edit' if is_community_article else 'building' }}" aria-hidden="true"></i> {{ article.author.name if is_community_article and article.author else article.source.name }}</span>
        <span class="meta-item" title="Published Date"><i class="far fa-calendar-alt" aria-hidden="true"></i> {{ (article.published_at | to_ist if is_community_article else (article.publishedAt | to_ist if article.publishedAt else 'N/A')) }}</span>
    </div>
    {% set image_to_display = article.image_url if is_community_article else article.urlToImage %}
    {% if image_to_display %}
    <div class="hero-image-wrap">
        <img src="{{ image_to_display }}" alt="{{ article.title|truncate(50) }}" class="hero-image" loading="eager" decoding="async">
        <div class="img-fallback-icon" aria-hidden="true"><i class="fas fa-newspaper"></i></div>
    </div>
    {% endif %}

    <div class="article-toolbar" role="toolbar" aria-label="Article tools">
        <span class="read-time-badge" id="readTimeBadge" {% if not is_community_article %}hidden{% endif %}>
            <i class="far fa-clock" aria-hidden="true"></i>
            <span id="readTimeValue">{% if is_community_article and article.full_text %}{{ [1, ((article.full_text|wordcount) / 220)|round(0, 'ceil')|int]|max }} min read{% endif %}</span>
        </span>

        <button type="button" class="toolbar-btn" id="listenBtn" hidden aria-pressed="false">
            <i class="fas fa-headphones" aria-hidden="true"></i> <span class="listen-label">Listen</span>
        </button>

        <button type="button" class="toolbar-btn" id="textSizeBtn" aria-label="Change text size">
            <i class="fas fa-font" aria-hidden="true"></i> <span id="textSizeLabel">Normal</span>
        </button>

        <div class="share-wrap">
            <button type="button" class="toolbar-btn" id="shareBtn" aria-haspopup="true" aria-expanded="false" aria-controls="shareMenu">
                <i class="fas fa-share-nodes" aria-hidden="true"></i> Share
            </button>
            <div class="share-menu" id="shareMenu" role="menu" aria-label="Share this article">
                <a href="#" data-share="whatsapp" role="menuitem" target="_blank" rel="noopener noreferrer"><i class="fab fa-whatsapp i-whatsapp" aria-hidden="true"></i> WhatsApp</a>
                <a href="#" data-share="x" role="menuitem" target="_blank" rel="noopener noreferrer"><i class="fab fa-x-twitter i-x" aria-hidden="true"></i> X (Twitter)</a>
                <a href="#" data-share="facebook" role="menuitem" target="_blank" rel="noopener noreferrer"><i class="fab fa-facebook i-facebook" aria-hidden="true"></i> Facebook</a>
                <a href="#" data-share="linkedin" role="menuitem" target="_blank" rel="noopener noreferrer"><i class="fab fa-linkedin i-linkedin" aria-hidden="true"></i> LinkedIn</a>
                <button type="button" data-share="copy" role="menuitem"><i class="fas fa-link i-link" aria-hidden="true"></i> Copy link</button>
            </div>
        </div>

        <button type="button" class="toolbar-btn" id="printBtn" aria-label="Print this article">
            <i class="fas fa-print" aria-hidden="true"></i>
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
        <p class="ai-skeleton-caption"><i class="fas fa-brain fa-fw" aria-hidden="true"></i> Generating AI summary&hellip;</p>
    </div>
    <div id="articleAnalysisContainer">
    {% if is_community_article %}
        {% if article.groq_summary %}<div class="summary-box my-3"><h2><i class="fas fa-book-open me-2" aria-hidden="true"></i>AI Summary</h2><p class="mb-0">{{ article.groq_summary|e|replace('\\n', '<br>')|safe }}</p></div>{% endif %}
        {% if article.parsed_takeaways %}<div class="takeaways-box my-3"><h2><i class="fas fa-list-check me-2" aria-hidden="true"></i>AI Key Takeaways</h2><ul>{% for takeaway in article.parsed_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul></div>{% endif %}
        <hr class="my-4"><h2 class="content-divider-heading">Full Article Content</h2><div class="content-text">{{ article.full_text }}</div>
    {% else %}<div id="apiArticleContent"></div>{% endif %}
    </div>

    <section class="comment-section mt-5" id="comment-section">
        <div class="comment-toolbar">
            <h2 class="mb-0">Community Discussion (<span id="comment-count">{{ total_comment_count }}</span>)</h2>
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
                <p id="no-comments-msg" class="text-muted mt-3"><i class="far fa-comment-dots me-2" aria-hidden="true"></i>No comments yet. Be the first to share your thoughts!</p>
            {% endfor %}
        </div>

        {% if session.user_id %}
            <div class="add-comment-form mt-4 pt-4 border-top">
                <h3 class="comment-form-heading">Leave a Comment</h3>
                <form id="comment-form">
                    <label class="visually-hidden" for="comment-content">Write a comment</label>
                    <div class="mb-3"><textarea class="form-control" id="comment-content" name="content" rows="4" placeholder="Share your insights..." required></textarea></div>
                    <button type="submit" class="btn btn-primary">Post Comment</button>
                </form>
            </div>
        {% else %}
            <div class="comment-login-prompt">
                <i class="fas fa-comments" aria-hidden="true"></i>
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
                btn.querySelector('i').className = on ? 'fas fa-stop' : 'fas fa-headphones';
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
                    confirmText: 'Delete Post',
                    danger: true
                });
                if (!confirmed) return;

                this.disabled = true;
                this.innerHTML = '<i class="fas fa-spinner fa-spin" aria-hidden="true"></i> Deleting...';

                BrieflyAI.postJSON(`/delete_community_article/${articleHashIdGlobal}`)
                .then(res => res.json().then(data => ({ ok: res.ok, data })))
                .then(({ ok, data }) => {
                    if (ok && data.success) {
                        window.location.href = data.redirect_url;
                    } else {
                        BrieflyAI.showToast('Deletion failed: ' + (data.error || 'Unknown error'), 'danger');
                        this.disabled = false;
                        this.innerHTML = '<i class="fas fa-trash-alt" aria-hidden="true"></i> Delete Post';
                    }
                })
                .catch(err => {
                    console.error("Admin delete error:", err);
                    BrieflyAI.showToast("A network error occurred. Could not delete the post.", 'danger');
                    this.disabled = false;
                    this.innerHTML = '<i class="fas fa-trash-alt" aria-hidden="true"></i> Delete Post';
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
                            if (analysis.groq_summary) { html += `<div class="summary-box my-3"><h2><i class="fas fa-book-open me-2" aria-hidden="true"></i>AI Summary</h2><p class="mb-0">${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`; }
                            if (analysis.groq_takeaways && analysis.groq_takeaways.length > 0) { html += `<div class="takeaways-box my-3"><h2><i class="fas fa-list-check me-2" aria-hidden="true"></i>AI Key Takeaways</h2><ul>${analysis.groq_takeaways.map(t => `<li>${String(t)}</li>`).join('')}</ul></div>`; }
                        }
                    }
                    if (articleUrl) { html += `<hr class="my-4"><a href="${articleUrl}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">Read Original Article at ${articleSourceName} <i class="fas fa-external-link-alt ms-1" aria-hidden="true"></i></a>`; }
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
                this.innerHTML = '<i class="fas fa-spinner fa-spin" aria-hidden="true"></i> Reporting...';
                BrieflyAI.postJSON(`/report_article/${articleHashIdGlobal}`)
                .then(res => res.json().then(data => ({ ok: res.ok, status: res.status, data })))
                .then(({ ok, status, data }) => {
                    if (ok) {
                        this.innerHTML = '<i class="fas fa-check" aria-hidden="true"></i> Reported';
                        BrieflyAI.showToast(data.message, 'success');
                    } else {
                        this.disabled = false;
                        this.innerHTML = '<i class="fas fa-flag" aria-hidden="true"></i> Report';
                        BrieflyAI.showToast('Error: ' + data.error, 'danger');
                    }
                })
                .catch(err => {
                    console.error("Report error:", err);
                    BrieflyAI.showToast("A network error occurred. Please try again.", 'danger');
                    this.disabled = false;
                    this.innerHTML = '<i class="fas fa-flag" aria-hidden="true"></i> Report';
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
        <div class="icon"><i class="fas fa-bolt-lightning" aria-hidden="true"></i></div>
        <h1>Welcome Back to BrieflyAI</h1>
    </div>
    <div class="auth-body">
        <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}" id="loginForm">
            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
            <div class="mb-3">
                <label for="username" class="form-label fw-medium">Username</label>
                <div class="input-group-icon">
                    <i class="fas fa-user input-icon" aria-hidden="true"></i>
                    <input type="text" class="form-control" id="username" name="username" required placeholder="e.g. user123" autocomplete="username">
                </div>
            </div>
            <div class="mb-4">
                <label for="password" class="form-label fw-medium">Password</label>
                <div class="input-group-icon">
                    <i class="fas fa-lock input-icon" aria-hidden="true"></i>
                    <input type="password" class="form-control" id="password" name="password" required placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;" autocomplete="current-password">
                </div>
            </div>
            <button type="submit" class="btn btn-primary w-100" id="loginSubmitBtn">Sign In</button>
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
        <div class="icon"><i class="fas fa-user-plus" aria-hidden="true"></i></div>
        <h1>Create Your Account</h1>
    </div>
    <div class="auth-body">
        <form method="POST" action="{{ url_for('register') }}" id="registerForm">
            <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
             <div class="mb-3">
                <label for="name" class="form-label fw-medium">Full Name</label>
                <div class="input-group-icon">
                    <i class="fas fa-id-card input-icon" aria-hidden="true"></i>
                    <input type="text" class="form-control" id="name" name="name" required placeholder="e.g. John Doe" autocomplete="name">
                </div>
            </div>
            <div class="mb-3">
                <label for="username" class="form-label fw-medium">Username</label>
                <div class="input-group-icon">
                    <i class="fas fa-user input-icon" aria-hidden="true"></i>
                    <input type="text" class="form-control" id="username" name="username" required minlength="3" placeholder="e.g. johndoe (min 3 chars)" autocomplete="username">
                </div>
            </div>
            <div class="mb-4">
                <label for="password" class="form-label fw-medium">Password</label>
                <div class="input-group-icon">
                    <i class="fas fa-lock input-icon" aria-hidden="true"></i>
                    <input type="password" class="form-control" id="password" name="password" required minlength="6" placeholder="min 6 chars" autocomplete="new-password">
                </div>
            </div>
            <button type="submit" class="btn btn-primary w-100" id="registerSubmitBtn">Create Account</button>
        </form>
    </div>
    <div class="auth-footer">
        <p class="mb-0 small">
            Already have an account? <a href="{{ url_for('login') }}" class="fw-bold text-decoration-none">Sign In</a>
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
            <div class="icon"><i class="fas fa-pen-to-square" aria-hidden="true"></i></div>
            <div class="count">{{ posted_articles|length }}</div>
            <div class="label">Articles Posted</div>
        </button>
        <button type="button" class="stat-item" data-target-tab="bookmarks-tab" aria-label="View your bookmarks">
            <div class="icon"><i class="fas fa-bookmark" aria-hidden="true"></i></div>
            <div class="count">{{ bookmarks_pagination.total if bookmarks_pagination else 0 }}</div>
            <div class="label">Bookmarks</div>
        </button>
    </div>
</div>

<div class="mt-4 animate-fade-in" style="animation-delay: 0.1s;">
    <ul class="nav nav-tabs profile-tabs nav-fill mb-4" id="profileTab" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="bookmarks-tab" data-bs-toggle="tab" data-bs-target="#bookmarks-content" type="button" role="tab" aria-controls="bookmarks-content" aria-selected="true"><i class="fas fa-bookmark me-2" aria-hidden="true"></i>My Bookmarks</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="posted-tab" data-bs-toggle="tab" data-bs-target="#posted-content" type="button" role="tab" aria-controls="posted-content" aria-selected="false"><i class="fas fa-pen-to-square me-2" aria-hidden="true"></i>My Articles</button>
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
                            <div class="img-fallback-icon" aria-hidden="true"><i class="fas fa-newspaper"></i></div>
                            {% if art.is_stale_bookmark %}<span class="badge bg-secondary position-absolute top-0 end-0 m-2">Cached Bookmark</span>{% endif %}
                        </div>
                        <div class="article-body d-flex flex-column">
                            <h3 class="article-title mb-2"><a href="{{ art.article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                            <div class="article-meta small mb-2">
                                <span class="meta-item text-muted"><i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}" aria-hidden="true"></i> {{ art.source.name|truncate(20) }}</span>
                                <span class="meta-item text-muted"><i class="far fa-calendar-alt" aria-hidden="true"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                            </div>
                            <p class="article-description small">{{ art.description|truncate(100) }}</p>
                            <a href="{{ art.article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small" aria-hidden="true"></i></a>
                        </div>
                    </article>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <div class="state-card">
                    <div class="state-card-icon"><i class="fas fa-bookmark" aria-hidden="true"></i></div>
                    <h2 class="state-card-title">No Bookmarks Yet</h2>
                    <p class="state-card-text">Find an article you like and tap the bookmark icon to save it here.</p>
                    <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Browse Articles</a></div>
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
                            <div class="img-fallback-icon" aria-hidden="true"><i class="fas fa-newspaper"></i></div>
                        </div>
                        <div class="article-body d-flex flex-column">
                            <h3 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                            <div class="article-meta small mb-2">
                                <span class="meta-item text-muted"><i class="fas fa-user-edit" aria-hidden="true"></i> {{ art.author.name|truncate(20) }}</span>
                                <span class="meta-item text-muted"><i class="far fa-calendar-alt" aria-hidden="true"></i> {{ art.published_at | to_ist }}</span>
                            </div>
                            <p class="article-description small">{{ art.description|truncate(100) }}</p>
                            <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small" aria-hidden="true"></i></a>
                        </div>
                    </article>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <div class="state-card">
                    <div class="state-card-icon"><i class="fas fa-pen-to-square" aria-hidden="true"></i></div>
                    <h2 class="state-card-title">Nothing Posted Yet</h2>
                    <p class="state-card-text">Share your first story with the BrieflyAI community &mdash; it only takes a minute.</p>
                    <div class="state-card-actions"><button type="button" class="btn btn-primary-modal" data-bs-toggle="modal" data-bs-target="#addArticleModal"><i class="fas fa-pen-to-square me-2" aria-hidden="true"></i>Write a Post</button></div>
                </div>
            {% endif %}
        </div>
    </div>
</div>
<section class="mt-5 pt-4 border-top" aria-labelledby="accountHeading">
    <h2 class="section-heading h5 mb-3" id="accountHeading"><i class="fas fa-user-gear me-2" aria-hidden="true"></i>Your account &amp; data</h2>
    <div class="row g-3">
        <div class="col-md-6 d-flex">
            <div class="contact-card text-start w-100">
                <h3 class="h6 mb-2"><i class="fas fa-download me-2" aria-hidden="true"></i>Download your data</h3>
                <p class="small text-muted mb-3">Get a JSON copy of your profile, articles, comments and bookmarks.</p>
                <a href="{{ url_for('export_my_data') }}" class="btn btn-sm btn-outline-primary">Export my data</a>
            </div>
        </div>
        <div class="col-md-6 d-flex">
            <div class="contact-card text-start w-100">
                <h3 class="h6 mb-2"><i class="fas fa-triangle-exclamation me-2" aria-hidden="true"></i>Delete your account</h3>
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
                        <i class="fas fa-lock input-icon" aria-hidden="true"></i>
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
                    <div class="img-fallback-icon" aria-hidden="true"><i class="fas fa-newspaper"></i></div>
                </div>
                <div class="article-body d-flex flex-column">
                    <h3 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h3>
                    <div class="article-meta small mb-2">
                        <span class="meta-item text-muted"><i class="far fa-calendar-alt" aria-hidden="true"></i> {{ art.published_at | to_ist }}</span>
                    </div>
                    <p class="article-description small">{{ art.description|truncate(100) }}</p>
                    <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small" aria-hidden="true"></i></a>
                </div>
            </article>
        </div>
        {% endfor %}
    {% else %}
        <div class="col-12">
            <div class="state-card">
                <div class="state-card-icon"><i class="fas fa-pen-to-square" aria-hidden="true"></i></div>
                <h2 class="state-card-title">No Articles Yet</h2>
                <p class="state-card-text">{{ user.name }} hasn't posted any articles yet. Check back soon.</p>
                <div class="state-card-actions"><a href="{{ url_for('index') }}" class="btn btn-primary-modal">Explore Other Articles</a></div>
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

        <h2><i class="icon fas fa-bullseye" aria-hidden="true"></i>Our Mission</h2>
        <p>
            In a world of information overload, our mission is to provide clarity and efficiency. We believe that everyone deserves access to accurate, unbiased news without spending hours sifting through lengthy articles. BrieflyAI cuts through the noise, offering insightful summaries that matter.
        </p>

        <h2><i class="icon fas fa-users" aria-hidden="true"></i>Community Hub</h2>
        <p>
            Beyond AI-driven news, BrieflyAI is a platform for discussion and community engagement. Our Community Hub allows users to post their own articles, share perspectives, and engage in meaningful conversations about the topics that shape our world. We are committed to fostering a respectful and intelligent environment for all our members.
        </p>

        <h2><i class="icon fas fa-microchip" aria-hidden="true"></i>Our Technology</h2>
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
        <h1>Get In Touch</h1>
    </div>
    <div class="static-content-container">
        <p class="lead text-center mb-5">
            We'd love to hear from you! Whether you have a question, feedback, or a news tip, feel free to reach out using one of the methods below.
        </p>
        <div class="row g-4">
            <div class="col-md-6">
                <div class="contact-card">
                    <div class="icon"><i class="fas fa-envelope" aria-hidden="true"></i></div>
                    <h2 class="h5">General Inquiries</h2>
                    <p class="text-muted">For general questions, feedback, or support, please email us at:</p>
                    <a href="mailto:vbansal639@gmail.com" class="fw-bold">vbansal639@gmail.com</a>
                </div>
            </div>
            <div class="col-md-6">
                <div class="contact-card">
                    <div class="icon"><i class="fas fa-handshake" aria-hidden="true"></i></div>
                    <h2 class="h5">Partnerships &amp; Media</h2>
                    <p class="text-muted">For partnership opportunities or media inquiries, please contact us at:</p>
                    <a href="mailto:vbansal639@gmail.com" class="fw-bold">vbansal639@gmail.com</a>
                </div>
            </div>
        </div>

        <div class="text-center mt-5">
            <h2 class="h3">Follow Us</h2>
            <p class="text-muted">Stay connected with us on social media.</p>
            <div class="contact-social-links mt-3">
                <a href="#" title="Twitter" aria-label="BrieflyAI on Twitter"><i class="fab fa-twitter" aria-hidden="true"></i></a>
                <a href="#" title="Facebook" aria-label="BrieflyAI on Facebook"><i class="fab fa-facebook-f" aria-hidden="true"></i></a>
                <a href="#" title="LinkedIn" aria-label="BrieflyAI on LinkedIn"><i class="fab fa-linkedin-in" aria-hidden="true"></i></a>
                <a href="#" title="Instagram" aria-label="BrieflyAI on Instagram"><i class="fab fa-instagram" aria-hidden="true"></i></a>
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
        <h1>Privacy Policy</h1>
    </div>
    <div class="static-content-container">
        <p class="text-muted">Last updated: June 10, 2025</p>
        <p>BrieflyAI ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you visit our website.</p>

        <h2><i class="icon fas fa-shield-halved" aria-hidden="true"></i>1. Information We Collect</h2>
        <p>We may collect personal information that you voluntarily provide to us when you register on the website, post articles or comments, bookmark articles, or subscribe to our newsletter. This information may include your name, username, email address, and your activities on our platform such as articles posted and bookmarked.</p>

        <h2><i class="icon fas fa-tasks" aria-hidden="true"></i>2. How We Use Your Information</h2>
        <p>We use the information we collect to:</p>
        <ul>
            <li>Create and manage your account.</li>
            <li>Operate and maintain the website, including your profile page.</li>
            <li>Display your posted and bookmarked articles as part of your profile.</li>
            <li>Send you newsletters or promotional materials, if you have opted in.</li>
            <li>Respond to your comments and inquiries.</li>
            <li>Improve our website and services.</li>
        </ul>

        <h2><i class="icon fas fa-share-nodes" aria-hidden="true"></i>3. Disclosure of Your Information</h2>
        <p>Your username and posted articles are publicly visible. Your bookmarked articles are visible on your profile page to you when logged in. We do not sell, trade, or otherwise transfer your personally identifiable information like your email address to outside parties without your consent, except to trusted third parties who assist us in operating our website, so long as those parties agree to keep this information confidential.</p>

        <h2><i class="icon fas fa-lock" aria-hidden="true"></i>4. Security of Your Information</h2>
        <p>We use administrative, technical, and physical security measures to help protect your personal information. While we have taken reasonable steps to secure the personal information you provide to us, please be aware that despite our efforts, no security measures are perfect or impenetrable.</p>

        <h2><i class="icon fas fa-edit" aria-hidden="true"></i>5. Your Choices</h2>
        <p>You can review and change your profile information by logging into your account. You may also request deletion of your account and associated data by contacting us.</p>

        <h2><i class="icon fas fa-sync-alt" aria-hidden="true"></i>6. Changes to This Privacy Policy</h2>
        <p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page. You are advised to review this Privacy Policy periodically for any changes.</p>
    </div>
</div>
{% endblock %}
"""
ERROR_404_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}404 Not Found - BrieflyAI{% endblock %}{% block content %}
<div class="state-card state-card-narrow animate-fade-in">
    <div class="state-card-icon"><i class="fas fa-map-signs" aria-hidden="true"></i></div>
    <h1 class="state-card-title">Page Not Found</h1>
    <p class="state-card-text">Sorry, the page you're looking for doesn't exist or may have been moved.</p>
    <div class="state-card-actions">
        <a href="{{ url_for('index') }}" class="btn btn-primary-modal"><i class="fas fa-house me-2" aria-hidden="true"></i>Go to Homepage</a>
    </div>
</div>
{% endblock %}"""

ERROR_500_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}500 Server Error - BrieflyAI{% endblock %}{% block content %}
<div class="state-card state-card-narrow state-card-danger animate-fade-in">
    <div class="state-card-icon"><i class="fas fa-server" aria-hidden="true"></i></div>
    <h1 class="state-card-title">Something Went Wrong</h1>
    <p class="state-card-text">We hit a snag on our end. We've been notified and are looking into it.</p>
    <div class="state-card-actions">
        <a href="{{ url_for('index') }}" class="btn btn-primary-modal"><i class="fas fa-house me-2" aria-hidden="true"></i>Go to Homepage</a>
        <button type="button" class="btn btn-outline-secondary" onclick="window.location.reload()"><i class="fas fa-rotate-right me-2" aria-hidden="true"></i>Try Again</button>
    </div>
</div>
{% endblock %}"""
OFFLINE_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}Offline - BrieflyAI{% endblock %}{% block content %}
<div class="state-card state-card-narrow animate-fade-in">
    <div class="state-card-icon"><i class="fas fa-wifi" aria-hidden="true"></i></div>
    <h1 class="state-card-title">You're Offline</h1>
    <p class="state-card-text">We couldn't reach the network. Any pages you've already opened are still available, and new stories will load as soon as you're back online.</p>
    <div class="state-card-actions">
        <button type="button" class="btn btn-primary-modal" onclick="window.location.reload()"><i class="fas fa-rotate-right me-2" aria-hidden="true"></i>Try Again</button>
        <a href="{{ url_for('index') }}" class="btn btn-outline-secondary"><i class="fas fa-house me-2" aria-hidden="true"></i>Homepage</a>
    </div>
</div>
{% endblock %}"""

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
