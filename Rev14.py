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
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Response

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

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'YOUR_FALLBACK_FLASK_SECRET_KEY_HERE_32_CHARS')
app.config['PER_PAGE'] = 9
app.config['CATEGORIES'] = ['All Articles', 'Popular Stories', "Yesterday's Headlines", 'Community Hub']

app.config['NEWS_API_QUERY'] = 'India OR "Indian politics" OR "Indian economy" OR "Bollywood"'
app.config['NEWS_API_DOMAINS'] = 'timesofindia.indiatimes.com,thehindu.com,ndtv.com,indianexpress.com,hindustantimes.com'
app.config['NEWS_API_DAYS_AGO'] = 7 # Fetch news from the last 7 days
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'publishedAt' #relevance, popularity, publishedAt
app.config['CACHE_EXPIRY_SECONDS'] = 1800 
app.permanent_session_lifetime = timedelta(days=30)

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
db = SQLAlchemy(app) 
app.logger.info(f"SQLAlchemy instance created.")
app.logger.info(f"--- End of Database Configuration ---")

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
    published_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    groq_summary = db.Column(db.Text, nullable=True)
    groq_takeaways = db.Column(db.Text, nullable=True) # Stored as JSON string
    comments = db.relationship('Comment', backref=db.backref('community_article', lazy='joined'), lazy='dynamic', foreign_keys='Comment.community_article_id', cascade="all, delete-orphan")

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    community_article_id = db.Column(db.Integer, db.ForeignKey('community_article.id'), nullable=True)
    api_article_hash_id = db.Column(db.String(32), nullable=True, index=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('comment.id'), nullable=True)
    replies = db.relationship('Comment', backref=db.backref('parent', remote_side=[id]), lazy='selectin', cascade="all, delete-orphan")
    votes = db.relationship('CommentVote', backref='comment', lazy='dynamic', cascade="all, delete-orphan")


class CommentVote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    comment_id = db.Column(db.Integer, db.ForeignKey('comment.id', ondelete="CASCADE"), nullable=False)
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
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    article_hash_id = db.Column(db.String(32), nullable=False, index=True)
    is_community_article = db.Column(db.Boolean, default=False, nullable=False)
    title_cache = db.Column(db.String(250), nullable=True)
    source_name_cache = db.Column(db.String(100), nullable=True)
    image_url_cache = db.Column(db.String(500), nullable=True)
    description_cache = db.Column(db.Text, nullable=True)
    published_at_cache = db.Column(db.DateTime, nullable=True) # Store as datetime for API articles
    bookmarked_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (db.UniqueConstraint('user_id', 'article_hash_id', name='_user_article_bookmark_uc'),)

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
MASTER_ARTICLE_STORE, API_CACHE = {}, {}
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
            API_CACHE[cache_key] = (result, time.time())
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
        "1. Provide a concise, neutral summary (3-4 paragraphs). "
        "2. List 5-7 key takeaways as bullet points. Each takeaway must be a complete sentence. "
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
        "1. For 'synthesis_text': Write a single, insightful, and cohesive paragraph (3-4 sentences) that synthesizes the most important themes, trends, or events of the day. Do not just list the news. Connect the ideas. "
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
        placeholder_text = urllib.parse.quote_plus(source_name[:20])
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
            'url': url, 'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
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
                placeholder_text = urllib.parse.quote_plus(source_name[:20])
                
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
                    'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
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
                placeholder_text = urllib.parse.quote_plus(source_name[:20])
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
                    'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
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
        response = requests.get('http://api.scraperapi.com', params=params, timeout=45)
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
            # If not in MASTER_ARTICLE_STORE or incomplete, generate it
            groq_analysis_result = get_article_analysis_with_groq(article_scraper.text, article_title_for_groq)
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

def get_paginated_articles(articles, page, per_page):
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
    flash("DEBUG: The correct search_results function was called!", "success")

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
                    placeholder_text = urllib.parse.quote_plus(source_name[:20])
                    
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
                        'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
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
    community_db_articles = []
    community_db_articles_query = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter(
        db.or_(CommunityArticle.title.ilike(f'%{query_str}%'), CommunityArticle.description.ilike(f'%{query_str}%'))
    ).order_by(CommunityArticle.published_at.desc())
    
    for art in community_db_articles_query.all():
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
def add_comment(article_hash_id):
    content = request.json.get('content', '').strip()
    parent_id = request.json.get('parent_id')
    if not content:
        return jsonify({"success": False, "error": "Comment cannot be empty."}), 400
    
    user = User.query.get(session['user_id'])
    if not user:
        # This is a fallback, the decorator should handle it.
        return jsonify({"success": False, "error": "User not found."}), 401

    # --- ROBUST LOGIC ---
    # The new logic is simpler and more reliable.
    # It trusts the article_hash_id from the page the user is on.
    
    new_comment = Comment(content=content, user_id=user.id, parent_id=parent_id)
    
    # First, check if it's a permanent community article from our database.
    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    
    if community_article:
        # If it is, link the comment to it.
        new_comment.community_article_id = community_article.id
    else:
        # If not, assume it's an API article and save the hash ID.
        # We no longer check against the volatile MASTER_ARTICLE_STORE cache.
        new_comment.api_article_hash_id = article_hash_id

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
    emoji = request.json.get('emoji')
    
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

    new_content = request.json.get('content', '').strip()
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
def post_article():
    title, description, content, source_name, image_url = map(lambda x: request.form.get(x, '').strip(), ['title', 'description', 'content', 'sourceName', 'imageUrl'])
    source_name = source_name or 'Community Post'
    if not all([title, description, content, source_name]):
        flash("Title, Description, Full Content, and Source Name are required.", "danger")
        return redirect(request.referrer or url_for('index'))
    article_hash_id = generate_article_id(title + str(session['user_id']) + str(time.time()))
    groq_analysis_result = get_article_analysis_with_groq(content, title)
    groq_summary_text, groq_takeaways_json_str = None, None
    if groq_analysis_result and not groq_analysis_result.get("error"):
        groq_summary_text = groq_analysis_result.get('groq_summary')
        takeaways_list = groq_analysis_result.get('groq_takeaways')
        if takeaways_list and isinstance(takeaways_list, list): groq_takeaways_json_str = json.dumps(takeaways_list)
    new_article = CommunityArticle(article_hash_id=article_hash_id, title=title, description=description, full_text=content, source_name=source_name, image_url=image_url or f'https://via.placeholder.com/400x220/1E3A5E/FFFFFF?text={urllib.parse.quote_plus(title[:20])}', user_id=session['user_id'], published_at=datetime.now(timezone.utc), groq_summary=groq_summary_text, groq_takeaways=groq_takeaways_json_str)
    db.session.add(new_article); db.session.commit()
    flash("Your article has been posted!", "success")
    return redirect(url_for('article_detail', article_hash_id=new_article.article_hash_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name, username, password = request.form.get('name', '').strip(), request.form.get('username', '').strip().lower(), request.form.get('password', '')
        if not all([name, username, password]): flash('All fields are required.', 'danger')
        elif len(username) < 3: flash('Username must be at least 3 characters.', 'warning')
        elif len(password) < 6: flash('Password must be at least 6 characters.', 'warning')
        elif User.query.filter_by(username=username).first(): flash('Username already exists. Please choose another.', 'warning')
        else:
            new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
            db.session.add(new_user); db.session.commit()
            flash(f'Registration successful, {name}! Please log in.', 'success')
            return redirect(url_for('login'))
        return redirect(url_for('register'))
    return render_template("REGISTER_HTML_TEMPLATE")

@app.route('/delete_community_article/<article_hash_id>', methods=['POST'])
@login_required
def delete_community_article(article_hash_id):
    # Security Check: Ensure the user has the admin flag from the session.
    if not session.get('is_admin') or session.get('username') != 'vbdevil':
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
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username, password = request.form.get('username', '').strip().lower(), request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.permanent = True
            session['user_id'] = user.id
            session['user_name'] = user.name
            # Store username for easy access in templates/routes
            session['username'] = user.username
            
            # Check if the logged-in user is the designated admin
            if user.username == "vbdevil":
                session['is_admin'] = True
            else:
                session['is_admin'] = False

            flash(f"Welcome back, {user.name}!", "success")
            next_url = request.args.get('next')
            session.pop('previous_list_page', None) 
            return redirect(next_url or url_for('index'))
        else: flash('Invalid username or password.', 'danger')
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
def subscribe():
    email = request.form.get('email', '').strip().lower()
    if not email: flash('Email is required to subscribe.', 'warning')
    elif Subscriber.query.filter_by(email=email).first(): flash('You are already subscribed to our newsletter.', 'info')
    else:
        try: db.session.add(Subscriber(email=email)); db.session.commit(); flash('Thank you for subscribing!', 'success')
        except Exception as e: db.session.rollback(); app.logger.error(f"Error subscribing email {email}: {e}"); flash('Could not subscribe at this time. Please try again later.', 'danger')
    return redirect(request.referrer or url_for('index'))

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
    page = request.args.get('page', 1, type=int)
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
            else: article_detail_data = {'id': bookmark.article_hash_id, 'title': bookmark.title_cache or "Bookmarked Article (Details N/A)", 'description': bookmark.description_cache or "Description not available.", 'urlToImage': bookmark.image_url_cache or f'https://via.placeholder.com/400x220/CCCCCC/000000?text=Preview+N/A', 'publishedAt': bookmark.published_at_cache.isoformat() if bookmark.published_at_cache else None, 'source': {'name': bookmark.source_name_cache or "Unknown Source"}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=bookmark.article_hash_id), 'is_stale_bookmark': True}
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
# --- 7. HTML Templates (Stored in memory) ---
# ==============================================================================
BASE_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}BrieflyAI{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4F46E5; --primary-light: #6366F1; --primary-dark: #4338CA; --secondary-color: #14B8A6; --secondary-light: #2DD4BF; --accent-color: #F97316; --text-color: #1F2937; --text-muted-color: #6B7280; --light-bg: #F9FAFB; --card-bg: #FFFFFF; --card-border-color: #E5E7EB; --footer-bg: #111827; --footer-text: #D1D5DB; --footer-link-hover: var(--primary-light);
            --primary-color-rgb: 79, 70, 229; --secondary-color-rgb: 20, 184, 166; --text-muted-color-rgb: 107, 114, 128;
            --bookmark-active-color: var(--secondary-color);
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05); --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --border-radius-sm: 0.375rem; --border-radius-md: 0.5rem; --border-radius-lg: 0.75rem;
        }
        body { padding-top: 85px; font-family: 'Inter', sans-serif; line-height: 1.65; color: var(--text-color); background-color: var(--light-bg); display: flex; flex-direction: column; min-height: 100vh; transition: background-color 0.3s ease, color 0.3s ease; }
        .main-content { flex-grow: 1; }
        body.dark-mode {
            --primary-color: #6366F1; --primary-light: #818CF8; --primary-dark: #4F46E5; --secondary-color: #2DD4BF; --secondary-light: #5EEAD4; --accent-color: #FB923C; --text-color: #F9FAFB; --text-muted-color: #9CA3AF; --light-bg: #111827; --card-bg: #1F2937; --card-border-color: #374151; --footer-bg: #000000; --footer-text: #9CA3AF;
            --primary-color-rgb: 99, 102, 241; --secondary-color-rgb: 45, 212, 191; --text-muted-color-rgb: 156, 163, 175;
            --bookmark-active-color: var(--secondary-light);
        }
        h1, h2, h3, h4, h5, .auth-title, .profile-card h2, .article-title-main, .modal-title { font-family: 'Poppins', sans-serif; font-weight: 700; }
        .alert-top { position: fixed; top: 90px; left: 50%; transform: translateX(-50%); z-index: 2050; min-width:320px; text-align:center; box-shadow: var(--shadow-lg); border-radius: var(--border-radius-md); }
        .navbar-main { background-color: var(--primary-color); padding: 0.75rem 0; box-shadow: var(--shadow-md); transition: background-color 0.3s ease; z-index: 1040; position: fixed; top: 0; width: 100%;}
        .navbar-content-wrapper { display: flex; align-items: center; justify-content: space-between; gap: 1rem; width: 100%; }
        .navbar-left { flex-shrink: 0; }
        .navbar-center { flex-grow: 1; min-width: 150px; max-width: 550px; }
        .navbar-right { flex-shrink: 0; }
        .navbar-brand-custom { color: white !important; font-weight: 700; font-size: 2rem; font-family: 'Poppins', sans-serif; display: flex; align-items: center; gap: 10px; text-decoration: none !important; }
        .navbar-brand-custom .brand-icon { color: var(--secondary-light); font-size: 2.2rem; }
        .search-container { position: relative; width: 100%; }
        .navbar-search { width: 100%; border-radius: 50px; padding: 0.6rem 1.25rem 0.6rem 2.8rem; border: 1px solid transparent; font-size: 0.95rem; transition: all 0.3s ease; background: rgba(255,255,255,0.15); color: white; }
        .navbar-search::placeholder { color: rgba(255,255,255,0.7); }
        .navbar-search:focus { background: rgba(255,255,255,0.25); box-shadow: 0 0 0 4px rgba(255,255,255,0.2); border-color: var(--secondary-light); outline: none; color:white; }
        .search-icon { color: rgba(255,255,255,0.8); transition: all 0.3s ease; left: 1.1rem; position: absolute; top: 50%; transform: translateY(-50%); }
        .header-controls { display: flex; gap: 0.8rem; align-items: center; }
        .header-btn { background: transparent; border: 1px solid rgba(255,255,255,0.4); padding: 0.5rem 1rem; border-radius: 50px; color: white; font-weight: 500; transition: all 0.3s ease; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; text-decoration:none; font-size: 0.9rem; }
        .header-btn:hover { background: rgba(255,255,255,0.9); border-color: transparent; color: var(--primary-dark); }
       
        /* === UNIFIED OFFCANVAS SIDEBAR === */
        .offcanvas { background-color: #111827; color: var(--footer-text); z-index: 1045; }
        body.dark-mode .offcanvas { background-color: var(--footer-bg); }
        .offcanvas-header { border-bottom-color: #374151 !important; }
        .offcanvas-header .btn-close { filter: invert(1) grayscale(100%) brightness(200%); }
        .sidebar-section { margin-bottom: 1.5rem; }
        .sidebar-heading { font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted-color); margin-bottom: 0.75rem; font-weight: 600; }
        .sidebar-btn { display: flex; align-items: center; padding: 0.75rem 1rem; width: 100%; text-align: left; background-color: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: white; text-decoration: none; border-radius: var(--border-radius-md); transition: all 0.2s ease; }
        .sidebar-btn:hover { background-color: rgba(255,255,255,0.1); border-color: var(--primary-light); color: white; }
        .sidebar-avatar { width: 40px; height: 40px; border-radius: 50%; background-color: var(--primary-color); color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; }
        .offcanvas .dropdown-toggle { color: white; }
        .offcanvas .dropdown-toggle::after { color: white; }
        .offcanvas .dropdown-menu { background-color: #374151; border-color: #4B5563;}
        .offcanvas .dropdown-item { color: var(--footer-text); }
        .offcanvas .dropdown-item:hover { background-color: var(--primary-color); color: white; }
        .offcanvas .nav-link { padding: 0.6rem 1rem; font-weight: 500; border-radius: var(--border-radius-sm); transition: background-color 0.2s ease, color 0.2s ease; color: var(--footer-text) !important; }
        .offcanvas .nav-link.active { background-color: var(--primary-color); color: white !important; }
        .offcanvas .nav-link:not(.active):hover { background-color: rgba(255,255,255,0.1); color: white !important; }
        #dateFilterForm .form-control { background-color: #374151; border-color: #4B5563; color: white; }
        #dateFilterForm .form-control:focus { background-color: #374151; border-color: var(--primary-color); box-shadow: none; color: white; }
        #dateFilterForm .btn-primary { background-color: var(--primary-color); border-color: var(--primary-color); }
        
        .article-card, .article-full-content-wrapper, .auth-container, .profile-card { background: var(--card-bg); border-radius: var(--border-radius-lg); transition: all 0.3s ease; border: 1px solid var(--card-border-color); box-shadow: var(--shadow-md); }
        .article-card:hover { transform: translateY(-5px); box-shadow: var(--shadow-lg); }
        .article-image-container { height: 220px; overflow: hidden; position: relative; border-top-left-radius: var(--border-radius-lg); border-top-right-radius: var(--border-radius-lg);}
        .article-image { width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s ease; }
        .article-card:hover .article-image { transform: scale(1.05); }
        .article-body { padding: 1.5rem; flex-grow: 1; display: flex; flex-direction: column; }
        .article-title { font-weight: 600; line-height: 1.4; margin-bottom: 0.6rem; font-size:1.15rem; }
        .article-title a { color: var(--text-color); text-decoration: none; transition: color 0.2s ease; }
        .article-card:hover .article-title a { color: var(--primary-color) !important; }
        .article-meta { display: flex; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.4rem 1rem; }
        .meta-item { display: flex; align-items: center; font-size: 0.8rem; color: var(--text-muted-color); }
        .meta-item i { font-size: 0.9rem; margin-right: 0.4rem; color: var(--secondary-color); }
        .article-description { color: var(--text-muted-color); margin-bottom: 1.25rem; font-size: 0.95rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .read-more { margin-top: auto; background: var(--primary-color); color: white !important; border: none; padding: 0.6rem 0; border-radius: var(--border-radius-md); font-weight: 600; font-size: 0.9rem; transition: all 0.3s ease; width: 100%; text-align: center; text-decoration: none; display:inline-block; }
        .read-more:hover { background: var(--primary-dark); transform: translateY(-2px); color: white !important; box-shadow: var(--shadow-md); }
        .pagination { flex-wrap: wrap; }
        .page-item .page-link { border-radius: 50%; width: 40px; height: 40px; display:flex; align-items:center; justify-content:center; color: var(--text-muted-color); background-color: var(--card-bg); border: 1px solid var(--card-border-color); font-weight: 600; transition: all 0.2s ease; font-size:0.9rem; margin: 0 0.2rem;}
        .page-item.active .page-link { background-color: var(--primary-color); border-color: var(--primary-color); color: white; box-shadow: 0 2px 8px rgba(var(--primary-color-rgb), 0.4); }
        .page-item.disabled .page-link { color: var(--text-muted-color); pointer-events: none; background-color: var(--light-bg); }
        .page-link-prev-next .page-link { width: auto; padding-left:1.2rem; padding-right:1.2rem; border-radius:50px; }
        footer { background: var(--footer-bg); color: var(--footer-text); margin-top: auto; padding: 3.5rem 0 1.5rem; font-size:0.9rem; }
        .footer-content.row { display: flex; flex-wrap: wrap; }
        .footer-section h5 { color: white; margin-bottom: 1.2rem; font-weight: 600; letter-spacing: 0.3px; font-size: 1.1rem; }
        .footer-links { display: flex; flex-direction: column; gap: 0.8rem; }
        .footer-links a { color: var(--footer-text); text-decoration: none; transition: all 0.2s ease; }
        .footer-links a:hover { color: var(--footer-link-hover); padding-left: 5px; }
        .social-links { display: flex; gap: 1rem; margin-top: 0.5rem; }
        .social-links a { color: var(--footer-text); font-size: 1.2rem; transition: all 0.2s ease; }
        .social-links a:hover { color: var(--secondary-light); transform: translateY(-2px); }
        .copyright { text-align: center; padding-top: 2rem; margin-top: 2rem; border-top: 1px solid #374151; font-size: 0.85rem; color: var(--text-muted-color); width: 100%; }
        .add-article-btn { width: 60px; height: 60px; border-radius: 50%; color: white; border: none; display: flex; align-items: center; justify-content: center; font-size: 24px; cursor: pointer; background-image: linear-gradient(to right, var(--primary-color) 0%, var(--primary-light) 100%); box-shadow: 0 4px 15px rgba(var(--primary-color-rgb), 0.35); transition: all 0.3s ease-out; }
        .add-article-btn:hover { transform: translateY(-4px) scale(1.05); box-shadow: 0 8px 25px rgba(var(--primary-color-rgb), 0.4); }
        .page-header-static { background-color: var(--card-bg); border-radius: var(--border-radius-lg); padding: 2.5rem; margin-bottom: 2rem; text-align: center; border-left: 5px solid var(--primary-color); }
        .page-header-static h1 { color: var(--text-color); font-size: 2.8rem; font-weight: 700; }
        body.dark-mode .page-header-static h1 { color: var(--primary-light); }
        .static-content-container { background-color: var(--card-bg); border-radius: var(--border-radius-lg); padding: clamp(1.5rem, 5vw, 3rem); font-size: 1.05rem; line-height: 1.8; box-shadow: var(--shadow-md); }
        .static-content-container h2 { font-family: 'Poppins', sans-serif; color: var(--primary-dark); border-bottom: 2px solid var(--secondary-color); padding-bottom: 0.5rem; margin-top: 2.5rem; margin-bottom: 1.5rem; display: inline-block; }
        .static-content-container h2 .icon { margin-right: 0.75rem; }
        .static-content-container p.lead { font-size: 1.25rem; font-weight: 400; color: var(--text-muted-color); }
        .static-content-container ul { padding-left: 25px; }
        .static-content-container li { margin-bottom: 0.5rem; }
        .contact-card { background-color: var(--light-bg); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-md); padding: 1.5rem; height: 100%; text-align: center; transition: all 0.3s ease; }
        .contact-card:hover { transform: translateY(-5px); box-shadow: var(--shadow-md); }
        .contact-card .icon { font-size: 2.5rem; color: var(--primary-color); margin-bottom: 1rem; }
        body.dark-mode .contact-card { background-color: var(--card-bg); }
        .contact-social-links { display: flex; gap: 1.5rem; justify-content: center; font-size: 1.5rem; }
        .contact-social-links a { color: var(--text-muted-color); transition: all 0.3s ease; }
        .contact-social-links a:hover { color: var(--secondary-color); transform: scale(1.1); }
        .auth-card { max-width: 480px; margin: 3rem auto; background: var(--card-bg); border-radius: var(--border-radius-lg); box-shadow: var(--shadow-lg); border: 1px solid var(--card-border-color); overflow: hidden; }
        .auth-header { padding: 2rem; background-color: var(--primary-color); text-align: center; }
        .auth-header .icon { font-size: 2.5rem; color: var(--secondary-light); }
        .auth-header h2 { color: white; font-weight: 600; margin-top: 0.75rem; margin-bottom: 0; }
        .auth-body { padding: 2rem 2.5rem; }
        .input-group-icon { position: relative; }
        .input-group-icon .form-control { padding-left: 2.5rem; }
        .input-group-icon .input-icon { position: absolute; left: 0.75rem; top: 50%; transform: translateY(-50%); color: var(--text-muted-color); }
        .auth-body .btn { padding: 0.75rem; font-weight: 600; font-size: 1rem; }
        .profile-header-card { background: var(--card-bg); border-radius: var(--border-radius-lg); padding: 2rem; box-shadow: var(--shadow-md); display: flex; flex-direction: column; align-items: center; text-align: center; }
        .profile-avatar-wrapper { position: relative; margin-bottom: 1rem; }
        .profile-avatar { width: 120px; height: 120px; border-radius: 50%; background-image: linear-gradient(to top, var(--primary-color), var(--primary-light)); color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 3.5rem; font-family: 'Poppins', sans-serif; border: 5px solid var(--card-bg); box-shadow: var(--shadow-md); }
        .profile-header-card h2 { margin-bottom: 0.25rem; font-size: 2rem; }
        .profile-header-card .username { color: var(--text-muted-color); font-weight: 500; margin-bottom: 1rem; }
        .profile-stats { display: flex; gap: 2rem; margin-top: 1.5rem; border-top: 1px solid var(--card-border-color); padding-top: 1.5rem; width: 100%; justify-content: center; }
        .stat-item { text-align: center; }
        .stat-item .icon { font-size: 1.5rem; color: var(--secondary-color); margin-bottom: 0.5rem; }
        .stat-item .count { font-size: 1.25rem; font-weight: 700; color: var(--text-color); }
        .stat-item .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted-color); }
        .profile-tabs .nav-link { padding: 0.75rem 1rem; }
        .empty-state-card { background-color: var(--card-bg); border-radius: var(--border-radius-lg); text-align: center; padding: 3rem; border: 2px dashed var(--card-border-color); }
        .empty-state-card .icon { font-size: 3.5rem; color: var(--text-muted-color); opacity: 0.5; margin-bottom: 1rem; }
        .admin-controls { position: fixed; bottom: 25px; right: 25px; z-index: 1030; }
        .bookmark-btn { background: none; border: none; font-size: 1.6rem; color: var(--text-muted-color); cursor: pointer; padding: 0.25rem 0.5rem; transition: all 0.2s ease; vertical-align: middle; }
        .bookmark-btn.active { color: var(--bookmark-active-color); transform: scale(1.1); }
        .bookmark-btn:hover { color: var(--secondary-light); }
        .article-card .bookmark-btn { font-size: 1.3rem; }
        .bookmark-btn:focus { outline: none; box-shadow: none; }
        
        /* === COMMENT SECTION STYLES (IMPROVED) === */
        .comment-section h3 { padding-bottom: 0.75rem; border-bottom: 1px solid var(--card-border-color); }
        .comment-thread { position: relative; }
        #comments-list > .comment-thread + .comment-thread { margin-top: 1.75rem; padding-top: 1.75rem; border-top: 1px solid var(--card-border-color); }
        .comment-container { display: flex; gap: 1rem; align-items: flex-start; }
        .comment-replies { margin-left: 3.5rem; padding-left: 1.25rem; margin-top: 1.25rem; border-left: 2px solid var(--card-border-color); }
        .comment-replies > .comment-thread + .comment-thread { margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px dashed var(--card-border-color); }
        .comment-avatar { width: 45px; height: 45px; border-radius: 50%; background: var(--primary-light); color: white; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0; }
        .comment-replies .comment-avatar { width: 40px; height: 40px; }
        .comment-body { flex-grow: 1; }
        .comment-header { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.25rem; }
        .comment-author { font-weight: 600; }
        .comment-date { font-size: 0.8rem; color: var(--text-muted-color); }
        .comment-content { word-wrap: break-word; }
        .comment-actions { position: relative; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }
        .comment-actions button { background: none; border: none; color: var(--text-muted-color); padding: 0.25rem 0.5rem; border-radius: var(--border-radius-md); font-size: 0.85rem; font-weight: 500; display: flex; align-items: center; gap: 0.3rem; transition: all 0.2s ease; }
        .comment-actions button:hover { color: var(--primary-color); background-color: rgba(var(--primary-color-rgb), 0.1); }
        .react-btn { position: relative; }
        .reaction-box { display: none; position: absolute; bottom: 100%; left: 0; margin-bottom: 8px; background-color: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: 50px; padding: 4px 8px; box-shadow: var(--shadow-md); z-index: 10; white-space: nowrap; animation: fadeInUp 0.2s ease-out; }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .reaction-box.show { display: flex; gap: 5px; }
        .reaction-emoji { font-size: 1.4rem; cursor: pointer; transition: transform 0.15s cubic-bezier(0.215, 0.610, 0.355, 1); padding: 2px; }
        .reaction-emoji:hover { transform: scale(1.25); }
        .reaction-summary { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
        .reaction-pill { display: flex; align-items: center; background-color: rgba(var(--primary-color-rgb), 0.08); border: 1px solid transparent; border-radius: 20px; padding: 2px 8px; font-size: 0.8rem; font-weight: 500; cursor: default; transition: all 0.2s ease; }
        .reaction-pill.user-reacted { background-color: var(--primary-color); color: white; border-color: var(--primary-dark); }
        .reaction-pill .emoji { font-size: 0.9rem; margin-right: 4px; }
        .reply-form-container { padding: 1rem; border-radius: var(--border-radius-md); margin-top: 0.75rem; background-color: var(--light-bg); border: 1px solid var(--card-border-color); }
        
        /* === FEATURED STORY & AI CARD SECTION === */
        .featured-story { display: flex; } /* Ensure it's visible by default */
        .ai-synthesis-card { display: block; } /* Ensure it's visible by default */
        .featured-story {
            background-color: var(--card-bg);
            border-radius: var(--border-radius-lg);
            box-shadow: var(--shadow-lg);
            margin-bottom: 2.5rem;
            overflow: hidden;
            border: 1px solid var(--card-border-color);
        }
        .featured-story-image {
            flex: 0 0 55%;
            background-size: cover;
            background-position: center;
            min-height: 450px;
        }
        .featured-story-content {
            flex: 0 0 45%;
            padding: 2.5rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .featured-story-content .meta-item {
            font-size: 0.9rem;
        }
        .featured-story-content h2 {
            font-size: 2.2rem;
            line-height: 1.3;
            margin: 1rem 0;
        }
        .featured-story-content h2 a {
            color: var(--text-color);
            text-decoration: none;
            transition: color 0.2s ease;
        }
        .featured-story-content h2 a:hover {
            color: var(--primary-color);
        }
        .featured-story-content .description {
            font-size: 1.05rem;
            color: var(--text-muted-color);
            margin-bottom: 2rem;
        }
        .featured-story-content .read-more-btn {
            background-color: var(--primary-color);
            color: white;
            padding: 0.8rem 1.5rem;
            text-decoration: none;
            border-radius: 50px;
            font-weight: 600;
            transition: all 0.3s ease;
            align-self: flex-start;
        }
        .featured-story-content .read-more-btn:hover {
            background-color: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .ai-synthesis-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border-color);
            border-radius: var(--border-radius-lg);
            padding: 2rem;
            margin-bottom: 2.5rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }
        .ai-synthesis-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: linear-gradient(135deg, rgba(var(--primary-color-rgb), 0.05) 25%, transparent 25%),
                              linear-gradient(225deg, rgba(var(--primary-color-rgb), 0.05) 25%, transparent 25%);
            background-size: 20px 20px;
            opacity: 0.5;
        }
        .synthesis-header {
            text-align: center;
            margin-bottom: 1.5rem;
            position: relative;
        }
        .synthesis-header i {
            font-size: 2rem;
            color: var(--primary-color);
        }
        .synthesis-header h2 {
            font-size: 1.5rem;
            margin-top: 0.5rem;
        }
        .synthesis-text {
            font-size: 1.15rem;
            line-height: 1.7;
            text-align: center;
            color: var(--text-color);
            position: relative;
            font-family: 'Inter', serif;
        }
        .synthesis-keywords {
            margin-top: 2rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--card-border-color);
            text-align: center;
            position: relative;
        }
        .synthesis-keywords .keyword-tag {
            display: inline-block;
            background-color: var(--light-bg);
            border: 1px solid var(--card-border-color);
            color: var(--text-muted-color);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            margin: 0.25rem;
            font-size: 0.9rem;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.2s ease-in-out;
        }
        .synthesis-keywords .keyword-tag:hover {
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: translateY(-2px);
        }

        /*
        ==========================================================================
        === MOBILE RESPONSIVE STYLES ===
        ==========================================================================
        */
        @media (max-width: 991.98px) {
            .featured-story {
                flex-direction: column;
            }
            .featured-story-image {
                flex-basis: auto;
                width: 100%;
                min-height: 300px;
            }
            .featured-story-content {
                flex-basis: auto;
                padding: 2rem;
            }
            .featured-story-content h2 {
                font-size: 2rem;
            }
        }
        @media (max-width: 767.98px) {
            .navbar-center {
                display: none;
            }
            .navbar-left {
                flex-grow: 1;
            }
            .page-header-static h1 {
                font-size: 2.2rem;
            }
            .featured-story-image {
                min-height: 220px;
            }
            .featured-story-content {
                padding: 1.5rem;
            }
            .featured-story-content h2 {
                font-size: 1.6rem;
            }
            .footer-section {
                text-align: center;
            }
            .footer-links {
                align-items: center;
            }
            .social-links {
                justify-content: center;
            }
            .footer-links a:hover {
                padding-left: 0;
            }
        }
        @media (max-width: 575.98px) {
            body {
                font-size: 0.95rem;
            }
            .navbar-brand-custom {
                font-size: 1.5rem;
            }
            .navbar-brand-custom .brand-icon {
                font-size: 1.6rem;
            }
            .comment-replies {
                margin-left: 1.25rem;
                padding-left: 1rem;
            }
            .comment-container {
                gap: 0.75rem;
            }
            .comment-avatar {
                width: 40px;
                height: 40px;
            }
            .comment-replies .comment-avatar {
                width: 35px;
                height: 35px;
            }
            .profile-avatar {
                width: 100px;
                height: 100px;
                font-size: 3rem;
            }
            .profile-header-card h2 {
                font-size: 1.5rem;
            }
            .profile-stats {
                gap: 1rem;
                flex-wrap: wrap;
            }
            .auth-card {
                margin: 1rem auto;
                border: none;
                box-shadow: none;
            }
            .ai-synthesis-card, .auth-body, .static-content-container,
            .profile-header-card, .article-body, .article-full-content-wrapper {
                padding: 1.5rem;
            }
            .article-full-content-wrapper {
                padding: 1.5rem 1rem;
            }
            .synthesis-text {
                font-size: 1.05rem;
            }
        }
    </style>
    {% block head_extra %}{% endblock %}
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-6975904325280886" crossorigin="anonymous"></script>
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-CV5LWJ7NQ7"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-CV5LWJ7NQ7');
    </script>
</head>
<body class="{{ request.cookies.get('darkMode', 'disabled') }}{% block body_class %}{% endblock %}">
    
    <header>
        <nav class="navbar-main">
            <div class="container">
                <div class="navbar-content-wrapper">
                    <div class="navbar-left">
                        <a class="navbar-brand-custom" href="{{ url_for('index') }}">
                            <i class="fas fa-bolt-lightning brand-icon"></i>
                            <span>BrieflyAI</span>
                        </a>
                    </div>
                    <div class="navbar-center">
                        <form action="{{ url_for('search_results') }}" method="GET" class="search-container">
                            <input type="search" name="query" class="form-control navbar-search" placeholder="Search news articles..." value="{{ request.args.get('query', '') }}">
                            <i class="fas fa-search search-icon"></i>
                        </form>
                    </div>
                    <div class="navbar-right">
                        <div class="header-controls">
                            <button class="header-btn" type="button" data-bs-toggle="offcanvas" data-bs-target="#mainOffcanvas" aria-controls="mainOffcanvas">
                                <i class="fas fa-bars"></i>
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
            <h5 class="offcanvas-title" id="mainOffcanvasLabel">BrieflyAI Menu</h5>
            <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close"></button>
        </div>
        <div class="offcanvas-body d-flex flex-column">

            <div class="sidebar-section">
                {% if session.user_id %}
                    <div class="dropdown">
                        <a href="#" class="sidebar-btn dropdown-toggle" id="offcanvasUserDropdown" data-bs-toggle="dropdown" aria-expanded="false">
                            <div class="sidebar-avatar">{{ session.user_name[0]|upper }}</div>
                            <strong class="ms-3">{{ session.user_name|truncate(20) }}</strong>
                        </a>
                        <ul class="dropdown-menu shadow" aria-labelledby="offcanvasUserDropdown">
                            <li><a class="dropdown-item" href="{{ url_for('profile') }}"><i class="fas fa-id-card fa-fw me-2"></i>Profile</a></li>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item" href="{{ url_for('logout') }}"><i class="fas fa-sign-out-alt fa-fw me-2"></i>Logout</a></li>
                        </ul>
                    </div>
                {% else %}
                    <a href="{{ url_for('login') }}" class="sidebar-btn">
                        <i class="fas fa-sign-in-alt fa-fw me-2"></i> Login / Register
                    </a>
                {% endif %}
            </div>

            <div class="sidebar-section">
                 <button class="sidebar-btn dark-mode-toggle w-100">
                    <i class="fas fa-moon fa-fw me-2"></i> <span class="theme-text">Dark Mode</span>
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
                            <a href="{{ url_for('index', **cat_url_params) }}" class="nav-link {% if selected_category == cat_item %}active{% endif %}">
                                <i class="fas fa-fw fa-{% if cat_item == 'All Articles' %}globe-americas{% elif cat_item == 'Popular Stories' %}fire-alt{% elif cat_item == "Yesterday's Headlines" %}history{% elif cat_item == 'Community Hub' %}users{% endif %} me-2"></i>
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

    <div id="alert-placeholder">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="alert alert-{{ category }} alert-dismissible fade show alert-top" role="alert">
                    <span>{{ message }}</span>
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
    </div>

    <main class="container main-content my-4">
        {% block content %}{% endblock %}
    </main>
    
    {% if session.user_id %}
    <div class="admin-controls">
        <button class="add-article-btn" data-bs-toggle="modal" data-bs-target="#addArticleModal" title="Post a New Article">
            <i class="fas fa-pen-to-square"></i>
        </button>
    </div>
    <div class="modal fade" id="addArticleModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content p-4">
                <div class="modal-header border-0 pb-0">
                    <h4 class="modal-title">Post New Article</h4>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <form id="addArticleForm" action="{{ url_for('post_article') }}" method="POST">
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
                        <i class="fas fa-bolt-lightning me-2" style="color:var(--secondary-light); font-size: 1.5rem;"></i>
                        <span class="h5 mb-0" style="color:white; font-family: 'Poppins', sans-serif;">BrieflyAI</span>
                    </div>
                    <p class="small text-light">Your premier source for AI summarized, India-centric news.</p>
                    <div class="social-links">
                        <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a><a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a><a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a><a href="#" title="Instagram"><i class="fab fa-instagram"></i></a>
                    </div>
                </div>
                <div class="footer-section col-lg-2 col-md-6 mb-4">
                    <h5>Quick Links</h5>
                    <div class="footer-links">
                        <a href="{{ url_for('index') }}">Home</a>
                        <a href="{{ url_for('about') }}">About Us</a>
                        <a href="{{ url_for('contact') }}">Contact</a>
                        <a href="{{ url_for('privacy') }}">Privacy Policy</a>
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
                        <div class="input-group">
                            <input type="email" name="email" class="form-control form-control-sm" placeholder="Your Email" aria-label="Your Email" required style="background: #374151; border-color: #4B5563; color: white;">
                            <button class="btn btn-sm btn-primary" type="submit">Subscribe</button>
                        </div>
                    </form>
                </div>
            </div>
            <div class="copyright">&copy; 2025 BrieflyAI. All rights reserved. Made with <i class="fas fa-heart text-danger"></i> in India.</div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    document.addEventListener('DOMContentLoaded', function () {
        try {
            const darkModeToggle = document.querySelector('.dark-mode-toggle');
            if (darkModeToggle) {
                const body = document.body;
                const updateThemeUI = () => {
                    const isDarkMode = body.classList.contains('dark-mode');
                    const themeIcon = darkModeToggle.querySelector('i');
                    const themeText = darkModeToggle.querySelector('.theme-text');
                    if (themeIcon) {
                        themeIcon.className = isDarkMode ? 'fas fa-sun fa-fw me-2' : 'fas fa-moon fa-fw me-2';
                    }
                    if (themeText) {
                        themeText.textContent = isDarkMode ? 'Light Mode' : 'Dark Mode';
                    }
                };
                const applyTheme = (theme) => {
                    body.classList.toggle('dark-mode', theme === 'enabled');
                    localStorage.setItem('darkMode', theme);
                    document.cookie = "darkMode=" + theme + ";path=/;max-age=31536000;SameSite=Lax";
                    updateThemeUI();
                };
                darkModeToggle.addEventListener('click', () => {
                    const isEnabled = body.classList.contains('dark-mode');
                    applyTheme(isEnabled ? 'disabled' : 'enabled');
                });
                updateThemeUI();
            }

            const flashedAlerts = document.querySelectorAll('#alert-placeholder .alert');
            flashedAlerts.forEach(function(alert) { 
                setTimeout(function() {
                    const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
                    if (bsAlert) bsAlert.close();
                }, 7000);
            });

            const dateFilterForm = document.getElementById('dateFilterForm');
            if (dateFilterForm) {
                dateFilterForm.addEventListener('submit', function(event) {
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
                    clearDateFilterBtn.addEventListener('click', function() {
                        window.location.href = "{{ url_for('index', category_name='All Articles') }}";
                    });
                }
            }
        } catch (e) {
            console.error("An error occurred in the base layout script:", e);
        }
    });
    </script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
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
                <i class="fas fa-brain"></i>
                <h2>Today's Briefing: The Big Picture</h2>
            </div>
            <p class="synthesis-text">"{{ synthesis }}"</p>
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
            <div class="featured-story-image" style="background-image: url('{{ featured_article.urlToImage }}')"></div>
            <div class="featured-story-content">
                <div class="article-meta">
                    <span class="meta-item"><i class="fas fa-fire-alt text-danger"></i> Top Story</span>
                    <span class="meta-item"><i class="fas fa-building"></i> {{ featured_article.source.name|truncate(20) }}</span>
                </div>
                <h2><a href="{{ url_for('article_detail', article_hash_id=featured_article.id) }}">{{ featured_article.title }}</a></h2>
                <p class="description">{{ featured_article.description|truncate(150) }}</p>
                <a href="{{ url_for('article_detail', article_hash_id=featured_article.id) }}" class="read-more-btn">Read Full Story <i class="fas fa-arrow-right ms-1"></i></a>
            </div>
        </article>
        {% endif %}

        <ul class="nav nav-tabs nav-fill mb-3" id="newsTab" role="tablist" style="font-weight: 600;">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="popular-tab" data-bs-toggle="tab" data-bs-target="#popular-tab-pane" type="button" role="tab" aria-controls="popular-tab-pane" aria-selected="true">
                    <i class="fas fa-fire-alt me-1"></i> POPULAR STORIES
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="yesterday-tab" data-bs-toggle="tab" data-bs-target="#yesterday-tab-pane" type="button" role="tab" aria-controls="yesterday-tab-pane" aria-selected="false">
                    <i class="fas fa-history me-1"></i> YESTERDAY'S HEADLINES
                </button>
            </li>
        </ul>

        <div class="tab-content" id="newsTabContent">
            <div class="tab-pane fade show active" id="popular-tab-pane" role="tabpanel" aria-labelledby="popular-tab">
                <div class="row g-4 pt-3">
                    {% if popular_articles %}
                        {% for art in popular_articles %}
                            <div class="col-md-6 col-lg-4 d-flex">
                                <article class="article-card d-flex flex-column w-100">
                                    {% set article_url = url_for('article_detail', article_hash_id=art.id) %}
                                    <div class="article-image-container"><a href="{{ article_url }}"><img src="{{ art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a></div>
                                    <div class="article-body d-flex flex-column">
                                        <div class="d-flex justify-content-between align-items-start">
                                            <h5 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                                            {% if session.user_id %}<button class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" style="margin-left: 10px; padding-top:0;" title="Bookmark" data-article-hash-id="{{ art.id }}" data-is-community="false" data-title="{{ art.title|e }}" data-source-name="{{ art.source.name|e }}" data-image-url="{{ art.urlToImage|e }}" data-description="{{ (art.description if art.description else '')|e }}" data-published-at="{{ (art.publishedAt if art.publishedAt else '')|e }}"><i class="fa-solid fa-bookmark"></i></button>{% endif %}
                                        </div>
                                        <div class="article-meta small mb-2">
                                            <span class="meta-item text-muted"><i class="fas fa-building"></i> {{ art.source.name|truncate(20) }}</span>
                                            <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                                        </div>
                                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                                    </div>
                                </article>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="col-12"><div class="alert alert-light text-center">More popular stories are currently unavailable.</div></div>
                    {% endif %}
                </div>
                {% if popular_articles %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('index', category_name='Popular Stories') }}" class="btn btn-outline-primary">View All Popular Stories <i class="fas fa-arrow-right ms-1"></i></a>
                </div>
                {% endif %}
            </div>
            <div class="tab-pane fade" id="yesterday-tab-pane" role="tabpanel" aria-labelledby="yesterday-tab">
                <div class="row g-4 pt-3">
                    {% if latest_yesterday_articles %}
                        {% for art in latest_yesterday_articles %}
                             <div class="col-md-6 col-lg-4 d-flex">
                                <article class="article-card d-flex flex-column w-100">
                                    {% set article_url = url_for('article_detail', article_hash_id=art.id) %}
                                    <div class="article-image-container"><a href="{{ article_url }}"><img src="{{ art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a></div>
                                    <div class="article-body d-flex flex-column">
                                        <div class="d-flex justify-content-between align-items-start">
                                            <h5 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                                            {% if session.user_id %}<button class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" style="margin-left: 10px; padding-top:0;" title="Bookmark" data-article-hash-id="{{ art.id }}" data-is-community="false" data-title="{{ art.title|e }}" data-source-name="{{ art.source.name|e }}" data-image-url="{{ art.urlToImage|e }}" data-description="{{ (art.description if art.description else '')|e }}" data-published-at="{{ (art.publishedAt if art.publishedAt else '')|e }}"><i class="fa-solid fa-bookmark"></i></button>{% endif %}
                                        </div>
                                        <div class="article-meta small mb-2">
                                            <span class="meta-item text-muted"><i class="fas fa-building"></i> {{ art.source.name|truncate(20) }}</span>
                                            <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                                        </div>
                                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                                    </div>
                                </article>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="col-12"><div class="alert alert-light text-center">Could not load yesterday's articles.</div></div>
                    {% endif %}
                </div>
                {% if latest_yesterday_articles %}
                <div class="text-center mt-4">
                    <a href="{{ url_for('index', category_name="Yesterday's Headlines") }}" class="btn btn-outline-primary">View All of Yesterday's Headlines <i class="fas fa-arrow-right ms-1"></i></a>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

{% else %}

    {# ============ LAYOUT 2: STANDARD PAGINATED LIST VIEW (RESTORED) ============ #}
    {# This block handles all other pages like categories, search, and date filters. #}

    {% if selected_category == 'All Articles' and current_filter_date %}
        <h4 class="mb-3 fst-italic">Showing articles for: {{ current_filter_date }}</h4>
    {% elif selected_category != 'All Articles' and selected_category != 'Community Hub' %}
         <h2 class="pb-2 border-bottom mb-4">{{ selected_category }}</h2>
    {% endif %}

    {% if articles and not is_main_homepage %} {# This section is for the paginated list view only #}
        <div class="row g-4">
            {% for art in articles %}
            <div class="col-md-6 col-lg-4 d-flex">
                <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ loop.index0 * 0.05 }}s">
                    {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
                    <div class="article-image-container">
                        <a href="{{ article_url }}">
                        <img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
                    </div>
                    <div class="article-body d-flex flex-column">
                        <div class="d-flex justify-content-between align-items-start">
                            <h5 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                            {% if session.user_id %}
                            <button class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" style="margin-left: 10px; padding-top:0;"
                                    title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}"
                                    data-article-hash-id="{{ art.article_hash_id if art.is_community_article else art.id }}"
                                    data-is-community="{{ 'true' if art.is_community_article else 'false' }}"
                                    data-title="{{ art.title|e }}"
                                    data-source-name="{{ (art.author.name if art.is_community_article and art.author else art.source.name)|e }}"
                                    data-image-url="{{ (art.image_url if art.is_community_article else art.urlToImage)|e }}"
                                    data-description="{{ (art.description if art.description else '')|e }}"
                                    data-published-at="{{ (art.published_at.isoformat() if art.is_community_article and art.published_at else (art.publishedAt if not art.is_community_article and art.publishedAt else ''))|e }}">
                                <i class="fa-solid fa-bookmark"></i>
                            </button>
                            {% endif %}
                        </div>
                        <div class="article-meta small mb-2">
                            <span class="meta-item text-muted"><i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}"></i> {% if art.is_community_article and art.author %}<a href="{{ url_for('public_profile', username=art.author.username) }}" class="text-muted text-decoration-none">{{ art.author.name|truncate(20) }}</a>{% else %}{{ art.source.name|truncate(20) }}{% endif %}</span>
                            <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.published_at | to_ist if art.is_community_article else (art.publishedAt | to_ist if art.publishedAt else 'N/A')) }}</span>
                        </div>
                        <p class="article-description small">{{ art.description|truncate(100) }}</p>
                        <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                    </div>
                </article>
            </div>
            {% endfor %}
        </div>
    {% elif not articles %}
        <div class="alert alert-info text-center my-5 p-4"><h4><i class="fas fa-search me-2"></i>No articles found.</h4><p>Please try a different category or search query.</p></div>
    {% endif %}

    {% if total_pages and total_pages > 1 %}
    <nav aria-label="Page navigation" class="mt-5"><ul class="pagination justify-content-center">
        {% set filter_date_for_url = request.args.get('filter_date') if selected_category == 'All Articles' and request.args.get('filter_date') else None %}
        <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}">
            <a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) if current_page > 1 else '#' }}">&laquo; Prev</a>
        </li>
        {% set page_window = 1 %}{% set show_first = 1 %}{% set show_last = total_pages %}
        {% if current_page - page_window > show_first %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">1</a></li>{% if current_page - page_window > show_first + 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}{% endif %}
        {% for p in range(1, total_pages + 1) %}{% if p == current_page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% elif p >= current_page - page_window and p <= current_page + page_window %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">{{ p }}</a></li>{% endif %}{% endfor %}
        {% if current_page + page_window < show_last %}{% if current_page + page_window < show_last - 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=total_pages, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">{{ total_pages }}</a></li>{% endif %}
        <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}">
            <a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) if current_page < total_pages else '#' }}">Next &raquo;</a>
        </li>
    </ul></nav>
    {% endif %}
{% endif %}
{% endblock %}

{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
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
                fetch(`{{ url_for('toggle_bookmark', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashId), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ is_community_article: isCommunity, title: title, source_name: sourceName, image_url: imageUrl, description: description, published_at: publishedAt })
                })
                .then(res => { if (!res.ok) { return res.json().then(err => { throw new Error(err.error || `HTTP error! status: ${res.status}`); }); } return res.json(); })
                .then(data => {
                    if (data.success) {
                        this.classList.toggle('active', data.status === 'added');
                        this.title = data.status === 'added' ? 'Remove Bookmark' : 'Add Bookmark';
                        const alertPlaceholder = document.getElementById('alert-placeholder');
                        if(alertPlaceholder) {
                            const existingAlerts = alertPlaceholder.querySelectorAll('.bookmark-alert');
                            existingAlerts.forEach(al => bootstrap.Alert.getOrCreateInstance(al)?.close());
                            const alertDiv = `<div class="alert alert-info alert-dismissible fade show alert-top bookmark-alert" role="alert" style="z-index: 2060;">${data.message}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>`;
                            alertPlaceholder.insertAdjacentHTML('beforeend', alertDiv);
                            const newAlert = alertPlaceholder.lastChild;
                            setTimeout(() => { bootstrap.Alert.getOrCreateInstance(newAlert)?.close(); }, 3000);
                        }
                    } else { alert('Error: ' + (data.error || 'Could not update bookmark.')); }
                })
                .catch(err => { console.error("Bookmark error on homepage:", err); alert("Could not update bookmark: " + err.message); });
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
{% block head_extra %}
<style>
    .article-full-content-wrapper { background-color: var(--card-bg); padding: clamp(1rem, 4vw, 2rem); border-radius: var(--border-radius-lg); box-shadow: var(--shadow-md); margin-bottom: 2rem; margin-top: 1rem; }
    .article-title-main {font-weight: 700; color: var(--text-color); line-height:1.3; font-family: 'Poppins', sans-serif;}
    .summary-box, .takeaways-box { background-color: rgba(var(--primary-color-rgb), 0.04); border: 1px solid rgba(var(--primary-color-rgb), 0.1); border-radius: var(--border-radius-md); margin: 1.5rem 0; padding: 1.5rem; }
    body.dark-mode .summary-box, body.dark-mode .takeaways-box { background-color: #2a30422e; }
    .takeaways-box { border-left: 4px solid var(--secondary-color); }
    .loader-container { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 200px; padding: 2rem; color: var(--text-muted-color); }
    .loader { border: 5px solid var(--light-bg); border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin-bottom: 1rem; }
    .content-text { white-space: pre-wrap; line-height: 1.8; font-size: 1.05rem; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .edit-form-container { display: none; } /* Hide edit form by default */
</style>
{% endblock %}
{% block content %}
{% if not article %}
    <div class="alert alert-danger text-center my-5 p-4"><h4><i class="fas fa-exclamation-triangle me-2"></i>Article Not Found</h4><p>The article you are looking for could not be found.</p><a href="{{ url_for('index') }}" class="btn btn-primary mt-2">Go to Homepage</a></div>
{% else %}
<article class="article-full-content-wrapper animate-fade-in">
    <div class="mb-3 d-flex justify-content-between align-items-center">
        <a href="{{ previous_list_page }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-arrow-left me-2"></i>Back to List</a>
        
        <div class="d-flex align-items-center gap-2">
            
            {% if session.get('is_admin') and is_community_article %}
            <button id="adminDeleteBtn" class="btn btn-sm btn-danger" title="Admin: Delete Post">
                <i class="fas fa-trash-alt"></i> Delete Post
            </button>
            {% endif %}

            {% if session.user_id %}
                {% if is_community_article %}
                <button id="reportBtn" class="btn btn-sm btn-outline-warning" title="Report this article for review">
                    <i class="fas fa-flag"></i> Report
                </button>
                {% endif %}
                
                <button id="bookmarkBtn" class="bookmark-btn {% if is_bookmarked %}active{% endif %}" title="{% if is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" data-article-hash-id="{{ article.article_hash_id if is_community_article else article.id }}" data-is-community="{{ 'true' if is_community_article else 'false' }}" data-title="{{ article.title|e }}" data-source-name="{{ (article.author.name if is_community_article and article.author else article.source.name)|e }}" data-image-url="{{ (article.image_url if is_community_article else article.urlToImage)|e }}" data-description="{{ (article.description if article.description else '')|e }}" data-published-at="{{ (article.published_at.isoformat() if is_community_article and article.published_at else (article.publishedAt if not is_community_article and article.publishedAt else ''))|e }}"><i class="fa-solid fa-bookmark"></i></button>
            {% endif %}
        </div>
    </div>

    <h1 class="mb-2 article-title-main display-6">{{ article.title }}</h1>
    <div class="article-meta-detailed d-flex align-items-center flex-wrap gap-3 text-muted small"><span class="meta-item" title="Source"><i class="fas fa-{{ 'user-edit' if is_community_article else 'building' }}"></i> {{ article.author.name if is_community_article and article.author else article.source.name }}</span><span class="meta-item" title="Published Date"><i class="far fa-calendar-alt"></i> {{ (article.published_at | to_ist if is_community_article else (article.publishedAt | to_ist if article.publishedAt else 'N/A')) }}</span></div>
    {% set image_to_display = article.image_url if is_community_article else article.urlToImage %}
    {% if image_to_display %}<img src="{{ image_to_display }}" alt="{{ article.title|truncate(50) }}" class="img-fluid rounded my-3 shadow-sm">{% endif %}
    
    <div id="contentLoader" class="loader-container my-4 {% if is_community_article %}d-none{% endif %}"><div class="loader"></div><div>Analyzing article and generating summary...</div></div>
    <div id="articleAnalysisContainer">
    {% if is_community_article %}
        {% if article.groq_summary %}<div class="summary-box my-3"><h5><i class="fas fa-book-open me-2"></i>AI Summary</h5><p class="mb-0">{{ article.groq_summary|replace('\\n', '<br>')|safe }}</p></div>{% endif %}
        {% if article.parsed_takeaways %}<div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>AI Key Takeaways</h5><ul>{% for takeaway in article.parsed_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul></div>{% endif %}
        <hr class="my-4"><h4 class="mb-3">Full Article Content</h4><div class="content-text">{{ article.full_text }}</div>
    {% else %}<div id="apiArticleContent"></div>{% endif %}
    </div>

    <section class="comment-section mt-5" id="comment-section">
        <h3 class="mb-4">Community Discussion (<span id="comment-count">{{ total_comment_count }}</span>)</h3>
        
        <div id="comments-list">
            {% for comment in comments %}
                {% include '_COMMENT_TEMPLATE' %}
            {% else %}
                <p id="no-comments-msg" class="text-muted mt-3">No comments yet. Be the first to share your thoughts!</p>
            {% endfor %}
        </div>

        {% if session.user_id %}
            <div class="add-comment-form mt-4 pt-4 border-top">
                <h5 class="mb-3">Leave a Comment</h5>
                <form id="comment-form">
                    <div class="mb-3"><textarea class="form-control" id="comment-content" name="content" rows="4" placeholder="Share your insights..." required></textarea></div>
                    <button type="submit" class="btn btn-primary">Post Comment</button>
                </form>
            </div>
        {% else %}
            <div class="alert alert-light mt-4 text-center">Please <a href="{{ url_for('login', next=request.url) }}" class="fw-bold">log in</a> to join the discussion.</div>
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

        const adminDeleteBtn = document.getElementById('adminDeleteBtn');
        if (adminDeleteBtn) {
            adminDeleteBtn.addEventListener('click', function() {
                if (!confirm('ADMIN ACTION: Are you sure you want to permanently delete this community post? This action cannot be undone.')) {
                    return;
                }
                this.disabled = true;
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
                
                fetch(`/delete_community_article/${articleHashIdGlobal}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                })
                .then(res => res.json().then(data => ({ ok: res.ok, data })))
                .then(({ ok, data }) => {
                    if (ok && data.success) {
                        // On successful deletion, redirect to the Community Hub
                        window.location.href = data.redirect_url;
                    } else {
                        alert('Deletion failed: ' + (data.error || 'Unknown error'));
                        this.disabled = false;
                        this.innerHTML = '<i class="fas fa-trash-alt"></i> Delete Post';
                    }
                })
                .catch(err => {
                    console.error("Admin delete error:", err);
                    alert("A network error occurred. Could not delete the post.");
                    this.disabled = false;
                    this.innerHTML = '<i class="fas fa-trash-alt"></i> Delete Post';
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
                            if (analysis.groq_summary) { html += `<div class="summary-box my-3"><h5><i class="fas fa-book-open me-2"></i>AI Summary</h5><p class="mb-0">${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`; }
                            if (analysis.groq_takeaways && analysis.groq_takeaways.length > 0) { html += `<div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>AI Key Takeaways</h5><ul>${analysis.groq_takeaways.map(t => `<li>${String(t)}</li>`).join('')}</ul></div>`; }
                        }
                    }
                    if (articleUrl) { html += `<hr class="my-4"><a href="${articleUrl}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">Read Original Article at ${articleSourceName} <i class="fas fa-external-link-alt ms-1"></i></a>`; }
                    apiArticleContent.innerHTML = html;
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
                fetch(`{{ url_for('add_comment', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashIdGlobal), {
                    method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify({ content, parent_id: parentId })
                })
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
                .catch(err => { console.error("Comment submission error:", err); alert("Error: " + err.message); })
                .finally(() => { submitButton.disabled = false; submitButton.innerHTML = originalButtonText; });
            };

            const updateReactionUI = (commentId, reactions, userReaction) => {
                const summaryContainer = document.getElementById(`reaction-summary-${commentId}`);
                if (!summaryContainer) return;
                let summaryHTML = '';
                if (reactions) {
                    for (const [emoji, count] of Object.entries(reactions)) {
                        if (count > 0) {
                            const userReactedClass = (userReaction === emoji) ? 'user-reacted' : '';
                            summaryHTML += `<div class="reaction-pill ${userReactedClass}" data-emoji="${emoji}"><span class="emoji">${emoji}</span> <span class="count">${count}</span></div>`;
                        }
                    }
                }
                summaryContainer.innerHTML = summaryHTML;
            };

            commentSection.addEventListener('click', function(e) {
                const target = e.target;
                
                const deleteBtn = target.closest('.delete-btn');
                if (deleteBtn) {
                    e.preventDefault();
                    const commentId = deleteBtn.dataset.commentId;
                    if (confirm('Are you sure you want to delete this comment? All replies will also be removed.')) {
                        fetch(`/delete_comment/${commentId}`, { method: 'POST' })
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
                                    
                                    commentElement.style.transition = 'opacity 0.5s ease';
                                    commentElement.style.opacity = '0';
                                    setTimeout(() => commentElement.remove(), 500);
                                } else {
                                    alert('Error: ' + data.error);
                                }
                            })
                            .catch(err => {
                                console.error("Delete error:", err);
                                alert("Could not delete comment: " + err.message);
                            });
                    }
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
                    commentBody.querySelector('.edit-form-container').style.display = 'block';
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
                        document.querySelectorAll('.reaction-box').forEach(box => box.classList.remove('show'));
                        if (!isShown) reactionBox.classList.add('show');
                    }
                    return;
                }

                const reactionEmoji = target.closest('.reaction-emoji');
                if (reactionEmoji) {
                    e.preventDefault();
                    const commentId = reactionEmoji.dataset.commentId;
                    const emoji = reactionEmoji.dataset.emoji;
                    reactionEmoji.closest('.reaction-box').classList.remove('show');
                    fetch(`/vote_comment/${commentId}`, {
                        method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                        body: JSON.stringify({ emoji: emoji })
                    })
                    .then(res => res.json())
                    .then(data => {
                        if (data.success) { updateReactionUI(commentId, data.reactions, data.user_reaction); } 
                        else { throw new Error(data.error || "Failed to vote."); }
                    })
                    .catch(err => { console.error("Reaction error:", err); alert("Error: " + err.message); });
                    return;
                }
                
                if (!target.closest('.reaction-box') && !target.closest('.react-btn')) {
                    document.querySelectorAll('.reaction-box.show').forEach(box => box.classList.remove('show'));
                }
            });

            commentSection.addEventListener('submit', function(e) {
                e.preventDefault();

                if (e.target.matches('.edit-comment-form')) {
                    const form = e.target;
                    const commentId = form.closest('.comment-thread').id.replace('comment-', '');
                    const newContent = form.querySelector('textarea[name="content"]').value.trim();
                    if (!newContent) return;

                    fetch(`/edit_comment/${commentId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ content: newContent })
                    })
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
                            alert('Error: ' + data.error);
                        }
                    })
                    .catch(err => {
                        console.error("Edit error:", err);
                        alert("Could not save changes: " + err.message);
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
            reportBtn.addEventListener('click', function() {
                if (!confirm('Are you sure you want to report this article for review?')) {
                    return;
                }
                this.disabled = true;
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Reporting...';
                fetch(`/report_article/${articleHashIdGlobal}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                })
                .then(res => res.json().then(data => ({ ok: res.ok, status: res.status, data })))
                .then(({ ok, status, data }) => {
                    if (ok) {
                        this.innerHTML = '<i class="fas fa-check"></i> Reported';
                        alert(data.message);
                    } else {
                        this.disabled = false;
                        this.innerHTML = '<i class="fas fa-flag"></i> Report';
                        alert('Error: ' + data.error);
                    }
                })
                .catch(err => {
                    console.error("Report error:", err);
                    alert("A network error occurred. Please try again.");
                    this.disabled = false;
                    this.innerHTML = '<i class="fas fa-flag"></i> Report';
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
                fetch(`{{ url_for('toggle_bookmark', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashId), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ is_community_article: isCommunity, title, source_name: sourceName, image_url: imageUrl, description, published_at: publishedAt }) })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        this.classList.toggle('active', data.status === 'added');
                        this.title = data.status === 'added' ? 'Remove Bookmark' : 'Add Bookmark';
                    } else { alert('Error: ' + (data.error || 'Could not update bookmark.')); }
                })
                .catch(err => { console.error("Bookmark error:", err); alert("Could not update bookmark: " + err.message); });
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

_COMMENT_TEMPLATE = """
<div class="comment-thread" id="comment-{{ comment.id }}">
    <div class="comment-container">
        <div class="comment-avatar" title="{{ comment.author.name if comment.author else 'Unknown' }}">{{ (comment.author.name[0]|upper if comment.author and comment.author.name else 'U') }}</div>
        <div class="comment-body">
            <div class="comment-header">
                <a href="{{ url_for('public_profile', username=comment.author.username) }}" class="comment-author text-decoration-none">{{ comment.author.name if comment.author else 'Anonymous' }}</a>
                <span class="comment-date">{{ comment.timestamp | to_ist }}</span>
            </div>

            {# This is the main content paragraph #}
            <p class="comment-content mb-2">{{ comment.content }}</p>

            {# NEW: This is the hidden form for editing the comment #}
            <div class="edit-form-container" style="display:none;">
                <form class="edit-comment-form">
                    <textarea class="form-control form-control-sm mb-2" name="content" rows="3" required>{{ comment.content }}</textarea>
                    <div class="d-flex justify-content-end gap-2">
                        <button type="button" class="btn btn-sm btn-outline-secondary cancel-edit-btn">Cancel</button>
                        <button type="submit" class="btn btn-sm btn-primary">Save Changes</button>
                    </div>
                </form>
            </div>
            
            {% if session.user_id %}
            <div class="comment-actions">
                <div class="reaction-box" id="reaction-box-{{ comment.id }}">
                    {% for emoji in ['👍', '❤️', '😂', '😮', '😢', '😠'] %}
                        <span class="reaction-emoji" data-emoji="{{ emoji }}" data-comment-id="{{ comment.id }}" title="{{ emoji }}">{{ emoji }}</span>
                    {% endfor %}
                </div>
                <button class="react-btn" data-comment-id="{{ comment.id }}" title="React"><i class="far fa-smile"></i> React</button>
                <button class="reply-btn" data-comment-id="{{ comment.id }}" title="Reply"><i class="fas fa-reply"></i> Reply</button>
                
                {# NEW: Edit and Delete buttons, only visible to the comment owner #}
                {% if session.user_id == comment.user_id %}
                    <button class="edit-btn" data-comment-id="{{ comment.id }}" title="Edit"><i class="fas fa-pencil-alt"></i> Edit</button>
                    <button class="delete-btn" data-comment-id="{{ comment.id }}" title="Delete"><i class="fas fa-trash-alt"></i> Delete</button>
                {% endif %}
            </div>
            <div class="reply-form-container" id="reply-form-container-{{ comment.id }}" style="display:none;">
                <form class="reply-form">
                    <input type="hidden" name="parent_id" value="{{ comment.id }}">
                    <div class="mb-2"><textarea class="form-control form-control-sm" name="content" rows="2" placeholder="Write a reply..." required></textarea></div>
                    <div class="d-flex justify-content-end gap-2">
                        <button type="button" class="btn btn-sm btn-outline-secondary cancel-reply-btn">Cancel</button>
                        <button type="submit" class="btn btn-sm btn-primary">Post Reply</button>
                    </div>
                </form>
            </div>
            {% endif %}

            <div class="reaction-summary" id="reaction-summary-{{ comment.id }}">
                {# Reaction pills will be dynamically inserted here by JS #}
            </div>
        </div>
    </div>
    <div class="comment-replies" id="replies-of-{{ comment.id }}">
        {% for comment in comment.replies %}
            {% include '_COMMENT_TEMPLATE' %}
        {% endfor %}
    </div>
</div>
"""

LOGIN_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Login - BrieflyAI{% endblock %}
{% block body_class %}body-auth{% endblock %}

{% block content %}
<div class="auth-card animate-fade-in">
    <div class="auth-header">
        <div class="brand-icon"><i class="fas fa-bolt-lightning"></i></div>
        <h2>Welcome Back to BrieflyAI</h2>
    </div>
    <div class="auth-body">
        <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}">
            <div class="mb-3">
                <label for="username" class="form-label fw-medium">Username</label>
                <div class="input-group-icon">
                    <i class="fas fa-user input-icon"></i>
                    <input type="text" class="form-control" id="username" name="username" required placeholder="e.g. user123">
                </div>
            </div>
            <div class="mb-4">
                <label for="password" class="form-label fw-medium">Password</label>
                <div class="input-group-icon">
                    <i class="fas fa-lock input-icon"></i>
                    <input type="password" class="form-control" id="password" name="password" required placeholder="••••••••">
                </div>
            </div>
            <button type="submit" class="btn btn-primary w-100">Sign In</button>
        </form>
        
        <div class="social-login-divider">Or</div>

        <div class="social-login-buttons d-flex gap-3">
            <a href="#" class="btn w-100"><i class="fab fa-google"></i> Google</a>
            <a href="#" class="btn w-100"><i class="fab fa-facebook"></i> Facebook</a>
        </div>
    </div>
    <div class="auth-footer">
        <p class="mb-0 small">
            Don't have an account? <a href="{{ url_for('register', next=request.args.get('next')) }}" class="fw-bold text-decoration-none">Sign up now</a>
        </p>
    </div>
</div>
{% endblock %}
"""

REGISTER_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Register - BrieflyAI{% endblock %}
{% block body_class %}body-auth{% endblock %}

{% block content %}
<div class="auth-card animate-fade-in">
     <div class="auth-header">
        <div class="brand-icon"><i class="fas fa-user-plus"></i></div>
        <h2>Create Your Account</h2>
    </div>
    <div class="auth-body">
        <form method="POST" action="{{ url_for('register') }}">
             <div class="mb-3">
                <label for="name" class="form-label fw-medium">Full Name</label>
                <div class="input-group-icon">
                    <i class="fas fa-id-card input-icon"></i>
                    <input type="text" class="form-control" id="name" name="name" required placeholder="e.g. John Doe">
                </div>
            </div>
            <div class="mb-3">
                <label for="username" class="form-label fw-medium">Username</label>
                <div class="input-group-icon">
                    <i class="fas fa-user input-icon"></i>
                    <input type="text" class="form-control" id="username" name="username" required placeholder="e.g. johndoe (min 3 chars)">
                </div>
            </div>
            <div class="mb-4">
                <label for="password" class="form-label fw-medium">Password</label>
                <div class="input-group-icon">
                    <i class="fas fa-lock input-icon"></i>
                    <input type="password" class="form-control" id="password" name="password" required placeholder="min 6 chars">
                </div>
            </div>
            <button type="submit" class="btn btn-primary w-100">Create Account</button>
        </form>
    </div>
    <div class="auth-footer">
        <p class="mb-0 small">
            Already have an account? <a href="{{ url_for('login') }}" class="fw-bold text-decoration-none">Sign In</a>
        </p>
    </div>
</div>
{% endblock %}
"""

PROFILE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ user.name }}'s Profile - BrieflyAI{% endblock %}
{% block content %}
<div class="profile-header-card animate-fade-in">
    <div class="profile-avatar-wrapper">
        <div class="profile-avatar">{{ user.name[0]|upper }}</div>
    </div>
    <h2>{{ user.name }}</h2>
    <p class="username">@{{ user.username }}</p>
    <p class="small text-muted mb-0">Joined: {{ user.created_at | to_ist }}</p>
    <div class="profile-stats">
        <div class="stat-item">
            <div class="icon"><i class="fas fa-feather-pointed"></i></div>
            <div class="count">{{ posted_articles|length }}</div>
            <div class="label">Articles Posted</div>
        </div>
        <div class="stat-item">
            <div class="icon"><i class="fas fa-bookmark"></i></div>
            <div class="count">{{ bookmarks_pagination.total if bookmarks_pagination else 0 }}</div>
            <div class="label">Bookmarks</div>
        </div>
    </div>
</div>

<div class="mt-4 animate-fade-in" style="animation-delay: 0.1s;">
    <ul class="nav nav-tabs profile-tabs nav-fill mb-4" id="profileTab" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="bookmarks-tab" data-bs-toggle="tab" data-bs-target="#bookmarks-content" type="button" role="tab" aria-controls="bookmarks-content" aria-selected="true"><i class="fas fa-bookmark me-2"></i>My Bookmarks</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="posted-tab" data-bs-toggle="tab" data-bs-target="#posted-content" type="button" role="tab" aria-controls="posted-content" aria-selected="false"><i class="fas fa-feather-alt me-2"></i>My Articles</button>
        </li>
    </ul>
    <div class="tab-content" id="profileTabContent">
        <div class="tab-pane fade show active" id="bookmarks-content" role="tabpanel" aria-labelledby="bookmarks-tab">
            {% if bookmarked_articles %}
            <div class="row g-4">
                {% for art in bookmarked_articles %}
                <div class="col-md-6 col-lg-4 d-flex">
                    <article class="article-card d-flex flex-column w-100">
                        <div class="article-image-container">
                            <a href="{{ art.article_url }}"><img src="{{ art.urlToImage if art.urlToImage else 'https://via.placeholder.com/400x220/EEEEEE/AAAAAA?text=No+Image' }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
                            {% if art.is_stale_bookmark %}<span class="badge bg-secondary position-absolute top-0 end-0 m-2">Cached Bookmark</span>{% endif %}
                        </div>
                        <div class="article-body d-flex flex-column">
                            <h5 class="article-title mb-2"><a href="{{ art.article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                            <div class="article-meta small mb-2">
                                <span class="meta-item text-muted"><i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}"></i> {{ art.source.name|truncate(20) }}</span>
                                <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                            </div>
                            <p class="article-description small">{{ art.description|truncate(100) }}</p>
                            <a href="{{ art.article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                        </div>
                    </article>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <div class="empty-state-card">
                    <div class="icon"><i class="fas fa-bookmark"></i></div>
                    <h4>No Bookmarks Yet</h4>
                    <p class="text-muted">You haven't bookmarked any articles. Find an article you like and click the bookmark icon to save it!</p>
                </div>
            {% endif %}

            {% if bookmarks_pagination and bookmarks_pagination.pages > 1 %}
            <nav aria-label="Bookmarks navigation" class="mt-5">
                <ul class="pagination justify-content-center">
                    <li class="page-item page-link-prev-next {% if not bookmarks_pagination.has_prev %}disabled{% endif %}"><a class="page-link" href="{{ url_for('profile', page=bookmarks_pagination.prev_num) if bookmarks_pagination.has_prev else '#' }}">&laquo; Prev</a></li>
                    {% for p in bookmarks_pagination.iter_pages(left_edge=1, right_edge=1, left_current=1, right_current=2) %}{% if p %}{% if p == bookmarks_pagination.page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% else %}<li class="page-item"><a class="page-link" href="{{ url_for('profile', page=p) }}">{{ p }}</a></li>{% endif %}{% else %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}{% endfor %}
                    <li class="page-item page-link-prev-next {% if not bookmarks_pagination.has_next %}disabled{% endif %}"><a class="page-link" href="{{ url_for('profile', page=bookmarks_pagination.next_num) if bookmarks_pagination.has_next else '#' }}">Next &raquo;</a></li>
                </ul>
            </nav>
            {% endif %}
        </div>
        <div class="tab-pane fade" id="posted-content" role="tabpanel" aria-labelledby="posted-tab">
            {% if posted_articles %}
            <div class="row g-4">
                {% for art in posted_articles %}
                <div class="col-md-6 col-lg-4 d-flex">
                    <article class="article-card d-flex flex-column w-100">
                        {% set article_url = url_for('article_detail', article_hash_id=art.article_hash_id) %}
                        <div class="article-image-container"><a href="{{ article_url }}"><img src="{{ art.image_url if art.image_url else 'https://via.placeholder.com/400x220/EEEEEE/AAAAAA?text=No+Image' }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a></div>
                        <div class="article-body d-flex flex-column">
                            <h5 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                            <div class="article-meta small mb-2">
                                <span class="meta-item text-muted"><i class="fas fa-user-edit"></i> {{ art.author.name|truncate(20) }}</span>
                                <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ art.published_at | to_ist }}</span>
                            </div>
                            <p class="article-description small">{{ art.description|truncate(100) }}</p>
                            <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                        </div>
                    </article>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <div class="empty-state-card">
                    <div class="icon"><i class="fas fa-feather-alt"></i></div>
                    <h4>Nothing Posted Yet</h4>
                    <p class="text-muted">You haven't posted any articles. Click the <i class="fas fa-pen-to-square"></i> button to share your first story!</p>
                </div>
            {% endif %}
        </div>
    </div>
</div>
{% endblock %}
"""

PUBLIC_PROFILE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ user.name }}'s Profile - BrieflyAI{% endblock %}
{% block content %}
<div class="animate-fade-in">
    <div class="profile-header-card mb-4">
        <div class="profile-avatar-wrapper">
            <div class="profile-avatar">{{ user.name[0]|upper }}</div>
        </div>
        <h2>{{ user.name }}</h2>
        <p class="username">@{{ user.username }}</p>
        <p class="small text-muted mb-0">Member Since: {{ user.created_at | to_ist }}</p>
    </div>

    <h3 class="mt-5 mb-4">Articles by {{ user.name }} ({{ posted_articles|length }})</h3>
    
    <div class="row g-4">
    {% if posted_articles %}
        {% for art in posted_articles %}
        <div class="col-md-6 col-lg-4 d-flex">
            <article class="article-card d-flex flex-column w-100">
                {% set article_url = url_for('article_detail', article_hash_id=art.article_hash_id) %}
                <div class="article-image-container"><a href="{{ article_url }}"><img src="{{ art.image_url if art.image_url else 'https://via.placeholder.com/400x220/EEEEEE/AAAAAA?text=No+Image' }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a></div>
                <div class="article-body d-flex flex-column">
                    <h5 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                    <div class="article-meta small mb-2">
                        <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ art.published_at | to_ist }}</span>
                    </div>
                    <p class="article-description small">{{ art.description|truncate(100) }}</p>
                    <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                </div>
            </article>
        </div>
        {% endfor %}
    {% else %}
        <div class="col-12">
            <div class="empty-state-card">
                <div class="icon"><i class="fas fa-feather-alt"></i></div>
                <h4>No Articles Yet</h4>
                <p class="text-muted">{{ user.name }} hasn't posted any articles yet.</p>
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

        <h2><i class="icon fas fa-bullseye"></i>Our Mission</h2>
        <p>
            In a world of information overload, our mission is to provide clarity and efficiency. We believe that everyone deserves access to accurate, unbiased news without spending hours sifting through lengthy articles. BrieflyAI cuts through the noise, offering insightful summaries that matter.
        </p>

        <h2><i class="icon fas fa-users"></i>Community Hub</h2>
        <p>
            Beyond AI-driven news, BrieflyAI is a platform for discussion and community engagement. Our Community Hub allows users to post their own articles, share perspectives, and engage in meaningful conversations about the topics that shape our world. We are committed to fostering a respectful and intelligent environment for all our members.
        </p>

        <h2><i class="icon fas fa-microchip"></i>Our Technology</h2>
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
                    <div class="icon"><i class="fas fa-envelope"></i></div>
                    <h4 class="h5">General Inquiries</h4>
                    <p class="text-muted">For general questions, feedback, or support, please email us at:</p>
                    <a href="mailto:vbansal639@gmail.com" class="fw-bold">vbansal639@gmail.com</a>
                </div>
            </div>
            <div class="col-md-6">
                <div class="contact-card">
                    <div class="icon"><i class="fas fa-handshake"></i></div>
                    <h4 class="h5">Partnerships & Media</h4>
                    <p class="text-muted">For partnership opportunities or media inquiries, please contact us at:</p>
                    <a href="mailto:vbansal639@gmail.com" class="fw-bold">vbansal639@gmail.com</a>
                </div>
            </div>
        </div>

        <div class="text-center mt-5">
            <h2 class="h3">Follow Us</h2>
            <p class="text-muted">Stay connected with us on social media.</p>
            <div class="contact-social-links mt-3">
                <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a>
                <a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a>
                <a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a>
                <a href="#" title="Instagram"><i class="fab fa-instagram"></i></a>
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
        
        <h2><i class="icon fas fa-shield-halved"></i>1. Information We Collect</h2>
        <p>We may collect personal information that you voluntarily provide to us when you register on the website, post articles or comments, bookmark articles, or subscribe to our newsletter. This information may include your name, username, email address, and your activities on our platform such as articles posted and bookmarked.</p>
        
        <h2><i class="icon fas fa-tasks"></i>2. How We Use Your Information</h2>
        <p>We use the information we collect to:</p>
        <ul>
            <li>Create and manage your account.</li>
            <li>Operate and maintain the website, including your profile page.</li>
            <li>Display your posted and bookmarked articles as part of your profile.</li>
            <li>Send you newsletters or promotional materials, if you have opted in.</li>
            <li>Respond to your comments and inquiries.</li>
            <li>Improve our website and services.</li>
        </ul>

        <h2><i class="icon fas fa-share-nodes"></i>3. Disclosure of Your Information</h2>
        <p>Your username and posted articles are publicly visible. Your bookmarked articles are visible on your profile page to you when logged in. We do not sell, trade, or otherwise transfer your personally identifiable information like your email address to outside parties without your consent, except to trusted third parties who assist us in operating our website, so long as those parties agree to keep this information confidential.</p>
        
        <h2><i class="icon fas fa-lock"></i>4. Security of Your Information</h2>
        <p>We use administrative, technical, and physical security measures to help protect your personal information. While we have taken reasonable steps to secure the personal information you provide to us, please be aware that despite our efforts, no security measures are perfect or impenetrable.</p>
        
        <h2><i class="icon fas fa-edit"></i>5. Your Choices</h2>
        <p>You can review and change your profile information by logging into your account. You may also request deletion of your account and associated data by contacting us.</p>
        
        <h2><i class="icon fas fa-sync-alt"></i>6. Changes to This Privacy Policy</h2>
        <p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page. You are advised to review this Privacy Policy periodically for any changes.</p>
    </div>
</div>
{% endblock %}
"""

ERROR_404_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}404 Not Found{% endblock %}{% block content %}<div class='text-center my-5 p-4 article-card animate-fade-in mx-auto' style='max-width: 600px;'><h1><i class='fas fa-exclamation-triangle text-warning me-2'></i>404 - Page Not Found</h1><p class='lead'>Sorry, the page you are looking for does not exist or has been moved.</p><a href='{{url_for("index")}}' class='btn btn-primary-modal mt-2'>Go to Homepage</a></div>{% endblock %}"""
ERROR_500_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}500 Server Error{% endblock %}{% block content %}<div class='text-center my-5 p-4 article-card animate-fade-in mx-auto' style='max-width: 600px;'><h1><i class='fas fa-cogs text-danger me-2'></i>500 - Internal Server Error</h1><p class='lead'>Something went wrong on our end. We've been notified and are looking into it.</p><a href='{{url_for("index")}}' class='btn btn-primary-modal mt-2'>Go to Homepage</a></div>{% endblock %}"""

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
