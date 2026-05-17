 
import json
import re
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from urllib.parse import quote, urlparse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator, HttpUrl
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.types import Tool, TextContent
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import StreamingResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

FACEBOOK_MAX_LENGTH = 63206
LINKEDIN_HEADLINE_MAX = 200
X_MAX_LENGTH = 280
INSTAGRAM_MAX_LENGTH = 2200
DISCORD_DESCRIPTION_MAX = 2048
DISCORD_TITLE_MAX = 256
DISCORD_DEFAULT_COLOR = 3447003

# HTML Parser config
HTML_PARSER_CONFIG = {
    "features": "html.parser",
    "from_encoding": "utf-8"
}

# Supported HTML parser types
class ParserType(str, Enum):
    TMIT = "tmit"
    VIK = "vik"
    BME_NEWS = "bme_news"
    BME_EVENT = "bme_event"
    SIMPLE_EVENT = "simple_event"

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class EventType(str, Enum):
    LECTURE = "lecture"
    WORKSHOP = "workshop"
    CONFERENCE = "conference"
    DEADLINE = "deadline"
    CONCERT = "concert"
    DOCTORAL_DEFENSE = "doctoral_defense"
    OTHER = "other"

class PlatformType(str, Enum):
    """Supported social media platforms"""
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    X = "x"
    INSTAGRAM = "instagram"
    DISCORD = "discord"

class HTMLParseRequest(BaseModel):
    """Validated HTML parsing request"""
    html_content: str = Field(..., min_length=1, description="HTML tartalom")
    source_url: HttpUrl = Field(..., description="Érvényes URL cím")
    
    @field_validator("html_content")
    @classmethod
    def validate_html_not_empty(cls, v: str) -> str:
        """Ensure HTML content is not just whitespace"""
        if not v or not v.strip():
            raise ValueError("HTML tartalom nem lehet üres")
        return v

class EventDetected(BaseModel):
    """Detektált esemény"""
    title: str = Field(..., description="Esemény címe")
    date: str = Field(..., description="Esemény dátuma (YYYY-MM-DD HH:MM formátumban)")
    location: Optional[str] = Field(None, description="Esemény helyszíne")
    event_type: EventType = Field(default=EventType.OTHER)
    description: str = Field(..., description="Esemény leírása")
    registration_url: Optional[str] = Field(None, description="Regisztrációs URL")
    source_url: Optional[str] = Field(None, description="Forrás URL")

class NewsItem(BaseModel):
    """Kinyert hírelemek"""
    title: str
    content: str
    image_url: Optional[str] = None
    source_url: str
    publish_date: Optional[str] = None
    events: List[EventDetected] = Field(default_factory=list)

class SocialPostFacebook(BaseModel):
    content: str = Field(..., description="Facebook poszt szövege (max 63.206 karakter)")
    hashtags: List[str] = Field(default_factory=list)
    image_description: Optional[str] = None
    cta_button: Optional[str] = Field(None, description="Call-to-action gomb szövege")
    cta_url: Optional[str] = None

class SocialPostLinkedIn(BaseModel):
    headline: str = Field(..., description="LinkedIn headline (max 200 karakter)")
    body: str = Field(..., description="LinkedIn poszt szövege")
    hashtags: List[str] = Field(default_factory=list)
    cta_text: str = Field(default="Tudj meg többet", description="Call-to-action szöveg")
    cta_url: Optional[str] = None

class SocialPostX(BaseModel):
    content: str = Field(..., description="X poszt (max 280 karakter)")
    hashtags: List[str] = Field(default_factory=list)

class SocialPostInstagram(BaseModel):
    caption: str = Field(..., description="Instagram felirat (max 2.200 karakter)")
    hashtags: List[str] = Field(default_factory=list)
    emoji_usage: Optional[str] = None

class SocialPostDiscord(BaseModel):
    embed_title: str
    embed_description: str
    embed_fields: Dict[str, str] = Field(default_factory=dict)
    embed_color: int = Field(default=3447003, description="Discord embed szín (hex)")

class SocialPostsResponse(BaseModel):
    """Összes platform posztja"""
    facebook: Optional[SocialPostFacebook] = None
    linkedin: Optional[SocialPostLinkedIn] = None
    x: Optional[SocialPostX] = None
    instagram: Optional[SocialPostInstagram] = None
    discord: Optional[SocialPostDiscord] = None

class NewsAnalysisResponse(BaseModel):
    """Teljes hírelemzés eredmény"""
    news_items: List[NewsItem]
    total_events_detected: int
    total_posts_generated: int

class ToolCallRequest(BaseModel):
    """MCP Tool hívás request"""
    name: str = Field(..., description="Tool neve")
    arguments: Dict[str, Any] = Field(..., description="Tool paraméterei")

class ToolCallResponse(BaseModel):
    """MCP Tool response"""
    status: str = Field(default="success")
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ============================================================================
# HELPER FUNCTIONS - Common parsing utilities
# ============================================================================

def safe_find_text(element, *args, **kwargs) -> Optional[str]:
    """Safely extract and strip text from BeautifulSoup element"""
    if element is None:
        return None
    try:
        found = element.find(*args, **kwargs)
        if found:
            text = found.get_text(strip=True)
            return text if text else None
        return None
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error extracting text: {e}")
        return None

def safe_get_attr(element, attr: str) -> Optional[str]:
    """Safely extract attribute from BeautifulSoup element"""
    if element is None:
        return None
    try:
        return element.get(attr)
    except (AttributeError, TypeError) as e:
        logger.warning(f"Error getting attribute {attr}: {e}")
        return None

def validate_url_safety(url: Optional[str]) -> Optional[str]:
    """Validate and sanitize URL to prevent XSS"""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            logger.warning(f"Invalid URL scheme: {parsed.scheme}")
            return None
        return url
    except Exception as e:
        logger.warning(f"URL validation failed: {e}")
        return None

# Pre-compiled regex patterns
REGEX_HUNGARIAN_DATE = re.compile(r'(\d{4})\D+(\d{1,2})\D+(\d{1,2})')
REGEX_TIME = re.compile(r'(\d{1,2}):(\d{2})')
REGEX_LOCATION = re.compile(r'(?:helyszín|location|hely)[\s:]*([^,.\n]+)', re.IGNORECASE)

# ============================================================================
# HTML PARSER UTILS
# ============================================================================

def parse_tmit_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """TMIT (node-hir) típusú hír parsése"""
    try:
        soup = BeautifulSoup(html_content, **HTML_PARSER_CONFIG)
        article = soup.find('article', class_='node-hir')
        
        if not article:
            logger.debug("TMIT article not found")
            return None
        
        title = safe_find_text(article, 'h2', class_='node-title') or "N/A"
        image_elem = article.find('img')
        image_url = validate_url_safety(safe_get_attr(image_elem, 'src'))
        content = safe_find_text(article, 'div', attrs={'property': 'content:encoded'}) or ""
        
        return NewsItem(
            title=title,
            content=content,
            image_url=image_url,
            source_url=str(source_url)
        )
    except Exception as e:
        logger.error(f"Error parsing TMIT news: {e}")
        return None

def parse_vik_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """VIK (news-title-important) típusú hír parsése"""
    try:
        soup = BeautifulSoup(html_content, **HTML_PARSER_CONFIG)
        title_elem = soup.find('h2', class_='news-title-important')
        
        if not title_elem:
            logger.debug("VIK title not found")
            return None
        
        title = title_elem.get_text(strip=True) or "N/A"
        publish_date = safe_find_text(soup, 'span', class_='news-date')
        content = safe_find_text(soup, 'div', class_='news-excerpt') or ""
        image_elem = soup.find('img', class_='news-image')
        image_url = validate_url_safety(safe_get_attr(image_elem, 'src'))
        
        return NewsItem(
            title=title,
            content=content,
            image_url=image_url,
            source_url=str(source_url),
            publish_date=publish_date
        )
    except Exception as e:
        logger.error(f"Error parsing VIK news: {e}")
        return None

def parse_bme_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """BME (bme_news_card) típusú hír parsése"""
    try:
        soup = BeautifulSoup(html_content, **HTML_PARSER_CONFIG)
        news_card = soup.find('div', class_='bme_news_card')
        
        if not news_card:
            logger.debug("BME news card not found")
            return None
        
        title = safe_find_text(news_card, 'h4', class_='bme_news_card-title') or "N/A"
        publish_date = safe_find_text(news_card, 'span', class_='field--name-created')
        content = safe_find_text(news_card, 'div', class_='bme_news_card-body') or ""
        image_elem = news_card.find('img')
        image_url = validate_url_safety(safe_get_attr(image_elem, 'src'))
        
        return NewsItem(
            title=title,
            content=content,
            image_url=image_url,
            source_url=str(source_url),
            publish_date=publish_date
        )
    except Exception as e:
        logger.error(f"Error parsing BME news: {e}")
        return None

def parse_bme_event(html_content: str, source_url: str) -> Optional[EventDetected]:
    """BME event card parsése"""
    try:
        soup = BeautifulSoup(html_content, **HTML_PARSER_CONFIG)
        
        title = safe_find_text(soup, 'h4', class_='bme_event_card-title') or "N/A"
        date_elem = soup.find('div', class_='bme_event_card-date')
        date_str = safe_find_text(date_elem, 'span', class_='nowrap') or ""
        location = safe_find_text(soup, 'p', class_='bme_event_card-location')
        description = safe_find_text(soup, 'div', class_='bme_event_card-body') or ""
        
        event_type = detect_event_type(title)
        
        return EventDetected(
            title=title,
            date=parse_date_string(date_str),
            location=location,
            event_type=event_type,
            description=description,
            source_url=str(source_url)
        )
    except Exception as e:
        logger.error(f"Error parsing BME event: {e}")
        return None

def parse_simple_event(html_content: str, source_url: str) -> Optional[EventDetected]:
    """Egyszerű event formátum parsése (VIK)"""
    try:
        soup = BeautifulSoup(html_content, **HTML_PARSER_CONFIG)
        date_str = safe_find_text(soup, 'span', class_='event-date') or ""
        title = safe_find_text(soup, 'a', class_='event-title') or "N/A"
        
        return EventDetected(
            title=title,
            date=parse_date_string(date_str),
            event_type=EventType.OTHER,
            description="",
            source_url=str(source_url)
        )
    except Exception as e:
        logger.error(f"Error parsing simple event: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_date_string(date_str: str) -> str:
    """Magyar dátum stringet ISO formátumra alakít - ROBUST ERROR HANDLING"""
    if not date_str or not isinstance(date_str, str):
        logger.warning(f"Invalid date_str: {date_str}")
        return datetime.now().isoformat()
    
    try:
        # Dátum és idő extraktálása regex-szel
        date_match = REGEX_HUNGARIAN_DATE.search(date_str)
        if not date_match:
            logger.debug(f"No date pattern found in: {date_str}")
            return datetime.now().isoformat()
        
        year, month, day = date_match.groups()
        
        # Idő keresése
        time_match = REGEX_TIME.search(date_str)
        if time_match:
            hour, minute = time_match.groups()
            result = f"{year}-{month.zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}:{minute}:00"
        else:
            result = f"{year}-{month.zfill(2)}-{day.zfill(2)}T00:00:00"
        
        # Validate the resulting ISO date
        datetime.fromisoformat(result)
        return result
        
    except (ValueError, AttributeError) as e:
        logger.warning(f"Date parsing validation failed for '{date_str}': {e}")
        return datetime.now().isoformat()
    except Exception as e:
        logger.error(f"Unexpected error parsing date '{date_str}': {e}")
        return datetime.now().isoformat()

def detect_event_type(title: str) -> EventType:
    """Eseménytípus detektálása a címből - TYPE SAFE"""
    if not title or not isinstance(title, str):
        return EventType.OTHER
    
    try:
        title_lower = title.lower()
        
        keywords_map = {
            EventType.DOCTORAL_DEFENSE: ['doktori', 'védés', 'phd'],
            EventType.CONCERT: ['koncert', 'zene', 'szimfónia'],
            EventType.CONFERENCE: ['konferencia', 'symposium'],
            EventType.WORKSHOP: ['workshop', 'tanfolyam', 'képzés'],
            EventType.LECTURE: ['előadás', 'lecture', 'diasor'],
            EventType.DEADLINE: ['határidő', 'deadline', 'pályázat'],
        }
        
        for event_type, keywords in keywords_map.items():
            if any(keyword in title_lower for keyword in keywords):
                return event_type
        
        return EventType.OTHER
        
    except Exception as e:
        logger.error(f"Error detecting event type: {e}")
        return EventType.OTHER

# ============================================================================
# MCP SERVER SETUP
# ============================================================================

server = Server("news-to-social-agent")

@server.list_tools()
async def list_mcp_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="parse_html_and_extract_news",
            description="HTML feldolgozás és hírek + események extraktálása. Támogatott formátumok: TMIT (node-hir), VIK (news-title-important), BME (bme_news_card), BME events",
            inputSchema={
                "type": "object",
                "properties": {
                    "html_content": {"type": "string", "description": "HTML tartalom"},
                    "source_url": {"type": "string", "description": "Forrás URL"}
                },
                "required": ["html_content", "source_url"]
            }
        ),
        Tool(
            name="detect_events_from_content",
            description="Eseményadatok detektálása szövegből regex alapú keresés",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Szöveg tartalom"},
                    "current_date": {"type": "string", "description": "Aktuális dátum (ISO format, opcional)"}
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="generate_social_posts",
            description="Platform-specifikus szociálmédiai posztok generálása (Facebook, LinkedIn, X, Instagram, Discord)",
            inputSchema={
                "type": "object",
                "properties": {
                    "news_title": {"type": "string"},
                    "news_content": {"type": "string"},
                    "source_url": {"type": "string"},
                    "events": {"type": "array", "description": "Eseményadatok (opcional)"},
                    "platforms": {"type": "array", "items": {"type": "string"}, "description": "Platform nevek (opcional, default: mind)"}
                },
                "required": ["news_title", "news_content", "source_url"]
            }
        ),
        Tool(
            name="enrich_with_registration_link",
            description="Regisztrációs link injektálása az eseményt tartalmazó posztokba",
            inputSchema={
                "type": "object",
                "properties": {
                    "post": {"type": "object", "description": "Poszt objektum"},
                    "event": {"type": "object", "description": "Event objektum"},
                    "platform": {"type": "string", "description": "Platform neve"}
                },
                "required": ["post", "event", "platform"]
            }
        )
    ]

@server.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> dict:
    """
    Handle MCP tool calls with proper error handling.
    
    Raises:
    - ValueError: Ha a tool nem létezik vagy arguments invalid
    - Specific exceptions logged with context
    """
    try:
        # Input validation
        if not name or not isinstance(name, str):
            logger.error("Invalid tool name: must be non-empty string")
            raise ValueError("Invalid tool name")
        
        if not isinstance(arguments, dict):
            logger.error(f"Invalid arguments: must be dict, got {type(arguments)}")
            raise ValueError("Invalid arguments format")
        
        logger.debug(f"Calling tool: {name} with args keys: {list(arguments.keys())}")
        
        try:
            if name == "parse_html_and_extract_news":
                result = await parse_html_and_extract_news(
                    arguments.get("html_content", ""),
                    arguments.get("source_url", "")
                )
            elif name == "detect_events_from_content":
                result = await detect_events_from_content(
                    arguments.get("content", ""),
                    arguments.get("current_date")
                )
            elif name == "generate_social_posts":
                result = await generate_social_posts(
                    arguments.get("news_title", ""),
                    arguments.get("news_content", ""),
                    arguments.get("source_url", ""),
                    arguments.get("events"),
                    arguments.get("platforms")
                )
            elif name == "enrich_with_registration_link":
                result = await enrich_with_registration_link(
                    arguments.get("post", {}),
                    arguments.get("event", {}),
                    arguments.get("platform", "")
                )
            else:
                logger.warning(f"Unknown tool requested: {name}")
                raise ValueError(f"Unknown tool: {name}")
            
            logger.info(f"Tool {name} executed successfully")
            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
            }
        
        except ValueError as e:
            logger.error(f"Validation error in tool {name}: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing argument in tool {name}: {e}")
            raise ValueError(f"Missing required argument: {str(e)}")
        except TypeError as e:
            logger.error(f"Type error in tool {name}: {e}")
            raise ValueError(f"Type error: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error in tool {name}: {e}")
            raise
    
    except ValueError as e:
        logger.error(f"ValueError calling tool {name}: {e}")
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Validation error: {str(e)}"}]
        }
    except Exception as e:
        logger.exception(f"Unexpected error calling tool {name}: {e}")
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Server error: {str(e)}"}]
        }

async def parse_html_and_extract_news(html_content: str, source_url: str) -> Dict[str, Any]:
    """
    HTML feldolgozás és hírek + események extraktálása.
    
    Támogatott formátumok:
    - TMIT: <article class="node-hir">
    - VIK: <h2 class="news-title-important">
    - BME: <div class="bme_news_card">
    - Events: BME event cards, simple events
    
    Raises:
    - ValueError: Ha az input nem megfelelő
    - Exception: Az egy-egy parser-specifikus hibákat loggol, nem dobja
    """
    try:
        # Input validation
        if not html_content or not html_content.strip():
            logger.warning("Empty HTML content provided")
            return {
                "news_count": 0,
                "event_count": 0,
                "news_items": [],
                "events": [],
                "status": "error",
                "error": "HTML tartalom nem lehet üres"
            }
        
        if not source_url:
            logger.warning("Missing source URL")
            return {
                "news_count": 0,
                "event_count": 0,
                "news_items": [],
                "events": [],
                "status": "error",
                "error": "Source URL szükséges"
            }
        
        soup = BeautifulSoup(html_content, **HTML_PARSER_CONFIG)
        news_items = []
        events = []
        
        # TMIT hírek
        try:
            tmit_articles = soup.find_all('article', class_='node-hir')
            for article in tmit_articles:
                html_str = str(article)
                news = parse_tmit_news(html_str, source_url)
                if news:
                    news_items.append(news.model_dump())
            logger.info(f"Found {len(tmit_articles)} TMIT articles")
        except Exception as e:
            logger.error(f"Error parsing TMIT articles: {e}")
        
        # VIK hírek
        try:
            vik_news = soup.find_all('h2', class_='news-title-important')
            for news_elem in vik_news:
                parent_div = news_elem.find_parent('div', class_='news-item')
                if parent_div:
                    html_str = str(parent_div)
                    news = parse_vik_news(html_str, source_url)
                    if news:
                        news_items.append(news.model_dump())
            logger.info(f"Found {len(vik_news)} VIK news items")
        except Exception as e:
            logger.error(f"Error parsing VIK news: {e}")
        
        # BME hírek
        try:
            bme_news = soup.find_all('div', class_='bme_news_card')
            for news_card in bme_news:
                html_str = str(news_card)
                news = parse_bme_news(html_str, source_url)
                if news:
                    news_items.append(news.model_dump())
            logger.info(f"Found {len(bme_news)} BME news cards")
        except Exception as e:
            logger.error(f"Error parsing BME news: {e}")
        
        # BME események
        try:
            bme_events = soup.find_all('div', class_='bme_event_card-date')
            for event_card in bme_events:
                parent = event_card.find_parent('div', class_='px-5')
                if parent:
                    html_str = str(parent)
                    event = parse_bme_event(html_str, source_url)
                    if event:
                        events.append(event.model_dump())
            logger.info(f"Found {len(bme_events)} BME events")
        except Exception as e:
            logger.error(f"Error parsing BME events: {e}")
        
        # Egyszerű események
        try:
            simple_events = soup.find_all('div', class_='event')
            for event_elem in simple_events:
                html_str = str(event_elem)
                event = parse_simple_event(html_str, source_url)
                if event:
                    events.append(event.model_dump())
            logger.info(f"Found {len(simple_events)} simple events")
        except Exception as e:
            logger.error(f"Error parsing simple events: {e}")
        
        logger.info(f"Total extracted: {len(news_items)} news, {len(events)} events")
        return {
            "news_count": len(news_items),
            "event_count": len(events),
            "news_items": news_items,
            "events": events,
            "status": "success"
        }
    
    except ValueError as e:
        logger.error(f"Validation error in parse_html_and_extract_news: {e}")
        return {
            "status": "error",
            "error": f"Validation error: {str(e)}",
            "news_items": [],
            "events": [],
            "news_count": 0,
            "event_count": 0
        }
    except Exception as e:
        logger.exception(f"Unexpected error in parse_html_and_extract_news: {e}")
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "news_items": [],
            "events": [],
            "news_count": 0,
            "event_count": 0
        }

async def detect_events_from_content(content: str, current_date: str = None) -> Dict[str, Any]:
    """
    LLM segítségével részletesebb eseményadatok detektálása szövegből.
    
    Kimenete: JSON lista EventDetected sémával.
    
    Raises:
    - ValueError: Ha input nem megfelelő
    """
    try:
        # Input validation
        if not content or not isinstance(content, str):
            logger.warning("Invalid content for event detection")
            raise ValueError("Content must be non-empty string")
        
        if not current_date:
            current_date = datetime.now().isoformat()
        else:
            # Validate ISO format if provided
            try:
                datetime.fromisoformat(current_date)
            except ValueError as e:
                logger.warning(f"Invalid date format: {current_date}")
                raise ValueError(f"Invalid date format: {str(e)}")
        
        # Regex alapú detektálás
        try:
            dates_found = re.findall(REGEX_HUNGARIAN_DATE.pattern, content)
            locations_found = re.findall(REGEX_LOCATION.pattern, content)
            
            logger.info(f"Found {len(dates_found)} dates and {len(locations_found)} locations")
            
            return {
                "status": "success",
                "dates_found": dates_found[:10],  # Limit to 10
                "locations_found": locations_found[:10],  # Limit to 10
                "message": "Content processed. Use with LLM for full event extraction.",
                "current_date": current_date
            }
        except re.error as e:
            logger.error(f"Regex error in event detection: {e}")
            raise ValueError(f"Regex parsing error: {str(e)}")
    
    except ValueError as e:
        logger.error(f"Validation error in detect_events_from_content: {e}")
        return {
            "status": "error",
            "error": f"Validation error: {str(e)}",
            "dates_found": [],
            "locations_found": []
        }
    except Exception as e:
        logger.exception(f"Unexpected error in detect_events_from_content: {e}")
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "dates_found": [],
            "locations_found": []
        }

async def generate_social_posts(
    news_title: str,
    news_content: str,
    source_url: str,
    events: List[Dict[str, Any]] = None,
    platforms: List[str] = None
) -> Dict[str, Any]:
    """
    Platform-specifikus szociálmédiai posztok generálása.
    
    Az n8n Mistral LLM-ét hívja meg a konkrét szöveg generálásához,
    ez a tool csak a Pydantic szerkezeteket validálja.
    
    platforms: ["facebook", "linkedin", "x", "instagram", "discord"]
    
    Raises:
    - ValueError: Ha input nem megfelelő
    """
    try:
        # Input validation
        if not news_title or not isinstance(news_title, str):
            logger.warning("Invalid news_title for post generation")
            raise ValueError("news_title must be non-empty string")
        
        if not news_content or not isinstance(news_content, str):
            logger.warning("Invalid news_content for post generation")
            raise ValueError("news_content must be non-empty string")
        
        if not source_url or not isinstance(source_url, str):
            logger.warning("Invalid source_url for post generation")
            raise ValueError("source_url must be non-empty string")
        
        # Validate URL
        source_url = validate_url_safety(source_url)
        if not source_url:
            logger.warning("Source URL failed validation")
            raise ValueError("Invalid source URL")
        
        if not platforms:
            platforms = ["facebook", "linkedin", "x", "instagram", "discord"]
        
        if not events:
            events = []
        
        # Validate platform names
        valid_platforms = {p.value for p in PlatformType}
        for platform in platforms:
            if platform not in valid_platforms:
                logger.warning(f"Invalid platform: {platform}")
                raise ValueError(f"Invalid platform: {platform}")
        
        # Alapértelmezett template-ek a posztokhoz
        templates = {
            "facebook": f"""
Kedves közösség! 🎓

{news_title}

{news_content}

Tudj meg többet: {source_url}
""",
            "linkedin": f"""
{news_title}

{news_content}

📌 Tudj meg többet:
{source_url}

#BME #Hírek #Oktatás
""",
            "x": f"{news_title}\n\n{news_content[:200]}...\n\n{source_url}",
            "instagram": f"{news_title}\n.\n{news_content}\n\n#BME #Egyetem #Hírek",
            "discord": {
                "title": news_title,
                "description": news_content,
                "url": source_url
            }
        }
        
        result = {}
        
        try:
            if "facebook" in platforms:
                result["facebook"] = {
                    "content": templates["facebook"][:FACEBOOK_MAX_LENGTH],
                    "hashtags": ["#BME", "#Hírek"],
                    "cta_button": "Tudj meg többet" if events else None,
                    "cta_url": events[0].get("registration_url") if events else None
                }
            
            if "linkedin" in platforms:
                result["linkedin"] = {
                    "headline": news_title[:LINKEDIN_HEADLINE_MAX],
                    "body": templates["linkedin"],
                    "hashtags": ["BME", "Hírek", "Oktatás"],
                    "cta_text": "Regisztrálj",
                    "cta_url": events[0].get("registration_url") if events else source_url
                }
            
            if "x" in platforms:
                result["x"] = {
                    "content": templates["x"][:X_MAX_LENGTH],
                    "hashtags": ["BME", "Hírek"]
                }
            
            if "instagram" in platforms:
                result["instagram"] = {
                    "caption": templates["instagram"][:INSTAGRAM_MAX_LENGTH],
                    "hashtags": ["BME", "Egyetem", "Hírek", "Oktatás"],
                    "emoji_usage": "🎓📚🏫"
                }
            
            if "discord" in platforms:
                result["discord"] = {
                    "embed_title": news_title[:DISCORD_TITLE_MAX],
                    "embed_description": news_content[:DISCORD_DESCRIPTION_MAX],
                    "embed_fields": {
                        "Forrás": source_url,
                        "Típus": "Hír"
                    },
                    "embed_color": DISCORD_DEFAULT_COLOR
                }
            
            logger.info(f"Generated {len(result)} social posts for platforms: {list(result.keys())}")
            
            return {
                "status": "success",
                "posts": result,
                "platforms_generated": list(result.keys()),
                "event_count": len(events)
            }
        
        except KeyError as e:
            logger.error(f"Missing field in template generation: {e}")
            raise ValueError(f"Template generation error: {str(e)}")
    
    except ValueError as e:
        logger.error(f"Validation error in generate_social_posts: {e}")
        return {
            "status": "error",
            "error": f"Validation error: {str(e)}",
            "posts": {},
            "platforms_generated": [],
            "event_count": 0
        }
    except Exception as e:
        logger.exception(f"Unexpected error in generate_social_posts: {e}")
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "posts": {},
            "platforms_generated": [],
            "event_count": 0
        }

async def enrich_with_registration_link(
    post: Dict[str, Any],
    event: Dict[str, Any],
    platform: str
) -> Dict[str, Any]:
    """
    Regisztrációs link injektálása az eseményt tartalmazó posztokba.
    
    Raises:
    - ValueError: Ha input nem megfelelő
    """
    try:
        if not post or not isinstance(post, dict):
            logger.warning("Invalid post object for enrichment")
            raise ValueError("Post objektum szükséges")
        
        if not event or not isinstance(event, dict):
            logger.warning("Invalid event object for enrichment")
            raise ValueError("Event objektum szükséges")
        
        if not platform or platform not in [p.value for p in PlatformType]:
            logger.warning(f"Invalid platform: {platform}")
            raise ValueError(f"Érvénytelen platform: {platform}")
        
        registration_url = event.get("registration_url") or event.get("source_url")
        
        if not registration_url:
            logger.info("No registration URL found for event enrichment")
            return {
                "status": "warning",
                "message": "No registration URL found",
                "enriched_post": post
            }
        
        # Sanitize URL
        registration_url = validate_url_safety(registration_url)
        if not registration_url:
            logger.warning("Registration URL failed validation")
            return {
                "status": "warning",
                "message": "Registration URL failed validation",
                "enriched_post": post
            }
        
        enriched = post.copy()
        event_title = event.get("title", "").strip()
        event_date = event.get("date", "").strip()
        
        if platform == PlatformType.FACEBOOK.value:
            enriched["content"] += f"\n\n📅 {quote(event_title, safe='')} ({event_date})\n🔗 Regisztrálj: {registration_url}"
        
        elif platform == PlatformType.LINKEDIN.value:
            enriched["body"] += f"\n\n🎯 {quote(event_title, safe='')}\n🗓️ {event_date}\n\n👉 {registration_url}"
        
        elif platform == PlatformType.X.value:
            enriched["content"] = enriched.get("content", "")[:250] + f"\n🔗 {registration_url}"
        
        elif platform == PlatformType.INSTAGRAM.value:
            enriched["caption"] = enriched.get("caption", "") + f"\n\n📌 {quote(event_title, safe='')}\n{registration_url}"
        
        elif platform == PlatformType.DISCORD.value:
            if "embed_fields" not in enriched:
                enriched["embed_fields"] = {}
            enriched["embed_fields"]["Regisztráció"] = registration_url
            enriched["embed_fields"]["Dátum"] = event_date
        
        logger.info(f"Successfully enriched post for {platform}")
        return {
            "status": "success",
            "enriched_post": enriched,
            "platform": platform
        }
    
    except ValueError as e:
        logger.error(f"Validation error in enrich_with_registration_link: {e}")
        return {
            "status": "error",
            "error": f"Validation error: {str(e)}",
            "enriched_post": post
        }
    except KeyError as e:
        logger.error(f"Missing key in enrich_with_registration_link: {e}")
        return {
            "status": "error",
            "error": f"Missing required field: {str(e)}",
            "enriched_post": post
        }
    except Exception as e:
        logger.exception(f"Unexpected error in enrich_with_registration_link: {e}")
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "enriched_post": post
        }

# ============================================================================
# SERVER RUN - Lifespan Event Handler (FastAPI 0.93+)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("=" * 60)
    logger.info("News to Social Media MCP Server starting up")
    logger.info(f"HTML Parser: {HTML_PARSER_CONFIG}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("News to Social Media MCP Server shutting down")
    logger.info("=" * 60)

# FastAPI app HTTP JSON-RPC wrapper MCP szerverhez
app = FastAPI(
    title="News to Social Media MCP Server",
    description="MCP szerver HTTP JSON-RPC wrapper-rel - n8n kompatibilis",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    """Health check - no auth required"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "service": "News to Social Media MCP Server",
        "version": "1.0.0"
    }

class ToolRequest(BaseModel):
    """Tool hívás request"""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)

class ToolResponse(BaseModel):
    """Tool response"""
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint - API information"""
    logger.info("Root endpoint accessed")
    return {
        "status": "running",
        "service": "News to Social Media MCP Server",
        "version": "1.0.0",
        "mcp_endpoint": "http://fastmcp-server:8000/mcp/tool",
        "auth_required": False,
        "api_docs": "http://fastmcp-server:8000/docs"
    }

@app.get("/mcp/tools")
async def list_tools_api():
    """Elérhető MCP toolok listája"""
    logger.info("Tool list requested")
    tools = await list_mcp_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema
            }
            for t in tools
        ],
        "count": len(tools)
    }

@app.post("/mcp/tool", response_model=ToolResponse)
async def call_tool_api(raw_request: dict = Body(...)):
    """
    MCP Tool hívás HTTP JSON-RPC protokollon
    
    Használat az n8n-ben:
    POST http://fastmcp-server:8000/mcp/tool
    
    Body:
    {
        "name": "parse_html_and_extract_news",
        "arguments": {
            "html_content": "...",
            "source_url": "https://example.com"
        }
    }
    """
    try:
        # Validate request structure
        name = raw_request.get("name")
        arguments = raw_request.get("arguments", {})

        if not name or not isinstance(name, str):
            logger.warning("Invalid request: missing or non-string 'name'")
            return ToolResponse(status="error", error="Invalid request: 'name' (string) is required")

        if not isinstance(arguments, dict):
            logger.warning(f"Invalid request: arguments must be dict, got {type(arguments)}")
            return ToolResponse(status="error", error="Invalid request: 'arguments' must be a dict")

        logger.info(f"Processing tool call: {name}")
        result = await call_mcp_tool(name, arguments)
        
        # Szöveges konverzió ha szükséges
        if result.get("isError", False):
            error_text = "Unknown error"
            content = result.get("content", [])
            if content and len(content) > 0:
                error_text = content[0].get("text", "Unknown error") if isinstance(content[0], dict) else str(content[0])
            logger.error(f"Tool error: {error_text}")
            return ToolResponse(status="error", error=error_text)
        
        # Sikeres hívás
        result_text = ""
        content = result.get("content", [])
        if content and len(content) > 0:
            result_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
        
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse result as JSON: {e}")
            result_json = {"raw_result": result_text}
        
        return ToolResponse(status="success", result=result_json)
    
    except Exception as e:
        logger.exception(f"Unexpected error in call_tool_api: {e}")
        return ToolResponse(status="error", error=f"Server error: {str(e)}")

def run_server():
    """HTTP szerver indítása (szinkron)"""
    # Uvicorn saját event loop-ot kezel; futtatjuk szinkron módon
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    # Indítsuk el a HTTP szervert
    run_server()