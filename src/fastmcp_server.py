import asyncio
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests
from mcp.server import Server
from mcp.types import Tool, TextContent, ToolResult
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

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
# HTML PARSER UTILS
# ============================================================================

def parse_tmit_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """TMIT (node-hir) típusú hír parsése"""
    soup = BeautifulSoup(html_content, 'html.parser')
    article = soup.find('article', class_='node-hir')
    
    if not article:
        return None
    
    title_elem = article.find('h2', class_='node-title')
    title = title_elem.text.strip() if title_elem else "N/A"
    
    img_elem = article.find('img')
    image_url = img_elem.get('src') if img_elem else None
    
    content_elem = article.find('div', {'property': 'content:encoded'})
    content = content_elem.text.strip() if content_elem else ""
    
    return NewsItem(
        title=title,
        content=content,
        image_url=image_url,
        source_url=source_url
    )

def parse_vik_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """VIK (news-title-important) típusú hír parsése"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Ha direkt h2-t keresünk, vagy parent wrapper-ben kell
    title_elem = soup.find('h2', class_='news-title-important')
    if not title_elem:
        return None
        
    title = title_elem.text.strip() if title_elem else "N/A"
    
    date_elem = soup.find('span', class_='news-date')
    publish_date = date_elem.text.strip() if date_elem else None
    
    excerpt_elem = soup.find('div', class_='news-excerpt')
    content = excerpt_elem.text.strip() if excerpt_elem else ""
    
    img_elem = soup.find('img', class_='news-image')
    image_url = img_elem.get('src') if img_elem else None
    
    return NewsItem(
        title=title,
        content=content,
        image_url=image_url,
        source_url=source_url,
        publish_date=publish_date
    )

def parse_bme_news(html_content: str, source_url: str) -> Optional[NewsItem]:
    """BME (bme_news_card) típusú hír parsése"""
    soup = BeautifulSoup(html_content, 'html.parser')
    news_card = soup.find('div', class_='bme_news_card')
    
    if not news_card:
        return None
    
    title_elem = news_card.find('h4', class_='bme_news_card-title')
    title = title_elem.text.strip() if title_elem else "N/A"
    
    date_elem = news_card.find('span', class_='field--name-created')
    publish_date = date_elem.text.strip() if date_elem else None
    
    body_elem = news_card.find('div', class_='bme_news_card-body')
    content = body_elem.text.strip() if body_elem else ""
    
    img_elem = news_card.find('img')
    image_url = img_elem.get('src') if img_elem else None
    
    return NewsItem(
        title=title,
        content=content,
        image_url=image_url,
        source_url=source_url,
        publish_date=publish_date
    )

def parse_bme_event(html_content: str, source_url: str) -> Optional[EventDetected]:
    """BME event card parsése"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    title_elem = soup.find('h4', class_='bme_event_card-title')
    title = title_elem.text.strip() if title_elem else "N/A"
    
    date_elem = soup.find('div', class_='bme_event_card-date')
    date_str = ""
    if date_elem:
        span = date_elem.find('span', class_='nowrap')
        date_str = span.text.strip() if span else ""
    
    location_elem = soup.find('p', class_='bme_event_card-location')
    location = location_elem.text.strip() if location_elem else None
    
    body_elem = soup.find('div', class_='bme_event_card-body')
    description = body_elem.text.strip() if body_elem else ""
    
    event_type = detect_event_type(title)
    
    return EventDetected(
        title=title,
        date=parse_date_string(date_str),
        location=location,
        event_type=event_type,
        description=description,
        source_url=source_url
    )

def parse_simple_event(html_content: str, source_url: str) -> Optional[EventDetected]:
    """Egyszerű event formátum parsése (VIK)"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    date_elem = soup.find('span', class_='event-date')
    date_str = date_elem.text.strip() if date_elem else ""
    
    title_elem = soup.find('a', class_='event-title')
    title = title_elem.text.strip() if title_elem else "N/A"
    
    return EventDetected(
        title=title,
        date=parse_date_string(date_str),
        event_type=EventType.OTHER,
        description="",
        source_url=source_url
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_date_string(date_str: str) -> str:
    """Magyar dátum stringet ISO formátumra alakít"""
    if not date_str:
        return datetime.now().isoformat()
    
    try:
        original_date_str = date_str
        # "2026. 05. 15. - 20:00" vagy "2026. május 15." formátum kezelése
        
        # Dátum és idő extraktálása regex-szel az eredeti stringből
        date_match = re.search(r'(\d{4})\.\s+(\d{1,2})\.\s+(\d{1,2})', original_date_str)
        if not date_match:
            return datetime.now().isoformat()
        
        year, month, day = date_match.groups()
        
        # Idő keresése
        time_match = re.search(r'(\d{1,2}):(\d{2})', original_date_str)
        if time_match:
            hour, minute = time_match.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}:{minute}:00"
        
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}T00:00:00"
    
    except Exception as e:
        print(f"Date parsing error: {e}")
        return datetime.now().isoformat()

def detect_event_type(title: str) -> EventType:
    """Eseménytípus detektálása a címből"""
    title_lower = title.lower()
    
    if any(word in title_lower for word in ['doktori', 'védés', 'phd']):
        return EventType.DOCTORAL_DEFENSE
    elif any(word in title_lower for word in ['koncert', 'zene', 'szimfónia']):
        return EventType.CONCERT
    elif any(word in title_lower for word in ['konferencia', 'symposium']):
        return EventType.CONFERENCE
    elif any(word in title_lower for word in ['workshop', 'tanfolyam', 'képzés']):
        return EventType.WORKSHOP
    elif any(word in title_lower for word in ['előadás', 'lecture', 'diasor']):
        return EventType.LECTURE
    elif any(word in title_lower for word in ['határidő', 'deadline', 'pályázat']):
        return EventType.DEADLINE
    
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
async def call_mcp_tool(name: str, arguments: dict) -> ToolResult:
    """Handle MCP tool calls"""
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
            return ToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )
        
        return ToolResult(
            isError=False,
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        )
    
    except Exception as e:
        return ToolResult(
            isError=True,
            content=[TextContent(type="text", text=f"Error: {str(e)}")]
        )

async def parse_html_and_extract_news(html_content: str, source_url: str) -> Dict[str, Any]:
    """
    HTML feldolgozás és hírek + események extraktálása.
    
    Támogatott formátumok:
    - TMIT: <article class="node-hir">
    - VIK: <h2 class="news-title-important">
    - BME: <div class="bme_news_card">
    - Events: BME event cards, simple events
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        news_items = []
        events = []
        
        # TMIT hírek
        tmit_articles = soup.find_all('article', class_='node-hir')
        for article in tmit_articles:
            html_str = str(article)
            news = parse_tmit_news(html_str, source_url)
            if news:
                news_items.append(news.model_dump())
        
        # VIK hírek
        vik_news = soup.find_all('h2', class_='news-title-important')
        for news_elem in vik_news:
            parent_div = news_elem.find_parent('div', class_='news-item')
            if parent_div:
                html_str = str(parent_div)
                news = parse_vik_news(html_str, source_url)
                if news:
                    news_items.append(news.model_dump())
        
        # BME hírek
        bme_news = soup.find_all('div', class_='bme_news_card')
        for news_card in bme_news:
            html_str = str(news_card)
            news = parse_bme_news(html_str, source_url)
            if news:
                news_items.append(news.model_dump())
        
        # BME események
        bme_events = soup.find_all('div', class_='bme_event_card-date')
        for event_card in bme_events:
            parent = event_card.find_parent('div', class_='px-5')
            if parent:
                html_str = str(parent)
                event = parse_bme_event(html_str, source_url)
                if event:
                    events.append(event.model_dump())
        
        # Egyszerű események
        simple_events = soup.find_all('div', class_='event')
        for event_elem in simple_events:
            html_str = str(event_elem)
            event = parse_simple_event(html_str, source_url)
            if event:
                events.append(event.model_dump())
        
        return {
            "news_count": len(news_items),
            "event_count": len(events),
            "news_items": news_items,
            "events": events,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "news_items": [],
            "events": []
        }

async def detect_events_from_content(content: str, current_date: str = None) -> Dict[str, Any]:
    """
    LLM segítségével részletesebb eseményadatok detektálása szövegből.
    
    Kimenete: JSON lista EventDetected sémával.
    """
    if not current_date:
        current_date = datetime.now().isoformat()
    
    # Regex alapú detektálás is lehetséges, de az LLM-et n8n hívja meg
    # Ez a tool csak validálja és strukturálja az eredményt
    try:
        # Próbálunk dátumokat szedni a szövegből
        date_pattern = r'(\d{4})\D+(\d{1,2})\D+(\d{1,2})'
        location_pattern = r'(?:helyszín|location|hely)[\s:]*([^,.\n]+)'
        
        dates_found = re.findall(date_pattern, content)
        locations_found = re.findall(location_pattern, content, re.IGNORECASE)
        
        return {
            "status": "success",
            "dates_found": dates_found,
            "locations_found": locations_found,
            "message": "Content processed. Use with LLM for full event extraction."
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
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
    """
    if not platforms:
        platforms = ["facebook", "linkedin", "x", "instagram", "discord"]
    
    if not events:
        events = []
    
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
    
    if "facebook" in platforms:
        result["facebook"] = {
            "content": templates["facebook"][:63206],
            "hashtags": ["#BME", "#Hírek"],
            "cta_button": "Tudj meg többet" if events else None,
            "cta_url": events[0].get("registration_url") if events else None
        }
    
    if "linkedin" in platforms:
        result["linkedin"] = {
            "headline": news_title[:200],
            "body": templates["linkedin"],
            "hashtags": ["BME", "Hírek", "Oktatás"],
            "cta_text": "Regisztrálj",
            "cta_url": events[0].get("registration_url") if events else source_url
        }
    
    if "x" in platforms:
        result["x"] = {
            "content": templates["x"][:280],
            "hashtags": ["BME", "Hírek"]
        }
    
    if "instagram" in platforms:
        result["instagram"] = {
            "caption": templates["instagram"][:2200],
            "hashtags": ["BME", "Egyetem", "Hírek", "Oktatás"],
            "emoji_usage": "🎓📚🏫"
        }
    
    if "discord" in platforms:
        result["discord"] = {
            "embed_title": news_title[:256],
            "embed_description": news_content[:2048],
            "embed_fields": {
                "Forrás": source_url,
                "Típus": "Hír"
            },
            "embed_color": 3447003
        }
    
    return {
        "status": "success",
        "posts": result,
        "platforms_generated": list(result.keys()),
        "event_count": len(events)
    }

async def enrich_with_registration_link(
    post: Dict[str, Any],
    event: Dict[str, Any],
    platform: str
) -> Dict[str, Any]:
    """
    Regisztrációs link injektálása az eseményt tartalmazó posztokba.
    """
    try:
        registration_url = event.get("registration_url") or event.get("source_url")
        
        if not registration_url:
            return {
                "status": "warning",
                "message": "No registration URL found",
                "enriched_post": post
            }
        
        enriched = post.copy()
        event_title = event.get("title", "")
        event_date = event.get("date", "")
        
        if platform == "facebook":
            enriched["content"] += f"\n\n📅 {event_title} ({event_date})\n🔗 Regisztrálj: {registration_url}"
        
        elif platform == "linkedin":
            enriched["body"] += f"\n\n🎯 {event_title}\n🗓️ {event_date}\n\n👉 {registration_url}"
        
        elif platform == "x":
            enriched["content"] = enriched["content"][:250] + f"\n🔗 {registration_url}"
        
        elif platform == "instagram":
            enriched["caption"] += f"\n\n📌 {event_title}\n{registration_url}"
        
        elif platform == "discord":
            enriched["embed_fields"]["Regisztráció"] = registration_url
            enriched["embed_fields"]["Dátum"] = event_date
        
        return {
            "status": "success",
            "enriched_post": enriched,
            "platform": platform
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "enriched_post": post
        }

# ============================================================================
# SERVER RUN
# ============================================================================

# FastAPI app HTTP JSON-RPC wrapper MCP szerverhez
app = FastAPI(
    title="News to Social Media MCP Server",
    description="MCP szerver HTTP JSON-RPC wrapper-rel - n8n kompatibilis",
    version="1.0.0"
)

class ToolRequest(BaseModel):
    """Tool hívás request"""
    name: str
    arguments: Dict[str, Any]

class ToolResponse(BaseModel):
    """Tool response"""
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "service": "News to Social Media MCP Server",
        "version": "1.0.0",
        "mcp_endpoint": "http://fastmcp-server:8000/mcp/tool"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

@app.get("/mcp/tools")
async def list_tools_api():
    """Elérhető MCP toolok listája"""
    tools = await list_mcp_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema
            }
            for t in tools
        ]
    }

@app.post("/mcp/tool", response_model=ToolResponse)
async def call_tool_api(request: ToolRequest):
    """
    MCP Tool hívás HTTP JSON-RPC protokollon
    
    Használat az n8n-ben:
    POST http://fastmcp-server:8000/mcp/tool
    
    Body:
    {
        "name": "parse_html_and_extract_news",
        "arguments": {
            "html_content": "...",
            "source_url": "..."
        }
    }
    """
    try:
        result = await call_mcp_tool(request.name, request.arguments)
        
        # Szöveges konverzió ha szükséges
        if result.isError:
            error_text = "Unknown error"
            if result.content and len(result.content) > 0:
                error_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            return ToolResponse(status="error", error=error_text)
        
        # Sikeres hívás
        result_text = ""
        if result.content and len(result.content) > 0:
            result_text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
        
        try:
            result_json = json.loads(result_text)
        except:
            result_json = {"raw_result": result_text}
        
        return ToolResponse(status="success", result=result_json)
    
    except Exception as e:
        return ToolResponse(status="error", error=str(e))

async def run_server():
    """HTTP szerver indítása"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    # HTTP mode - n8n JSON-RPC wrapper-rel
    asyncio.run(run_server())