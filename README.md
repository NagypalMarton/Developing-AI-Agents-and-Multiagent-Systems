
# Developing-AI-Agents-and-Multiagent-Systems

A gyakorlatorientált kurzus bevezet az MI ágensek és multiágens rendszerek fejlesztésébe. Középpontjában az LLM-ágensek, azok eszközhasználata (pl. MCP) és az ipari keretrendszerek (pl. AgentKit) állnak. Tárgyalja a rendszerek működését, mérését, valamint biztonsági, etikai és jogi aspektusaikat. E kurzusnak a plusz pontos feladatait tartalmaza.

## FastMCP + n8n futtatása Dockerben

### 1) Indítás

```bash
docker compose up --build
```

Elérhető szolgáltatások:

- n8n UI: `http://localhost:5678`
- FastMCP endpoint: `http://localhost:8000/mcp` (vagy a `FASTMCP_HOST_PORT` értéke szerint)

### 2) n8n MCP Client beállítás

Ha az n8n és a FastMCP ugyanabban a `docker-compose` hálózatban fut:

- MCP szerver URL: `http://fastmcp:8000/mcp`

Ha nem konténerből, hanem host gépről hívod az MCP szervert:

- MCP szerver URL: `http://localhost:8000/mcp`

### 2.1) Elérhető MCP toolok a hírfeldolgozó workflow-hoz

Az `src/fastmcp_server.py` szerver a következő, n8n-ből hívható MCP toolokat adja:

- `parse_news_html`
  - Bemenet: HTML forrás (`html`), opcionális `source_url`, `language`
  - Kimenet: strukturált cikk (`title`, `lead`, `body_text`, `tags`, `events`)
- `crawl_news_and_events_from_roots`
  - Bemenet: pontosan 2 db gyökér URL (`root_urls`), pl. egyetemi és tanszéki kezdőoldal
  - Működés: a gyökéroldalakról hírek/események linkjeit felderíti, majd ezeket bejárva gyűjti a BME kártyás elemeket és a tanszéki `article.node-hir` hírblokkokat
  - Kimenet: forrásonként `news`, `events`, `visited_urls`, `errors` + összesített statisztika
- `detect_events`
  - Bemenet: cikk szöveg + opcionális regisztrációs linkek
  - Kimenet: detektált eseménylista (`name`, `start_date`, `location`, `registration_url`)

### 2.2) Javasolt n8n feldolgozási lánc

1. HTML betöltése (pl. HTTP Request)
2. `crawl_news_and_events_from_roots` (2 kezdő URL-ről közvetett begyűjtés)
3. cikkenként `parse_news_html` (ha részletes feldolgozás kell)
4. események kinyerése `detect_events` segítségével
5. opcionális saját LLM lépés külső node-ban

Megjegyzés: a jelenlegi parser a BME eseménykártya mintákat kezeli,TMIT tanszéki oldalon külön esemény HTML-minta hiányában ott csak hírek kerülnek kigyűjtésre.

### 3) Leállítás

```bash
docker compose down
```
