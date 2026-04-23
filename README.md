
# Developing-AI-Agents-and-Multiagent-Systems

A gyakorlatorientált kurzus bevezet az MI ágensek és multiágens rendszerek fejlesztésébe. Középpontjában az LLM-ágensek, azok eszközhasználata (pl. MCP) és az ipari keretrendszerek (pl. AgentKit) állnak. Tárgyalja a rendszerek működését, mérését, valamint biztonsági, etikai és jogi aspektusaikat. E kurzusnak a plusz pontos feladatait tartalmaza.

## FastMCP + n8n futtatása Dockerben

### 1) Indítás

```bash
docker compose up --build
```

Elérhető szolgáltatások:

- n8n UI: `http://localhost:5678`
- FastMCP endpoint: `http://localhost:8000/mcp`

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
- `generate_platform_posts`
  - Bemenet: strukturált cikk + platformok (`linkedin`, `facebook`, `x`)
  - Kimenet: platform-specifikus draft posztok karakterlimittel
- `build_llm_prompt_pack`
  - Bemenet: strukturált cikk
  - Kimenet: platform-specifikus promptok külső LLM node-okhoz
- `news_workflow_bundle`
  - Bemenet: strukturált cikk + opciók
  - Kimenet: egyben cikk + draft posztok + LLM prompt pack (`n8n`-barát bundle)
- `health`
  - Egyszerű állapotellenőrzés

### 2.2) Javasolt n8n feldolgozási lánc

1. HTML betöltése (pl. HTTP Request)
2. `crawl_news_and_events_from_roots` (2 kezdő URL-ről közvetett begyűjtés)
3. cikkenként `parse_news_html` (ha részletes feldolgozás kell)
4. `generate_platform_posts` (vagy `news_workflow_bundle`)
5. opcionálisan `build_llm_prompt_pack` + LLM node finomátírás
6. platformonként publikálás (LinkedIn/Facebook/X node-ok)

Megjegyzés: a jelenlegi parser a BME eseménykártya mintákat kezeli,TMIT tanszéki oldalon külön esemény HTML-minta hiányában ott csak hírek kerülnek kigyűjtésre.

### 3) Leállítás

```bash
docker compose down
```
