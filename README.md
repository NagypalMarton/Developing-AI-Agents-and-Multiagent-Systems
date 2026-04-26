
# Developing-AI-Agents-and-Multiagent-Systems

A gyakorlatorientált kurzus bevezet az MI ágensek és multiágens rendszerek fejlesztésébe. Középpontjában az LLM-ágensek, azok eszközhasználata (pl. MCP) és az ipari keretrendszerek (pl. AgentKit) állnak. Tárgyalja a rendszerek működését, mérését, valamint biztonsági, etikai és jogi aspektusaikat. E kurzusnak a plusz pontos feladatait tartalmaza.

## FastMCP + n8n futtatása Dockerben

### 1) Indítás

```bash
docker compose up --build
```

Elérhető szolgáltatások:

- n8n UI: `http://localhost:5678`
- FastMCP endpoint: `http://localhost:8001/mcp` (vagy a `FASTMCP_HOST_PORT` értéke szerint)

### 2) n8n MCP Client beállítás

Ha az n8n és a FastMCP ugyanabban a `docker-compose` hálózatban fut:

- MCP szerver URL: `http://fastmcp:8000/mcp`

Ha nem konténerből, hanem host gépről hívod az MCP szervert:

- MCP szerver URL: `http://localhost:8001/mcp`

### 2.1) Elérhető MCP toolok

Az `src/fastmcp_server.py` szerver a következő, n8n-ből hívható MCP toolokat adja:

- `discover_news_event_urls`
  - Bemenet: HTML forrás (`html_content`) vagy letöltendő oldal (`page_url`), opcionális `base_url`
  - Működés: az oldalon lévő `<a>` linkeket kigyűjti, normalizálja, és kulcsszavak alapján hírekre vagy eseményekre bontja
  - Kimenet: `news_urls`, `event_urls`, `total_links_scanned`
- `extract_page_content`
  - Bemenet: feldolgozandó URL-ek (`urls`), opcionális `max_items`, `summary_sentence_count`
  - Működés: az oldalakat letölti, a fő szöveget kinyeri, a forrást BME/tmit/aut/ttk/other szerint besorolja, majd az LLM dönti el, hogy az adott oldal `news` vagy `event`
  - Kimenet: `page_items`, ahol minden elem tartalmazza a forrás URL-t, a forrás egységet, a típust és a kinyert tartalmat

A második tool LLM-konfigurációt igényel: állítsd be a `NEWS_EVENTS_LLM_MODEL` változót, és ha nem OpenAI-alapú végpontot használsz, a `NEWS_EVENTS_LLM_BASE_URL` értékét is.

### 2.2) Javasolt n8n feldolgozási lánc

1. HTML betöltése (pl. HTTP Request)
2. `discover_news_event_urls` a gyűjtőoldalon
3. `extract_page_content` a kiválasztott URL-ekre
4. opcionális további LLM vagy szűrési lépések külső node-ban

### 3) Leállítás

```bash
docker compose down
```
