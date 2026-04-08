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

### 3) Leállítás

```bash
docker compose down
```
