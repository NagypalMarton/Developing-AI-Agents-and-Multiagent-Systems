# Software Requirements Specification (SRS)

## AI-vezérelt tartalomgeneráló workflow egyetemi hírekhez (n8n alapokon)

---

## 1. Bevezetés

### 1.1 Cél

A rendszer célja egy olyan automatizált workflow megvalósítása, amely egyetemi/tanszéki híreket dolgoz fel, és ezekből platform-specifikus közösségi média bejegyzéseket generál és publikál.

### 1.2 Hatókör (Scope)

A rendszer:

* magyar nyelvű egyetemi híreket dolgoz fel
* RSS feed és HTML forrásokból gyűjt adatot
* AI segítségével generál posztokat
* eseményeket detektál (binárisan: event / nem event)
* human-in-the-loop jóváhagyás után publikál

A rendszer **nem**:

* általános híraggregátor
* nem többnyelvű
* nem teljesen autonóm (jóváhagyás kötelező)

---

## 2. Áttekintés

### 2.1 Rendszer architektúra

```plaintext
Input (RSS / HTML)
    ↓
n8n (orchestration)
    ↓
Parsing + Cleaning
    ↓
Pydantic AI (structured extraction)
    ↓
Mistral (Ollama Cloud) – szöveggenerálás
    ↓
Validáció + duplikáció ellenőrzés
    ↓
Human approval (email)
    ↓
Publikálás (API)
```

---

## 3. Adatforrások

### 3.1 Input típusok

* RSS feed (XML)
* HTML oldalak (scraping)

### 3.2 Támogatott HTML template-ek

#### 3.2.1 Drupal node-hir

* title: `.node-title a`
* link: `href`
* summary: `.field-name-body p`

#### 3.2.2 BME news card

* title: `.bme_news_card-title`
* summary: `.bme_news_card-body p`
* date: `.field--name-created`
* tags: `.field--name-field-tags li`
* link: `<a href>`

### 3.3 Parsing szabály

* CSS selector alapú extraction
* ha parsing sikertelen → hiba + skip

---

## 4. Adatmodell

```python
class EventInfo(BaseModel):
    is_event: bool
    confidence: float  # [0,1]

class NewsItem(BaseModel):
    source: str
    source_url: str
    title: str
    summary: str
    full_text: str | None
    published_at: datetime | None
    tags: list[str] = []
    event: EventInfo
```

---

## 5. Funkcionális követelmények

### 5.1 Adatgyűjtés

* RSS feed-ek periodikus lekérése
* HTML oldalak scraping-je

### 5.2 Tartalomfeldolgozás

* HTML tisztítás
* szöveg normalizálás

### 5.3 Strukturált kinyerés

* Pydantic AI használata
* JSON validáció kötelező

### 5.4 Event detection

* bináris döntés: event / nem event
* ha confidence < 0.6 → event = false

### 5.5 Szöveggenerálás

* Mistral modell (Ollama Cloud)
* bemenet: strukturált NewsItem
* kimenet: közösségi poszt

### 5.6 Duplikáció kezelés

* hash = SHA256(title + link)
* ha létezik → skip

### 5.7 Human-in-the-loop

* email küldés:

  * generált szöveg
  * approve / reject link
* approve → publish
* reject → discard

### 5.8 Publikálás

* REST API-n keresztül
* platform adaptereken át

---

## 6. Output specifikáció

Minden generált posztra:

* max 280 karakter
* nincs emoji
* hangnem: közvetlen, hivatalos
* 3–5 hashtag
* kötelező link

### 6.1 Event esetén

* CTA kötelező

### 6.2 Nem event esetén

* nincs CTA

---

## 7. Nem-funkcionális követelmények

### 7.1 Teljesítmény

* max 30 sec / hír feldolgozás

### 7.2 Skálázhatóság

* nincs napi limit

### 7.3 Megbízhatóság (Reliability)

#### Retry policy

* max 3 retry
* backoff: 1s → 5s → 15s

#### JSON validáció

* invalid → retry
* ha továbbra is invalid → fail

#### Timeout

* max 30 sec / hír

#### Hibakezelés

* minden hiba → email
* workflow folytatódik

---

## 8. AI konfiguráció

* modell: Mistral (Ollama Cloud)
* input max: 2500 token
* output max: 300 token
* temperature: 0.4

---

## 9. Edge case-ek

* hiányzó dátum → nem kritikus
* hiányzó event info → LLM dönt
* parsing hiba → skip
* invalid JSON → retry → fail
* duplikált hír → skip

---

## 10. Acceptance criteria

* poszt ≤ 280 karakter
* tartalmaz linket
* 3–5 hashtag
* nincs emoji
* event → CTA van
* nem event → nincs CTA
* duplikáció nem történik
* feldolgozás ≤ 30 sec

---

## 11. Technológiai stack

* n8n (self-hosted)
* Pydantic AI (inline)
* Mistral (Ollama Cloud)
* Email service (SMTP)
* REST API integrációk

---

## 12. Biztonság

* API kulcsok környezeti változókban
* HTTPS kommunikáció
* input sanitization HTML parsing során

---
