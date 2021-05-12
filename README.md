# NaturalLanguageProcessing

## Logistische Regression mit Textdaten und positiv - negativ Aussagen (binär)

### Teil 1: 
- Scrapen von Newsfeed Headlines (feedparser) als Data Generation abgelegt in einer Datenbank
- Datenbank Connection (open, close) und Laden der Daten zum Preprocessing



Imports für Feedparser und anwendung sqlite3
```python 
import feedparser
import sqlite3
```

Feedparser Funktion (get_entries) headlines mit Spalten published, updated, title
```python 
def get_entries(url):
    d = feedparser.parse(url)
    headlines = []
    for entry in d['entries']:
        if 'published' in entry:
            published = entry['published']
        elif 'updated' in entry:
            published = entry['updated']
        title = entry['title']
        try:  # versuche dies (record =m usw.), wenn fehlschlägt, nimm Exception & speichere in e,
              # print entry, gib trotzdem Fehler aus (raise)!
            record = (published, title, url)
        except Exception as e:
            print(entry) # hier wird nur die Exception geprinted !
            raise e
        headlines.append(record)
    return headlines
```

Aufruf Datenbank-Connection und Erzeugen der Tabelle
```python 
def init_db(filename):
    conn = sqlite3.connect(filename)# standardisierter Aufruf der DAB Connection immer als Erstes !
    cur = conn.cursor()
    sql = """
        CREATE TABLE IF NOT EXISTS headlines(
        ts text,
        title text,
        src text,
        PRIMARY KEY(ts,title)
        );
    """
    cur.execute(sql)
    return (conn, cur)
```

Funktion zum Erstellen einer Liste als "headlines" (Methode readlines)
```python 
def get_urls(filename):
    with open(filename) as f:
        urls = f.readlines()
    return urls
```

Scrapen der Daten (entries) und Anhängen an "headlines"
```python 
def scrape(urls):
    headlines = []
    for url in urls:
        entries = get_entries(url)
        headlines += entries
    return headlines
```

Schreiben der Daten in die Datenbank und Commit
```python 
def insert_headlines(headlines):
    sql = '''
        INSERT OR IGNORE INTO headlines (ts,title,src)
        VALUES (?,?,?)
    '''
    cur.executemany(sql, headlines)
    conn.commit() # commiten damit in die DB richtig geschrieben wird
```


Initialisieren der headlines.db und Überabe der Teilfunktionen
Insert der headlines und Schließen der Datenbankconnection
```python 
conn, cur = init_db('headlines.db') # zweiter Teil, main als eigentlicher Ablauf ! (1.Teilfunktionen abgesetzt)
urls = get_urls('urls.txt')
headlines = scrape(urls)
insert_headlines(headlines)

conn.close() # schließt die Datenconnection
```
