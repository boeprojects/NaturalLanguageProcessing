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


### Teil 2: 
- Datenanalyse& Cleaning - Reading Data - Preprocessing - SpaCy (lemmatizing Grundform, etc.)
- Feature Engineering - wordcount - Sentimentanalyse

Imports
```python 
%matplotlib inline
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
import string
import json
import spacy
import numpy as np

from spacy_sentiws import spaCySentiWS
from tqdm.auto import tqdm
from collections import Counter # Counter ist in diesem Fall auch eine Klasse, daher Großschreibung

tqdm.pandas()

nlp = spacy.load('de_core_news_lg')
pd.set_option('display.max_colwidth', -1)
```

Lesen der Daten (Methoden connect & cursor)
```python 
conn = sqlite3.connect('headlines.db')# standardisierter Aufruf der DAB Connection!
cur = conn.cursor()
```

Lookup 5 Datensätze
```python 
sql = '''
    SELECT * FROM headlines
'''
cur.execute(sql)
headlines = cur.fetchall()
headlines[:5]
```
![nlp_01](https://user-images.githubusercontent.com/67191365/117983144-f54cf780-b336-11eb-9f85-09cf86e0b5d9.PNG)

Einfügen in einen Pandas Dataframe
```python 
df = pd.DataFrame.from_records(headlines, columns=['ts', 'title', 'src']) # Pandas Data frame
df
```
![nlp_02](https://user-images.githubusercontent.com/67191365/117983526-570d6180-b337-11eb-89c4-95aef4c93fcc.PNG)

Preprocessing
```python 
df.info() # non-null object entspricht string (kann er nicht genau erkennen)
```
![nlp_03](https://user-images.githubusercontent.com/67191365/117984305-03e7de80-b338-11eb-915f-931235a4e0b6.PNG)

Konvertieren der Spalte ts in datetime
```python 
df['ts'] = pd.to_datetime(df['ts'], utc=True) #utc - Zeitschema universal time coordinated
df
```
![nlp_04](https://user-images.githubusercontent.com/67191365/117984658-4e695b00-b338-11eb-8c47-efb691a86d38.PNG)


Typische Text Preprocessing Möglichkeiten (für Text)

 -  Alles in lower case umwandeln
 -  Punctuation entfernen - Sonderzechen, Punkte, etc. (Spezialfälle Smilies, Emojys (wg. Twtter Feeds),...)
 -  Named Entities - Eigennamen - können sehr vielfältig sein, Rausfiltern (Bsp. url, hashtag,... )
 -  Stop Words entfernen (Artikel, Füllwörter, etc.)
 -  Stemming - Wortstamm - Zeitformen, Fälle, ... Alle Wörter in ihre Grundformen bringen
 -  Tokenize(POS Tagging)part of speech tagging - den Text, Satz in Tokens teilen 
    jedes Wort steht für sich und seine Aufgabe im Text (Ort, Protagonist, usw.)


Neue Spalte 'title_processed'
```python 
df['title_processed'] = df['title'].str.lower()# Hinzufügen neue Spalte; Ergebnis lower ist in neuer Spalte, daher nur hier
```

Punktuierung entfernen
```python 
def remove_punctuation(s):
    for p in string.punctuation+"–“„’«»":
        s = s.replace(p, '')
    return s


df['title_processed'] = df['title_processed'].apply(remove_punctuation)
df
```
![nlp_05](https://user-images.githubusercontent.com/67191365/117985882-655c7d00-b339-11eb-8dfd-bb0a8eb04ef7.PNG)


Stopwords entfernen mit JSON file mit With-Open Anweisung
```python 
with open("stopwords-de.json", "rb") as f:
    stopwords = json.load(f)
    
    
def remove_stopwords(words):
    return [word for word in words if word not in stopwords]
    
    
df["title_processed"] = df["title_processed"].apply(remove_stopwords)
df
```
![nlp_06](https://user-images.githubusercontent.com/67191365/117993295-e28af080-b33f-11eb-81c9-39c084457629.PNG)

Token für Token iterieren - Lemma Grundform - neue Liste bilden
```python 
nlp = spacy.load('de_core_news_lg')

def build_lemma(words):
    s =  " ".join(words) # füge zusammen mit leerzeichen words (satz)
    doc = nlp(s)
    return [token.lemma_ for token in doc] # token für token iterieren, lemma rausnehmeun + neue Liste bilden (return)


df['title_processed'] = df['title_processed'].progress_apply(build_lemma)
df[['title', 'title_processed']]
```
![nlp_07](https://user-images.githubusercontent.com/67191365/117995006-4235cb80-b341-11eb-86ff-b2c4b3233590.PNG)


Spalten neu erzeugen nach Überarbeitung oben
```python 
df["title_processed"] = df["title_processed"].str.join(" ")

df['weekday'] = df['ts'].dt.weekday # Erzeugen neue Spalte weekday (0-7 für die Tage)
df['len'] = df['title'].apply(len) # Länge der Zeichen als neue Spalte
df
```
![nlp_07](https://user-images.githubusercontent.com/67191365/117995574-ba03f600-b341-11eb-8391-89ed3ac62f69.PNG)

### Feature Engineering & Data Exploration

Was ist die durchschnittliche Länge einer Schlagzeile?
```python 
df['len'].mean() # Anzahl Wörter durchschnittlich
```
61.34068260876872

```python 
df['len'].min(), df['len'].max()
```
(0, 269)

Wie ist die Verteilung der Schalgzeilenlänge (len)?
```python 
df['len'].hist()
```
![nlp_09](https://user-images.githubusercontent.com/67191365/117996955-e0766100-b342-11eb-9b7d-a722e35c4a39.PNG)

###Analysis
wordcount
```python 
words = ' '.join(df['title_processed'].to_list()).split() # Python Liste mit allen Schlagzeilen, mit join zum Textblock konvertiert + Split
c = Counter(words)
df_words = pd.DataFrame.from_dict(dict(c), orient='index').reset_index() # Index ist der Key (in diesem fall word), Spalte values
df_words.columns = ['word', 'occurence']
df_words![nlp_10]
```
![nlp_10](https://user-images.githubusercontent.com/67191365/117997518-52e74100-b343-11eb-821e-4c6bee238274.PNG)

Wie viel Counts gab es überhaupt? Wieviel Wörter hatte der ganze Datensatz?
```python 
df_words['occurence'].sum()
```
599186

Wie viele einzigartige Wörter gibt es?
```python 
df_words.shape[0]
```
71780

Was sind die top 30 Wörter?
```python 
top_10 = df_words.sort_values('occurence', ascending=False).head(30) # False heißt nicht aufsteigend ! geht nicht descending; top 10
top_10.plot.barh(y='occurence', x='word', figsize=(10, 30))
```
![nlp_11](https://user-images.githubusercontent.com/67191365/117999215-e8370500-b344-11eb-8ea7-eef46454ed75.PNG)
![nlp_12](https://user-images.githubusercontent.com/67191365/117999360-0f8dd200-b345-11eb-8d36-afad2fb8c956.PNG)
![nlp_13](https://user-images.githubusercontent.com/67191365/117999493-33511800-b345-11eb-828c-1563beec17b6.PNG)


Blick auf Tabellenanfang und Wortpaare
```python 
for i, title in df['title'].head(100).iteritems():
    doc = nlp(title)
    for ent in doc.ents:
        if ent.label_ == 'LOC':
            print(ent)
```
Mainz
Paris
Niederlande
Großbritanniens
US-Präsident
Wahlkampfauftritten
Brüssel
Belarus
Frankfurt
US-Präsidenten
Belarus
USA
Niederlande
Schottland
den USA
US-Präsident
Frankreich
Corona-Wirtschaftskrise
Deutschland
Bremen
Dannenröder Forst
Honduras
Zentralamerika
USA
Dannenröder Forst
Berlin
Anti-Corona-Proteste
Konstanz
Breitscheidplatz
Nordrhein-Westfalen
Livegespräche
Zombie-Corona
US-Moderator
Argentinien
Mafalda-Schöpfer Quino
Buenos Aires
Drachen-Millionär
Europa
Corona-Ampel
Deutschland
Vertrauensprüfung
Huawei-Technik
Deutschland
den USA
Union


###Sentimentanalyse

- Es gibt 2 Textkörper (positiv, negativ)
- Für jeden Textkörper wird ein Word Count durchgeführt Example: Auto: 11, schön: 27, ...
- Bedingte Wahrscheinlickeit für jedes einzelne Wort in dem jeweiligen Textkörper (laplacian smoothing)

Problem: Produkt aus leerer Liste ist wahrscheinlich 1.000000
```python 
sentiws = spaCySentiWS(sentiws_path='SentiWS_v2.0')
nlp.add_pipe(sentiws)

def compute_sentiment(headline):
    doc = nlp(headline)
    sentiments = []
    for token in doc:
        if token._.sentiws is not None:
            sentiments.append(token._.sentiws)
    if len(sentiments) > 0:
         return np.prod(sentiments)
    else:
        return np.nan
            


# df.head(20)['title'].apply(compute_sentiment)
last_1000 = df.copy().tail(1000)
last_1000['sentiment'] = last_1000['title'].progress_apply(compute_sentiment)
last_1000[['title', 'sentiment']]
```
![nlp_14](https://user-images.githubusercontent.com/67191365/118001532-1ddced80-b347-11eb-9d69-d8181fead152.PNG)


NaN Ausfiltern
```python 
last_1000.dropna().sort_values('sentiment')[['title', "sentiment"]]   #.to_excel('sentiment.xlsx')  # .to_html()
```
![nlp_15](https://user-images.githubusercontent.com/67191365/118002066-8fb53700-b347-11eb-80b1-999aef029bd7.PNG)


###Top 10 Positive
```python 
last_1000.dropna().sort_values('sentiment', ascending=False).head(10)[['title', "sentiment"]]
```
![nlp_16](https://user-images.githubusercontent.com/67191365/118002366-d60a9600-b347-11eb-8a94-3dabce5b898b.PNG)


###Top 10 Negative
```python 
last_1000.dropna().sort_values('sentiment').head(10)[['title', "sentiment"]]
```
![nlp_17](https://user-images.githubusercontent.com/67191365/118002603-110cc980-b348-11eb-9b9f-05677480ec44.PNG)


Allgemein Sentiment (mittelwert, statistisch)
```python 
last_1000['sentiment'].mean()
```
-0.027560041302126775

weitere Möglichkeiten:

- alle Parteien als Liste, alle Schlagzeilen mit z.B. CDU und Durchschnittsentiment dazu, für weitere Parteien
- Gruppieren nach Uhrzeiten (h tag 1-24), alle Schlagzeilen zwischen 0 und 1 Uhr und Durchschnittsentiment
- Wichtig ist Ideen generieren die voraussichtlich gute Aussagen ermöglichen


Speichern des Dataframe
```python 
df.to_pickle('headlines.pickle')
df
```
![nlp_18](https://user-images.githubusercontent.com/67191365/118003373-d192ad00-b348-11eb-9284-0885a0bc67c6.PNG)
