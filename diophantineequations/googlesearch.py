from googleapiclient.discovery import build

API_KEY = "<apikey>"
API_CX = "<cx>"

def exact_search(phrase: str):
    service = build("customsearch", "v1", developerKey=API_KEY, cache_discovery=False)
    req = service.cse().list(q=phrase, cx=API_CX, exactTerms=phrase, num=10)
    return req.execute()["items"]

if __name__ == '__main__':
    results = exact_search("this is a test")
    for r in results:
        print(r["title"], r["link"])
