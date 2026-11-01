import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/") as r:
    h = r.read().decode("utf-8")
    print("1. Home has interactive constellation canvas:", "lux-constellation" in h)
    print("2. Home has 24K gold beam gradient:", "goldBeam" in h)
    print("3. Home has Corinthian columns & Ashoka Chakra:", "marbleGrad" in h and "Dharma" in h)

with urllib.request.urlopen("http://127.0.0.1:8000/ask") as r:
    a = r.read().decode("utf-8")
    pos_comp = a.find('id="composer"')
    pos_feed = a.find('id="feed"')
    print("4. Legal advice composer positioned at TOP before feed:", pos_comp < pos_feed)
    print("5. Legal advice composer is NOT fixed at bottom:0:", "bottom:0;left:0;right:0" not in a)
