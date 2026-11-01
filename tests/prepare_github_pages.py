import os
import re

# 1. Create .nojekyll file
with open(".nojekyll", "w", encoding="utf-8") as f:
    f.write("")
print("Created .nojekyll")

# 2. Update relative paths in all HTML files
files = ['index.html', 'ask.html', 'case-studies.html', 'news.html', 'reviews.html', 'login.html', '404.html']

replacements = [
    # Assets
    ('href="/lux.css"', 'href="./lux.css"'),
    ('href="/social-preview.svg"', 'href="./social-preview.svg"'),
    ('src="/social-preview.svg"', 'src="./social-preview.svg"'),
    ('content="/social-preview.svg"', 'content="./social-preview.svg"'),
    ('src="/lux.js"', 'src="./lux.js"'),
    ('src="/js/lenis.min.js"', 'src="./js/lenis.min.js"'),
    ('src="/js/gsap.min.js"', 'src="./js/gsap.min.js"'),
    ('src="/js/ScrollTrigger.min.js"', 'src="./js/ScrollTrigger.min.js"'),
    # Nav links
    ('href="/"', 'href="./index.html"'),
    ('href="/ask"', 'href="./ask.html"'),
    ('href="/case-studies"', 'href="./case-studies.html"'),
    ('href="/news"', 'href="./news.html"'),
    ('href="/reviews"', 'href="./reviews.html"'),
    ('href="/login"', 'href="./login.html"'),
    ('href="/robots.txt"', 'href="./robots.txt"'),
    ('href="/sitemap.xml"', 'href="./sitemap.xml"'),
]

for fname in files:
    if not os.path.exists(fname):
        continue
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated relative paths in {fname}")

# 3. Create .github/workflows/deploy.yml
os.makedirs(".github/workflows", exist_ok=True)
workflow_content = '''name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload Artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: '.'

      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
'''

with open(".github/workflows/deploy.yml", "w", encoding="utf-8") as f:
    f.write(workflow_content)
print("Created .github/workflows/deploy.yml")
