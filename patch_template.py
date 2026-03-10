"""
Run this script from the project root to patch index.html after the running process completes.
Usage: python patch_template.py
"""
import os

file_path = os.path.join("predictor", "templates", "predictor", "index.html")

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Consolidate classification-analysis multiline if tag
bad1 = "{% if prediction and active_section == 'classification-analysis'\n                %}"
good1 = "{% if prediction and active_section == 'classification-analysis' %}"

# Fix 2: Consolidate regression-analysis multiline if tag (just in case)
bad2 = "{% if price and active_section == 'regression-analysis'\n                %}"
good2 = "{% if price and active_section == 'regression-analysis' %}"

# Fix 3: Consolidate clustering-analysis multiline if tag
bad3 = "{% if prediction and active_section == 'clustering-analysis'\n                %}"
good3 = "{% if prediction and active_section == 'clustering-analysis' %}"

changes = 0
for bad, good in [(bad1, good1), (bad2, good2), (bad3, good3)]:
    if bad in content:
        content = content.replace(bad, good)
        changes += 1
        print(f"Fixed: {bad[:50]}...")

# Also handle CRLF (\r\n) variants
for bad, good in [(bad1, good1), (bad2, good2), (bad3, good3)]:
    bad_crlf = bad.replace("\n", "\r\n")
    if bad_crlf in content:
        content = content.replace(bad_crlf, good)
        changes += 1
        print(f"Fixed CRLF: {bad[:50]}...")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nTotal fixes applied: {changes}")
print("Done.")
