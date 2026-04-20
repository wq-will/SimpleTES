#!/bin/bash

# Download the cache from Google Drive
wget --no-check-certificate \
  "https://drive.google.com/uc?export=download&id=1bA044QSbhsQWLjgs467ygoCpoxH3NevD" \
  -O cache_ahc.zip

# Unzip the cache
python3 -c "import zipfile; zipfile.ZipFile('cache_ahc.zip').extractall('.')"

# Remove the zip file
rm cache_ahc.zip

# Rename to cache/ (right here in datasets/ahc/)
mv cache cache_tmp 2>/dev/null  # in case 'cache' extracts differently
mv cache_tmp cache 2>/dev/null || true

echo ""
echo "Cache extracted to $(pwd)/cache/"
echo ""
ls cache/