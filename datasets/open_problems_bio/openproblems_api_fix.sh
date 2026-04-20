#!/bin/bash
# Apply all openproblems fixes
# Usage: cd <openproblems_repo_root> && bash apply_patches.sh

set -e

echo "Resetting modified files..."
git checkout -- .

echo ""
echo "1/4: Patching openproblems/data/__init__.py (cache dir fix)..."
cd openproblems/data
python -c "
content = open('__init__.py').read()
old = '''    tempdir = os.path.join(tempfile.gettempdir(), \"openproblems_cache\")
    try:
        os.mkdir(tempdir)
        log.debug(\"Created data cache directory\")
    except OSError:
        log.debug(\"Data cache directory exists\")'''
new = '''    cache_dir = os.environ.get(\"OPENPROBLEMS_CACHE_DIR\")
    if cache_dir:
        tempdir = cache_dir
    else:
        tempdir = os.path.join(tempfile.gettempdir(), \"openproblems_cache\")
    os.makedirs(tempdir, exist_ok=True)'''
assert old in content, 'Patch 1/4 failed: pattern not found'
open('__init__.py', 'w').write(content.replace(old, new))
print('  OK')
"
cd ../..

echo "2/4: Patching openproblems/data/multimodal/utils.py (tocsr fix)..."
cd openproblems/data/multimodal
python -c "
content = open('utils.py').read()
old = '''    adata = anndata.AnnData(
        scprep.utils.to_array_or_spmatrix(X).tocsr(),
        obs=pd.DataFrame(index=joint_index),
        var=pd.DataFrame(index=X_columns),
    )
    adata.obsm[\"mode2\"] = scprep.utils.to_array_or_spmatrix(Y).tocsr()'''
new = '''    X_mat = scprep.utils.to_array_or_spmatrix(X)
    if hasattr(X_mat, 'tocsr'):
        X_mat = X_mat.tocsr()
    Y_mat = scprep.utils.to_array_or_spmatrix(Y)
    if hasattr(Y_mat, 'tocsr'):
        Y_mat = Y_mat.tocsr()
    adata = anndata.AnnData(
        X_mat,
        obs=pd.DataFrame(index=joint_index),
        var=pd.DataFrame(index=X_columns),
    )
    adata.obsm[\"mode2\"] = Y_mat'''
assert old in content, 'Patch 2/4 failed: pattern not found'
open('utils.py', 'w').write(content.replace(old, new))
print('  OK')
"
cd ../../..

echo "3/4: Patching openproblems/data/tabula_muris_senis.py (CZI API fix)..."
cd openproblems/data
python -c "
content = open('tabula_muris_senis.py').read()
old1 = '''    dataset_id = dataset[\"id\"]
    assets_path = (
        f\"/curation/v1/collections/{COLLECTION_ID}/datasets/{dataset_id}/assets\"
    )
    url = f\"{API_BASE}{assets_path}\"
    assets = _get_json(url)
    assets = [asset for asset in assets if asset[\"filetype\"] == \"H5AD\"]'''
new1 = '''    dataset_id = dataset[\"dataset_id\"]
    assets = [asset for asset in dataset[\"assets\"] if asset[\"filetype\"] == \"H5AD\"]'''
assert old1 in content, 'Patch 3a/4 failed: pattern not found'
content = content.replace(old1, new1)

old2 = '    filename = f\"{COLLECTION_ID}_{dataset_id}_{asset[' + \"'\" + 'filename' + \"'\" + ']}\"'
new2 = '    filename = f\"{COLLECTION_ID}_{dataset_id}.h5ad\"'
assert old2 in content, 'Patch 3b/4 failed: pattern not found'
content = content.replace(old2, new2)

old3 = '        scprep.io.download.download_url(asset[\"presigned_url\"], filepath)'
new3 = '        scprep.io.download.download_url(asset[\"url\"], filepath)'
assert old3 in content, 'Patch 3c/4 failed: pattern not found'
content = content.replace(old3, new3)

old4 = '''    utils.filter_genes_cells(adata)
    # If \`raw\` exists, raw counts are there'''
new4 = '''    utils.filter_genes_cells(adata)'''
assert old4 in content, 'Patch 3d/4 failed: pattern not found'
content = content.replace(old4, new4)

open('tabula_muris_senis.py', 'w').write(content)
print('  OK')
"
cd ../..

echo "4/4: Patching openproblems/tasks/denoising/datasets/utils.py (np.int fix)..."
cd openproblems/tasks/denoising/datasets
python -c "
content = open('utils.py').read()
old = 'X.astype(np.int)'
new = 'X.astype(int)'
assert content.count(old) == 2, f'Patch 4/4 failed: expected 2 occurrences, found {content.count(old)}'
open('utils.py', 'w').write(content.replace(old, new))
print('  OK')
"
cd ../../../..

echo ""
echo "All 4 patches applied successfully!"