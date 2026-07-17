#!/usr/bin/env bash
# Gera data/libras_dataset_drive.zip — pacote de dados para treinar no Colab.
#
# Conteúdo (só .npy, symlinks resolvidos):
#   processed/alphabet/   Brazilian Alphabet processado (15 classes)
#   raw/                  capturas próprias na webcam alvo (8 classes)
#   luiz_split/static/    DATA_LUIZ estático (20 classes)
#   luiz_split/dynamic/   DATA_LUIZ letras dinâmicas h/j/k/x/y/z (6 classes)
#
# Uso:
#   bash scripts/make_drive_zip.sh
#   → subir o zip para MyDrive/libras/ no Google Drive
#   → o notebook notebooks/relatorio_final.ipynb descompacta em data/ no Colab
set -euo pipefail
cd "$(dirname "$0")/../data"

rm -f libras_dataset_drive.zip
# zip segue symlinks por padrão (sem -y) — resolve os links do luiz_split
zip -qr libras_dataset_drive.zip processed/alphabet raw luiz_split -i '*.npy'

echo "Gerado: data/libras_dataset_drive.zip"
unzip -l libras_dataset_drive.zip | tail -1
du -h libras_dataset_drive.zip
