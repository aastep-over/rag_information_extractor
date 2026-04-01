#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# percorso del venv (modifica se il tuo venv si chiama diversamente)
VENV_DIR=".venv"

# prova a trovare e attivare l'activate script (Linux/Mac o Windows Git Bash)
if [ -f "${VENV_DIR}/bin/activate" ]; then
  # venv creato su Linux/Mac style
  source "${VENV_DIR}/bin/activate"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
  # venv creato su Windows (Git Bash/MSYS)
  source "${VENV_DIR}/Scripts/activate"
else
  echo "Nessun venv trovato in ${VENV_DIR}. Impostare il path corretto al venv_dir nel run_extraction.sh. Exiting..." >&2
  exit 1
  # Se vuoi che lo script fallisca se non trova il venv, usa: exit 1
fi

# launch ollama
ollama serve > /dev/null 2>&1 &

# esegui lo script Python e passa eventuali argomenti
cd scripts
# python extract_info.py "$@"
python extract_info.py --chunks-type "fixed_size_chunks" # run this


# python extract_info.py --chunks-type "semantic_chunks"
# python extract_info.py --chunks-type "semantic_chunks"

# prova a disattivare (se è stato attivato)
if type deactivate >/dev/null 2>&1; then
  deactivate || true
fi
