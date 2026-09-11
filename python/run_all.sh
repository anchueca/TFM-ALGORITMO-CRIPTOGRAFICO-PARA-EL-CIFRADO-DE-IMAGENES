#!/usr/bin/env bash
# Ejecuta todos los analisis graficos de python sin modificar sus scripts.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
LOG_DIR="$OUTPUT_DIR/logs"
STATS_IMAGE=""
ONLY=""

usage() {
    cat <<EOF
Uso: $(basename "$0") [opciones]

Ejecuta los run.sh de los analisis Python en modo no interactivo.

Opciones:
  --only LISTA       Ejecuta solo nombres separados por comas.
                     Ejemplo: cml_spectrum,lyapunov
  --stats IMAGE      Incluye stats.py usando IMAGE como entrada.
  --output-dir DIR   Directorio para resultados y logs.
  -h, --help         Muestra esta ayuda.

Variables:
  OUTPUT_DIR=DIR     Alternativa a --output-dir.

Los argumentos no se reenvian a los scripts: cada uno usa sus valores por
 defecto para conservar el comportamiento de sus run.sh originales.
EOF
}

while (($#)); do
    case "$1" in
        --only)
            [[ $# -ge 2 ]] || { echo "Falta valor para --only" >&2; exit 2; }
            ONLY="$2"
            shift 2
            ;;
        --stats)
            [[ $# -ge 2 ]] || { echo "Falta imagen para --stats" >&2; exit 2; }
            STATS_IMAGE="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "Falta directorio para --output-dir" >&2; exit 2; }
            OUTPUT_DIR="$2"
            LOG_DIR="$OUTPUT_DIR/logs"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Opcion desconocida: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -x "$REPO_ROOT/.env/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.env/bin/python"
elif [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] No se encontro Python: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

ANALYSES=(
    bifurcacion
    chaos_generator
    cml_analysis
    cml_evolution
    cml_spectrum
    coupled_lyapunov
    coupled_lyapunov_diagram
    coupled_map
    differential_analysis
    key_sensitivity
    local_entropy
    lyapunov
    nist_tests
)

selected() {
    [[ -z "$ONLY" ]] && return 0
    local name
    IFS=',' read -ra requested <<< "$ONLY"
    for name in "${requested[@]}"; do
        [[ "$name" == "$1" ]] && return 0
    done
    return 1
}

run_analysis() {
    local name="$1"
    local log_file="$LOG_DIR/${name}.log"
    local started finished elapsed

    started=$(date +%s)
    echo "[RUN] $name"
    if [[ -x "$SCRIPT_DIR/$name/run.sh" ]]; then
        (
            cd "$SCRIPT_DIR/$name" || exit 1
            MPLBACKEND="${MPLBACKEND:-Agg}" "$SCRIPT_DIR/$name/run.sh"
        ) >"$log_file" 2>&1
    else
        (
            cd "$SCRIPT_DIR/$name" || exit 1
            MPLBACKEND="${MPLBACKEND:-Agg}" "$PYTHON_BIN" "$name.py"
        ) >"$log_file" 2>&1
    fi
    local result=$?
    finished=$(date +%s)
    elapsed=$((finished - started))

    if ((result == 0)); then
        echo "[OK ] $name (${elapsed}s)"
    else
        echo "[FAIL] $name (${elapsed}s); ver $log_file"
    fi
    return "$result"
}

failures=0
executed=0
for name in "${ANALYSES[@]}"; do
    selected "$name" || continue
    ((executed += 1))
    run_analysis "$name" || ((failures += 1))
done

if [[ -n "$STATS_IMAGE" ]]; then
    if [[ ! -f "$STATS_IMAGE" ]]; then
        echo "[FAIL] stats: no existe la imagen $STATS_IMAGE"
        ((failures += 1))
    else
        ((executed += 1))
        echo "[RUN] stats"
        if (cd "$REPO_ROOT" && MPLBACKEND="${MPLBACKEND:-Agg}" "$PYTHON_BIN" "$SCRIPT_DIR/stats.py" "$STATS_IMAGE") >"$LOG_DIR/stats.log" 2>&1; then
            echo "[OK ] stats"
        else
            echo "[FAIL] stats; ver $LOG_DIR/stats.log"
            ((failures += 1))
        fi
    fi
fi

if ((executed == 0)); then
    echo "[ERROR] No se seleccionaron analisis."
    exit 2
fi

echo "Procesados: $executed; fallos: $failures"
echo "Logs: $LOG_DIR"
exit "$failures"
