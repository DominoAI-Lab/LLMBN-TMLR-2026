#!/bin/bash
# LLMBN Installation Script
# - Enforces Python 3.10–3.12
# - Requires .env with at least one provider API key
# - Verifies R >= 4.3, R packages, and Python↔R (rpy2) integration

set -e  # Exit on any error

echo "🚀 LLMBN Installation Script"
echo "=============================="

# ---------- Helpers ----------
ver_ge () {
  [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

is_placeholder () {
  echo "$1" | grep -Eiq '(^$|your-?.*key|example|replace|xxx)'
}

# ---------- Python: 3.10–3.12 only ----------
if ! command -v python3 &> /dev/null; then
  echo "❌ Python 3 is required but not installed."
  echo "Please install Python 3.10–3.12 (3.13+ is NOT supported)."
  exit 1
fi

PY_VER_STR=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
PY_MAJ=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MIN=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJ" -ne 3 ] || [ "$PY_MIN" -lt 10 ] || [ "$PY_MIN" -gt 12 ]; then
  echo "❌ Unsupported Python version: $PY_VER_STR"
  echo "Required: Python 3.10–3.12 (3.13+ NOT supported)."
  exit 1
fi
echo "✅ Python found: $PY_VER_STR (supported)"

# ---------- pip ----------
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
  echo "❌ pip is required but not installed."
  exit 1
fi
PIP_CMD="pip3"; command -v pip3 &> /dev/null || PIP_CMD="pip"
echo "✅ pip found: $($PIP_CMD --version)"

# ---------- .env & API keys ----------
echo ""
echo "🔑 Checking .env and API keys..."
if [ ! -f ".env" ]; then
  echo "❌ .env file not found. Create .env with at least one key:"
  echo "   OPENAI_API_KEY=...  OR  GEMINI_API_KEY=...  OR  DEEPSEEK_API_KEY=..."
  exit 1
fi

OPENAI_VAL=$(grep -E '^[[:space:]]*OPENAI_API_KEY=' .env | head -n1 | cut -d= -f2- | tr -d '[:space:]')
GEMINI_VAL=$(grep -E '^[[:space:]]*GEMINI_API_KEY=' .env | head -n1 | cut -d= -f2- | tr -d '[:space:]')
DEEPSEEK_VAL=$(grep -E '^[[:space:]]*DEEPSEEK_API_KEY=' .env | head -n1 | cut -d= -f2- | tr -d '[:space:]')

AT_LEAST_ONE=false
for VAL in "$OPENAI_VAL" "$GEMINI_VAL" "$DEEPSEEK_VAL"; do
  if [ -n "$VAL" ] && ! is_placeholder "$VAL"; then
    AT_LEAST_ONE=true
    break
  fi
done

if [ "$AT_LEAST_ONE" != "true" ]; then
  echo "❌ No valid API key found in .env."
  echo "Add at least ONE real key (not a placeholder): OPENAI_API_KEY / GEMINI_API_KEY / DEEPSEEK_API_KEY."
  exit 1
fi
echo "✅ .env present with at least one valid API key"

# ---------- Python dependencies ----------
echo ""
echo "📦 Installing Python dependencies..."
$PIP_CMD install -r requirements.txt
echo "✅ Python dependencies installed"

# ---------- R (must be installed and >= 4.3) ----------
echo ""
echo "🔍 Checking R installation and version..."
if ! command -v R &> /dev/null; then
  echo "❌ R not found. R (≥ 4.3) is required."
  echo "Install R and re-run."
  exit 1
fi

R_VERSION_LINE=$(R --version | head -n1)
R_VERSION=$(echo "$R_VERSION_LINE" | sed -E 's/.* ([0-9]+\.[0-9]+(\.[0-9]+)?).*/\1/')
MIN_R="4.3.0"

if ver_ge "$R_VERSION" "$MIN_R"; then
  echo "✅ R found: $R_VERSION_LINE (meets ≥ $MIN_R)"
else
  echo "❌ R version too low: $R_VERSION_LINE"
  echo "Please install R ≥ $MIN_R and re-run."
  exit 1
fi

# ---------- R packages ----------
echo ""
echo "🧪 Testing required R packages (graph, RBGL, pcalg)..."
if R -e "suppressMessages({ library(graph); library(RBGL); library(pcalg) }); cat('OK\n')" >/dev/null 2>&1; then
  echo "✅ R packages are installed and load correctly"
else
  echo "❌ Required R packages missing or failed to load."
  echo "Run in R console:"
  echo "  if (!requireNamespace('BiocManager', quietly=TRUE)) install.packages('BiocManager')"
  echo "  BiocManager::install(c('graph','RBGL','pcalg'))"
  exit 1
fi

# ---------- Basic Python functionality test ----------
echo ""
echo "🧪 Testing basic Python functionality..."
python3 - <<'PY'
import sys
sys.path.insert(0, 'promptbn')  # adjust if your package path differs
try:
    from utils.path_utils import resolve_path
    print('✅ Path utilities working')
    from utils.data_utils import parse_var_csv_to_string
    print('✅ Data utilities working')
    print('✅ Basic functionality test passed')
except Exception as e:
    print(f'❌ Basic functionality test failed: {e}')
    raise
PY
echo "✅ Basic functionality test passed"

# ---------- Python ↔ R (rpy2) integration ----------
echo ""
echo "🧪 Testing Python ↔ R integration (rpy2)..."
python3 - <<'PY'
try:
    from rpy2.robjects.packages import importr
    graph = importr("graph")
    rbgl  = importr("RBGL")
    pcalg = importr("pcalg")
    print("✅ Successfully imported R packages via rpy2")
except Exception as e:
    print(f"❌ Python↔R integration test failed: {e}")
    raise
PY
echo "✅ Python ↔ R integration check passed"

echo ""
echo "🎉 Installation completed successfully!"
