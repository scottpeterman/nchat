#!/usr/bin/env bash
#
# snapshot.sh — make one coherent, share-ready archive of a project tree.
#
# Why this exists: zipping subdirs separately off a remote box gives you an
# incoherent bundle (spine without its wiring, modules with no integration,
# and — worse — your real SQLite DB along for the ride). This takes ONE
# snapshot of the whole root, drops build/dep/cache noise and data files,
# and writes a MANIFEST (file list + sha256 + git state) INTO the archive so
# the next person to open it can verify exactly what they got.
#
# Portable across macOS (BSD) and Linux (GNU): tar -T/-r/-C, find, gzip are
# common to both; stat/sha are probed below.
#
# Usage:
#   ./snapshot.sh                       # snapshot $PWD
#   ./snapshot.sh -r ~/code/nchat       # snapshot a specific root
#   ./snapshot.sh --include-dist        # keep frontend/dist (default: drop)
#   ./snapshot.sh --include-db          # keep *.db/*.sqlite (default: drop)
#   ./snapshot.sh --out ~/Desktop       # where to write the archive
#   ./snapshot.sh --dry-run             # print the file list, build nothing
#
set -euo pipefail

# --- args -------------------------------------------------------------------
ROOT="$PWD"; OUT_DIR=""; INCLUDE_DIST=false; INCLUDE_DB=false; DRY_RUN=false
while [ $# -gt 0 ]; do
  case "$1" in
    -r|--root)        ROOT="$2"; shift 2 ;;
    --out)            OUT_DIR="$2"; shift 2 ;;
    --include-dist)   INCLUDE_DIST=true; shift ;;
    --include-db)     INCLUDE_DB=true; shift ;;
    --dry-run)        DRY_RUN=true; shift ;;
    -h|--help)        sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
ROOT="$(cd "$ROOT" && pwd)"                       # absolutize
NAME="$(basename "$ROOT")"
TS="$(date +%Y%m%d-%H%M%S)"
# Default: write the archive OUTSIDE the tree being archived, so a re-run never
# tars a previous snapshot and `find` never trips over it.
[ -n "$OUT_DIR" ] || OUT_DIR="$(dirname "$ROOT")"
OUT="$OUT_DIR/${NAME}-snapshot-${TS}.tar.gz"

# --- portable stat-size + sha256 -------------------------------------------
if stat -f%z "$ROOT" >/dev/null 2>&1; then statsize() { stat -f%z "$1"; }   # BSD/macOS
else                                       statsize() { stat -c%s "$1"; }; fi # GNU/Linux
if   command -v shasum    >/dev/null 2>&1; then sha() { shasum -a 256 "$1" | awk '{print $1}'; }
elif command -v sha256sum >/dev/null 2>&1; then sha() { sha256sum  "$1" | awk '{print $1}'; }
else                                            sha() { echo "----------------------------------------------------------------"; }; fi

# --- what to drop -----------------------------------------------------------
# Directory NAMES pruned anywhere in the tree.
PRUNE_DIRS=(.git .venv venv env node_modules __pycache__ .pytest_cache \
            .mypy_cache .ruff_cache .idea .vscode .DS_Store .cache .next .turbo)
$INCLUDE_DIST || PRUNE_DIRS+=(dist build .output)
# File GLOBS excluded anywhere.
EXCLUDE_GLOBS=('*.pyc' '*.pyo' '*.pyd' '*.log' '.DS_Store' \
               '*.tar.gz' '*.tgz' '*.zip' '*-snapshot-*')
$INCLUDE_DB || EXCLUDE_GLOBS+=('*.db' '*.sqlite' '*.sqlite3' '*.db-journal' '*.db-wal')

# --- build the include list (one source of truth for archive + manifest) ----
cd "$ROOT"
prune_expr=()
for d in "${PRUNE_DIRS[@]}"; do prune_expr+=(-name "$d" -o); done
unset 'prune_expr[${#prune_expr[@]}-1]'          # drop trailing -o
file_expr=(-type f)
for g in "${EXCLUDE_GLOBS[@]}"; do file_expr+=(! -name "$g"); done

LIST="$(mktemp)"; MAN="$(mktemp)"
trap 'rm -f "$LIST" "$MAN"' EXIT
find . \( -type d \( "${prune_expr[@]}" \) -prune \) -o \( "${file_expr[@]}" -print \) \
  | LC_ALL=C sort > "$LIST"

COUNT="$(wc -l < "$LIST" | tr -d ' ')"
[ "$COUNT" -gt 0 ] || { echo "nothing to snapshot under $ROOT" >&2; exit 1; }

# --- manifest header: provenance + git state (the coherence breadcrumb) -----
{
  echo "# snapshot manifest"
  echo "# project : $NAME"
  echo "# root    : $ROOT"
  echo "# created : $(date '+%Y-%m-%d %H:%M:%S %z')  on $(hostname)"
  if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "# git     : branch $(git -C "$ROOT" rev-parse --abbrev-ref HEAD) @ $(git -C "$ROOT" rev-parse --short HEAD)"
    dirty="$(git -C "$ROOT" status --short)"
    if [ -n "$dirty" ]; then
      echo "# git     : WORKING TREE DIRTY — uncommitted changes below are what got snapshotted"
      echo "$dirty" | sed 's/^/#   /'
    else
      echo "# git     : working tree clean"
    fi
  else
    echo "# git     : (not a git repo)"
  fi
  echo "# files   : $COUNT"
  echo "#"
  echo "# sha256                                                            bytes  path"
} > "$MAN"

TOTAL=0
while IFS= read -r f; do
  sz="$(statsize "$f")"; TOTAL=$((TOTAL + sz))
  printf '%s  %9d  %s\n' "$(sha "$f")" "$sz" "${f#./}" >> "$MAN"
done < "$LIST"

# --- nchat-specific coherence check (delete or adapt for other projects) ----
# Catches the exact failure that motivated this: a module present in the tree
# but never imported/wired. Warn loudly, never block — you may have staged it
# deliberately. Pattern: "if FILE exists, GREP must match somewhere".
warn() { printf '  \033[33m! %s\033[0m\n' "$*"; }
checks_ran=false
coherence_check() {
  checks_ran=true
  local hit=false
  _need() {  # _need <file-that-exists> <grep-pattern> <where> <human msg>
    [ -f "$1" ] || return 0
    if ! grep -rqE "$2" $3 2>/dev/null; then warn "$4"; hit=true; fi
  }
  _need backend/search.py 'import .*search|from .*search'   backend/main.py \
        "search.py present but main.py never imports it — search is unwired"
  _need backend/tts.py    '\btts\b|import tts|from .*tts'   backend/main.py \
        "tts.py present but main.py never imports it — TTS routes unwired"
  _need backend/stt.py    '\bstt\b|import stt|from .*stt'   backend/main.py \
        "stt.py present but main.py never imports it — STT route unwired"
  _need frontend/src/voice.js "voice" "frontend/src/components frontend/src/App.jsx" \
        "voice.js present but no component imports it — voice UI unwired"
  $hit || echo "  coherence: wiring checks passed"
}
[ "$DRY_RUN" = true ] || coherence_check

# --- emit -------------------------------------------------------------------
HUMAN_TOTAL="$(awk -v b="$TOTAL" 'BEGIN{u="B KB MB GB";split(u,a," ");i=1;while(b>=1024&&i<4){b/=1024;i++}printf "%.1f %s",b,a[i]}')"
if [ "$DRY_RUN" = true ]; then
  echo "DRY RUN — $COUNT files, $HUMAN_TOTAL"
  sed 's/^/  /' "$LIST"
  echo "  (manifest preview)"; sed 's/^/  /' "$MAN" | head -20
  exit 0
fi

# Stage MANIFEST.txt so it lands at the archive's top level without polluting
# the source tree. Build uncompressed, append the manifest, then gzip — the one
# sequence that works the same on BSD and GNU tar (append needs an uncompressed
# archive; no --transform / -s rename tricks required).
STAGE="$(mktemp -d)"; trap 'rm -f "$LIST" "$MAN"; rm -rf "$STAGE"' EXIT
cp "$MAN" "$STAGE/MANIFEST.txt"
tar -cf  "$STAGE/base.tar" -C "$ROOT"  -T "$LIST"
tar -rf  "$STAGE/base.tar" -C "$STAGE" MANIFEST.txt
gzip -c  "$STAGE/base.tar" > "$OUT"

echo "  wrote $OUT"
echo "  $COUNT files, $HUMAN_TOTAL (excludes deps/build/caches$($INCLUDE_DB && echo '' || echo '/*.db'))"
echo "  MANIFEST.txt is at the archive root — open it first to confirm coherence."