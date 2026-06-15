#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-mjh}"
REMOTE_HOST="${REMOTE_HOST:-192.168.1.94}"
REMOTE_DIR="${REMOTE_DIR:-/home/mjh/WorkSpace/Weather-Platform/backend/camera}"
LOCAL_DIR="${LOCAL_DIR:-/home/wdz/BT/Stage1/rain_retrieval/data/camera}"
SSH_KEY="${SSH_KEY:-/home/wdz/.ssh/id_rsa}"
LOCK_FILE="${LOCK_FILE:-/tmp/wdz_camera_pull.lock}"
MODE="${MODE:-latest}"  # latest or all

LOCK_TIMEOUT="${LOCK_TIMEOUT:-300}"

if [[ -f "$LOCK_FILE" ]]; then
  lock_mtime="$(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0)"
  lock_age="$(($(date +%s) - lock_mtime))"
  if [[ "$lock_age" -lt "$LOCK_TIMEOUT" ]]; then
    echo "$(date): sync is already running, skip"
    exit 1
  fi
  echo "$(date): stale lock detected, remove and continue"
  rm -f "$LOCK_FILE"
fi

touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

mkdir -p "$LOCAL_DIR"

ssh_cmd=(
  /usr/bin/ssh
  -i "$SSH_KEY"
  -o BatchMode=yes
  -o ConnectTimeout=15
)

remote="${REMOTE_USER}@${REMOTE_HOST}"

if [[ "$MODE" == "latest" ]]; then
  latest_file="$("${ssh_cmd[@]}" "$remote" \
    "find '$REMOTE_DIR' -maxdepth 1 -name 'capture_*.jpg' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-")"

  if [[ -z "$latest_file" ]]; then
    echo "$(date): no remote capture_*.jpg found in $REMOTE_DIR"
    exit 0
  fi

  /usr/bin/rsync -av --partial -e "${ssh_cmd[*]}" "$remote:$latest_file" "$LOCAL_DIR/"
else
  /usr/bin/rsync -av --partial --include='capture_*.jpg' --exclude='*' \
    -e "${ssh_cmd[*]}" "$remote:$REMOTE_DIR/" "$LOCAL_DIR/"
fi

echo "$(date): camera sync finished: $remote:$REMOTE_DIR -> $LOCAL_DIR"
