#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# CryingSense — Raspberry Pi 3B+ one-shot setup
#
# Run once after copying this folder to the Pi:
#   chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/.venv"
SERVICE_NAME="cryingsense"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "═══════════════════════════════════════════════════════════"
echo "  CryingSense — Raspberry Pi 3B+ Setup"
echo "═══════════════════════════════════════════════════════════"

# ── System packages ─────────────────────────────────────────
echo "[1/5] Installing system dependencies…"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv python3-pip python3-dev \
    portaudio19-dev libsndfile1 \
    libatlas-base-dev libopenblas-dev

# ── Python venv ─────────────────────────────────────────────
echo "[2/5] Creating virtual environment…"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# ── Python deps ─────────────────────────────────────────────
echo "[3/5] Installing Python packages…"
pip install -r "$DIR/requirements.txt"

# ── Directories ─────────────────────────────────────────────
echo "[4/5] Creating directories…"
mkdir -p "$DIR/saved_models" "$DIR/recordings"

# ── systemd service ─────────────────────────────────────────
echo "[5/5] Installing systemd service…"
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=CryingSense Cry Detection Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$DIR
ExecStart=$VENV/bin/python $DIR/pipeline.py
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Place your model in: $DIR/saved_models/"
echo "    2. Create .env with Firebase settings (see config.py)"
echo "    3. Start:  sudo systemctl start $SERVICE_NAME"
echo "    4. Logs:   journalctl -u $SERVICE_NAME -f"
echo "═══════════════════════════════════════════════════════════"
