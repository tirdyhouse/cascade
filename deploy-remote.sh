#!/bin/bash
set -e

REMOTE_HOST="192.168.56.29"
REMOTE_DIR="/opt/cascade"
PASSWORD="123456"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "🚀 Deploying Cascade to $REMOTE_HOST ..."

# 1. rsync 源码
echo "📦 Syncing source code..."
rsync -az -e "sshpass -p '$PASSWORD' ssh $SSH_OPTS" \
  go.mod go.sum root@$REMOTE_HOST:$REMOTE_DIR/

rsync -az --delete -e "sshpass -p '$PASSWORD' ssh $SSH_OPTS" \
  --include='*.go' --include='static/' --include='static/*' --include='*/' --exclude='*' \
  engine/ root@$REMOTE_HOST:$REMOTE_DIR/engine/

# 2. 远程编译
echo "🔨 Building on remote..."
sshpass -p "$PASSWORD" ssh $SSH_OPTS root@$REMOTE_HOST "
export PATH=\$PATH:/usr/local/go/bin
cd $REMOTE_DIR
go build -o disk-cache ./engine/cmd/disk-cache/
"

# 3. 重启服务
echo "🔄 Restarting service..."
sshpass -p "$PASSWORD" ssh $SSH_OPTS root@$REMOTE_HOST "pkill -f disk-cache 2>/dev/null; sleep 1"
sshpass -p "$PASSWORD" ssh $SSH_OPTS root@$REMOTE_HOST "nohup $REMOTE_DIR/disk-cache -cache-path $REMOTE_DIR/data/storage -metadata-path $REMOTE_DIR/data/meta -max-size 10GB > $REMOTE_DIR/cascade.log 2>&1 &"

sleep 2

# 4. 验证
echo "✅ Verifying..."
sshpass -p "$PASSWORD" ssh $SSH_OPTS root@$REMOTE_HOST "
echo \"  Process: \$(ps aux | grep disk-cache | grep -v grep | awk '{print \$2}')\"
echo \"  Port:    \$(ss -tlnp | grep 9100 | awk '{print \$4}')\"
echo \"  API:     \$(curl -s http://localhost:9100/stats | head -c 100)\"
"

echo ""
echo "✨ Done! Cascade is running on $REMOTE_HOST:9100"
