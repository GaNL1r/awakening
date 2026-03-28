#!/bin/bash

TARGET="$1"
shift
ARGS=("${@:3}")


echo "[GUARD] target: $TARGET"
echo "[GUARD] args: ${ARGS[*]}"
echo "[GUARD] Starting monitor..."

while true; do
    echo "[GUARD] Launching program..."
    "$TARGET" "${ARGS[@]}"
    RET=$?

    if [ $RET -eq 0 ]; then
        echo "[GUARD] Program exited normally. Stopping guard."
        exit 0
    fi

    echo "[GUARD] Crash detected, restarting in 1 second..."
    sleep 1
done
