B=/w/h3-bin
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,memory.total --format=csv
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverInfo" | head -4
echo "== check wgpu (linear4 + flash3)"; timeout 2400 $B check --backend wgpu 2>&1 | grep -E "NVIDIA|llvmpipe|checks|FAIL|Error" | head -8
echo "== check wgpu ops (linear3 + flash)"; H3_WGPU_LINEAR=3 H3_WGPU_FLASH=1 timeout 1800 $B check --backend wgpu --ops-only 2>&1 | grep -E "checks|FAIL|Error" | head -6
echo "== check cuda ops"; timeout 900 $B check --backend cuda --ops-only 2>&1 | grep -E "checks|FAIL|Error" | head -4
