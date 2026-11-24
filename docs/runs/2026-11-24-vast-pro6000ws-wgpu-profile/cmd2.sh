B=/w/h3-bin
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,memory.total --format=csv
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverInfo|apiVersion" | head -6
echo "== check wgpu"; timeout 1800 $B check --backend wgpu --ops-only 2>&1 | grep -E "NVIDIA|llvmpipe|checks|FAIL|Error" | head -8
echo "== check cuda"; timeout 900 $B check --backend cuda --ops-only 2>&1 | grep -E "checks|FAIL|Error" | head -4
