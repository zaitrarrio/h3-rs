B=/w/h3-bin
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,memory.total --format=csv
vulkaninfo 2>/dev/null | grep -E "deviceName|driverInfo|shaderFloat16" | head -6
echo "== check wgpu (f32)"; timeout 2400 $B check --backend wgpu 2>&1 | grep -E "NVIDIA|llvmpipe|checks|FAIL|Error" | head -6
echo "== check wgpu H3_WGPU_F16=attn"; H3_WGPU_F16=attn timeout 2400 $B check --backend wgpu 2>&1 | grep -E "\+f16|checks|FAIL|Error" | head -8
echo "== check wgpu ops H3_WGPU_F16=1"; H3_WGPU_F16=1 timeout 1800 $B check --backend wgpu --ops-only 2>&1 | grep -E "\+f16|checks|FAIL|Error" | head -8
echo "== check cuda ops"; timeout 900 $B check --backend cuda --ops-only 2>&1 | grep -E "checks|FAIL|Error" | head -4
