set -x
T=/workspace/h3-rs/tables; mkdir -p $T
echo "== V2 AdaLN tables"; [ -f $T/v2.tables ] || { ( time timeout 3000 /w/h3-bin adaln-tables --backend cuda --variant fasth3-v2 --model-dir /w/h3-8step --out $T/v2.tables ) 2>&1 | tail -4; }
echo "== stored MXFP8 V2 (h3 quantize --skip-adaln)"; mkdir -p /workspace/weights/h3-8step-mxfp8; [ -f /workspace/weights/h3-8step-mxfp8/transformer.safetensors ] || { ( time timeout 3000 /w/h3-bin quantize --variant fasth3-v2 --model-dir /w/h3-8step --weights mxfp8 --skip-adaln --out /workspace/weights/h3-8step-mxfp8/transformer.safetensors ) 2>&1 | tail -4; }
d=/workspace/weights/h3-8step-mxfp8; [ -f $d/transformer.safetensors ] && [ ! -f $d/.complete ] && { echo "h3-rs $(date -u +%F): h3 quantize --weights mxfp8 --skip-adaln of h3-8step (4c12af4)" > $d/.complete; echo "3da2ddfe1954d9cda4c05b643dc0f26007a655c5-mxfp8" > $d/.fv-revision; echo "h3-rs" > $d/.fv-source; }; ls -la $d
echo "== Omni AdaLN tables"; [ -f $T/omni.tables ] || { ( time timeout 3000 /w/h3-bin adaln-tables --backend cuda --variant fasth3-omni --model-dir /w/h3-base --variant-dir /w/fasth3-omni-8step-v1 --out $T/omni.tables ) 2>&1 | tail -4; }
ls -la $T
du -sh /workspace/weights/* /workspace/h3-rs/* 2>/dev/null | sort -h; du -sh /workspace; df -h /workspace | cat
for t in /workspace/weights/*/; do printf "%-52s complete=%s rev=%s\n" "$t" "$([ -f "$t/.complete" ] && echo y || echo n)" "$(head -c 20 "$t/.fv-revision" 2>/dev/null)"; done
echo "fill done; terminating pod $RUNPOD_POD_ID"; terminate
