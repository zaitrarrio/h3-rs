. /root/common.sh
echo "== full-width DiT parity, first 8 blocks of h3-base"
(apt-get install -y -qq python3-venv python3-pip git >/dev/null 2>&1 && echo apt-ok)
python3 -m venv /root/pv && . /root/pv/bin/activate
pip install -q torch==2.12.0 torchvision 2>&1 | grep -v notice | tail -2
pip install -q "transformers>=5.15" diffusers safetensors einops numpy scipy cloudpickle imageio remote_pdb ftfy psutil 2>&1 | grep -v notice | tail -2
git clone -q https://github.com/hao-ai-lab/FastVideo /root/FastVideo && (cd /root/FastVideo && git log --oneline -1)
for L in 2 8; do
  ( time PYTHONPATH=/root/FastVideo timeout 2400 python /root/full_parity.py --transformer /w/h3-base/transformer --layers $L --out /root/res/dit_full_$L.safetensors --device cuda ) 2>&1 | grep -E "loaded|wrote|Error|error|real" | tail -4
  for be in cuda cpu; do echo "-- parity-dit $be, $L blocks"; timeout 2400 $B parity-dit --backend $be --transformer /w/h3-base/transformer --fixture /root/res/dit_full_$L.safetensors 2>&1 | grep -E "rel|parity|Error"; done
done
deactivate
