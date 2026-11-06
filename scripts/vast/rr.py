"""Talk to a rental through R2 (no SSH): the box polls for numbered command
scripts and uploads each one's output. Run id from $H3_RUN.

    rr.py cmd N FILE         queue FILE as command N
    rr.py out N | log        command N's output | the boot log
    rr.py upload SRC KEY     put a local file under the run
    rr.py download KEY DST
    rr.py presign-get KEY | presign-put KEY   6-hour URLs for the box
"""
import sys, os
exec(open(os.path.join(os.path.dirname(__file__), "r2.py")).read())
run = os.environ["H3_RUN"]; B = os.environ.get("H3_BUCKET", "h3-runs")
op = sys.argv[1]
if op == 'cmd':
    s3.upload_file(sys.argv[3], B, f'{run}/cmd/{sys.argv[2]}.sh'); print('queued cmd', sys.argv[2])
elif op in ('out', 'log'):
    key = f'{run}/out/{sys.argv[2]}.log' if op == 'out' else f'{run}/boot.log'
    try: print(s3.get_object(Bucket=B, Key=key)['Body'].read().decode(errors='replace'))
    except s3.exceptions.NoSuchKey: print('(not yet)')
elif op == 'presign-put':
    print(s3.generate_presigned_url('put_object', Params={'Bucket': B, 'Key': f'{run}/{sys.argv[2]}'}, ExpiresIn=6*3600))
elif op == 'presign-get':
    print(s3.generate_presigned_url('get_object', Params={'Bucket': B, 'Key': f'{run}/{sys.argv[2]}'}, ExpiresIn=6*3600))
elif op == 'upload':
    s3.upload_file(sys.argv[2], B, f'{run}/{sys.argv[3]}'); print('uploaded', sys.argv[3])
elif op == 'download':
    s3.download_file(B, f'{run}/{sys.argv[2]}', sys.argv[3]); print('downloaded', sys.argv[3])
