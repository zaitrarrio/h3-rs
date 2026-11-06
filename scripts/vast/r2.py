"""S3 client for Cloudflare R2 from a Cloudflare API token (R2 read/write).

R2 derives S3 credentials from the token: access key = token id, secret =
sha256(token). Needs boto3. Env: CLOUDFLARE_API_KEY (the token),
CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_TOKEN_ID.
"""
import hashlib, os
import boto3
from botocore.config import Config
ACCOUNT = os.environ["CLOUDFLARE_ACCOUNT_ID"]
tok = os.environ["CLOUDFLARE_API_KEY"]
s3 = boto3.client("s3", endpoint_url=f"https://{ACCOUNT}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["CLOUDFLARE_TOKEN_ID"],
    aws_secret_access_key=hashlib.sha256(tok.encode()).hexdigest(),
    region_name="auto", config=Config(signature_version="s3v4"))
