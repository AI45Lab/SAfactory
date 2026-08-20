# S3 + LanceDB Storage

Safactory can optionally persist trajectory and environment data to an S3-backed LanceDB data platform through `wt-data-platform-sdk`. SQLite remains the default local strategy; cloud dependencies are kept separately in `requirements-cloud.txt`.

Install the optional dependencies:

```bash
pip install -r requirements-cloud.txt
```

Create a local `.env` file (do not commit credentials) with the data platform connection settings:

```bash
# production or test
WT_SDK_PROFILE=test
WT_SDK_DB_URI=s3://YOUR_DATA_DATABASE
WT_SDK_ENV_CONFIG_DB_URI=s3://YOUR_ENV_CONFIG_DATABASE
WT_SDK_S3_ENDPOINT=https://YOUR_S3_ENDPOINT
WT_SDK_S3_ALLOW_HTTP=true
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
AWS_EC2_METADATA_DISABLED=true
```

Load it into the process environment before starting Safactory:

```bash
set -a
source .env
set +a
```

Then set the gateway `storage_type` to `cloud` and launch Safactory with `--storage-type cloud`. The SDK profile selects Safactory's Cloud landing target; Safactory does not access serving tables. See [AI45Lab/wt-data-platform-sdk](https://github.com/AI45Lab/wt-data-platform-sdk) for the complete configuration and table documentation.
