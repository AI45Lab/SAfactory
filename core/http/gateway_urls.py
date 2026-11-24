from urllib.parse import quote, unquote, urlsplit, urlunsplit


def job_sessions_url(base_url: str, job_id: str) -> str:
    """Build the job's sessions collection from a configured Gateway URL."""
    parts = urlsplit(base_url.strip())
    if not parts.netloc or parts.scheme not in {"http", "https"} or parts.query or parts.fragment:
        raise ValueError("gateway_base_url must be an absolute HTTP(S) base URL without a query or fragment")
    if not job_id.strip():
        raise ValueError("job_id is required for a Gateway session")

    path = parts.path.rstrip("/")
    if path.endswith("/sessions"):
        path = path.removesuffix("/sessions")
        prefix, marker, scoped_job = path.rpartition("/jobs/")
        if marker:
            if unquote(scoped_job) != job_id:
                raise ValueError("gateway_base_url belongs to a different job")
            path = prefix
    root = path or "/v1"
    return urlunsplit((parts.scheme, parts.netloc, f"{root}/jobs/{quote(job_id, safe='')}/sessions", "", ""))
