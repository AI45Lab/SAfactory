FROM python:3.11-slim

# Geo3K runner dependencies are installed at image build time so runtime
# containers do not need outbound network access.
RUN python -m pip install \
    --no-cache-dir \
    --disable-pip-version-check \
    requests==2.32.5 \
    sympy==1.14.0 \
    pylatexenc==2.10
