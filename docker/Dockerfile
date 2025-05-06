FROM postgres:latest

# Install necessary dependencies for building the extension in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-server-dev-$PG_MAJOR \
    git \
    ca-certificates \
    && update-ca-certificates \
    && git clone --depth 1 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make install PG_CONFIG=/usr/bin/pg_config \
    && cd .. \
    && rm -rf pgvector \
    && apt-get purge -y build-essential postgresql-server-dev-$PG_MAJOR git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*