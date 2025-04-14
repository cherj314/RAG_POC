FROM postgres:latest

# Install necessary dependencies for building the extension, including git and ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-server-dev-$PG_MAJOR \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Try updating ca-certificates explicitly
RUN update-ca-certificates

# Download and install the pgvector extension
RUN git clone https://github.com/pgvector/pgvector.git
WORKDIR /pgvector
RUN make install PG_CONFIG=/usr/bin/pg_config

# Set the working directory back
WORKDIR /