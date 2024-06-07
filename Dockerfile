# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required dependencies and Microsoft ODBC driver for SQL Server
RUN apt-get update && \
    apt-get clean && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
        gnupg && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/11/prod.list -o /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    apt-get install -y unixodbc-dev && \
    # Install ODBC Driver 17 for SQL Server
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 && \
    rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Specify command to run on container start
CMD ["python", "app.py"]