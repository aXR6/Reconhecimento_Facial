-- Estrutura inicial do banco de dados

CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    image TEXT,
    faces INTEGER,
    caption TEXT,
    obstruction TEXT,
    recognized TEXT,
    result_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS people (
    id SERIAL PRIMARY KEY,
    name TEXT,
    embedding BYTEA,
    photo BYTEA
);
