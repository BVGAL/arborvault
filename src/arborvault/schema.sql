CREATE TABLE IF NOT EXISTS documents (
    tenant text NOT NULL,
    namespace text NOT NULL,
    id uuid NOT NULL,
    source_key text NOT NULL,
    title text NOT NULL,
    source_uri text,
    revision integer NOT NULL CHECK (revision > 0),
    digest text NOT NULL,
    PRIMARY KEY (tenant, namespace, id),
    UNIQUE (tenant, namespace, source_key)
);
CREATE TABLE IF NOT EXISTS nodes (
    tenant text NOT NULL,
    namespace text NOT NULL,
    document_id uuid NOT NULL,
    id uuid NOT NULL,
    parent_id uuid,
    title text NOT NULL,
    content text NOT NULL,
    path jsonb NOT NULL,
    ordinal integer NOT NULL,
    locator jsonb NOT NULL,
    search tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', title), 'A') ||
        setweight(to_tsvector('simple', content), 'B')
    ) STORED,
    PRIMARY KEY (tenant, namespace, document_id, id),
    FOREIGN KEY (tenant, namespace, document_id)
        REFERENCES documents (tenant, namespace, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant, namespace, document_id, parent_id)
        REFERENCES nodes (tenant, namespace, document_id, id)
        DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes (tenant, namespace, parent_id);
CREATE INDEX IF NOT EXISTS nodes_search ON nodes USING gin(search);
