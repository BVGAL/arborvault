CREATE TABLE document_revisions (
    tenant text NOT NULL,
    namespace text NOT NULL,
    document_id uuid NOT NULL,
    revision integer NOT NULL CHECK (revision > 0),
    title text NOT NULL,
    source_uri text,
    digest text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant, namespace, document_id, revision),
    FOREIGN KEY (tenant, namespace, document_id)
        REFERENCES documents (tenant, namespace, id) ON DELETE CASCADE
);
INSERT INTO document_revisions (tenant,namespace,document_id,revision,title,source_uri,digest)
    SELECT tenant,namespace,id,revision,title,source_uri,digest FROM documents;
ALTER TABLE nodes ADD COLUMN revision integer;
UPDATE nodes n SET revision=d.revision FROM documents d
    WHERE (n.tenant,n.namespace,n.document_id)=(d.tenant,d.namespace,d.id);
ALTER TABLE nodes ALTER COLUMN revision SET NOT NULL;
ALTER TABLE nodes DROP CONSTRAINT nodes_tenant_namespace_document_id_parent_id_fkey;
ALTER TABLE nodes DROP CONSTRAINT nodes_pkey;
ALTER TABLE nodes ADD PRIMARY KEY (tenant,namespace,document_id,revision,id);
ALTER TABLE nodes ADD FOREIGN KEY (tenant,namespace,document_id,revision)
    REFERENCES document_revisions (tenant,namespace,document_id,revision) ON DELETE CASCADE;
ALTER TABLE nodes ADD FOREIGN KEY (tenant,namespace,document_id,revision,parent_id)
    REFERENCES nodes (tenant,namespace,document_id,revision,id) DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX nodes_current ON nodes(tenant,namespace,document_id,revision,ordinal);
