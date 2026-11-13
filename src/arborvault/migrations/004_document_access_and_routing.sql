-- All changes and backfills are atomic with the migration transaction.
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;
ALTER TABLE nodes DISABLE ROW LEVEL SECURITY;

CREATE TABLE document_access (
    tenant text NOT NULL,
    namespace text NOT NULL,
    document_id uuid NOT NULL,
    owner text NOT NULL,
    readers text[] NOT NULL,
    writers text[] NOT NULL,
    deleters text[] NOT NULL,
    PRIMARY KEY (tenant,namespace,document_id),
    FOREIGN KEY (tenant,namespace,document_id)
        REFERENCES documents(tenant,namespace,id) ON DELETE CASCADE
);
INSERT INTO document_access
    SELECT tenant,namespace,id,'',ARRAY['*'],ARRAY['*'],ARRAY['*'] FROM documents;
ALTER TABLE document_access ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_access FORCE ROW LEVEL SECURITY;
CREATE POLICY scoped_access ON document_access
    USING (tenant=current_setting('arborvault.tenant',true)
        AND namespace=current_setting('arborvault.namespace',true))
    WITH CHECK (tenant=current_setting('arborvault.tenant',true)
        AND namespace=current_setting('arborvault.namespace',true));

CREATE FUNCTION arborvault_document_allowed(t text,n text,i uuid) RETURNS boolean
LANGUAGE sql STABLE AS $$
    SELECT EXISTS(SELECT 1 FROM document_access a
      WHERE (a.tenant,a.namespace,a.document_id)=(t,n,i) AND (
        (a.owner<>'' AND a.owner=current_setting('arborvault.principal',true))
        OR (current_setting('arborvault.operation',true)='manage'
            AND current_setting('arborvault.manage_acl',true)='true')
        OR CASE current_setting('arborvault.operation',true)
            WHEN 'read' THEN a.readers
            WHEN 'write' THEN a.writers
            WHEN 'delete' THEN a.deleters
            ELSE ARRAY[]::text[] END
          && ARRAY['*',current_setting('arborvault.principal',true)]
      ))
$$;

ALTER POLICY scoped_documents ON documents USING (
    tenant=current_setting('arborvault.tenant',true)
    AND namespace=current_setting('arborvault.namespace',true)
    AND arborvault_document_allowed(tenant,namespace,id)
);
ALTER POLICY scoped_revisions ON document_revisions USING (
    tenant=current_setting('arborvault.tenant',true)
    AND namespace=current_setting('arborvault.namespace',true)
    AND arborvault_document_allowed(tenant,namespace,document_id)
);
ALTER POLICY scoped_nodes ON nodes USING (
    tenant=current_setting('arborvault.tenant',true)
    AND namespace=current_setting('arborvault.namespace',true)
    AND arborvault_document_allowed(tenant,namespace,document_id)
);

ALTER TABLE nodes ADD COLUMN routing_text text NOT NULL DEFAULT '';
DO $$
DECLARE depth integer;
BEGIN
    FOR depth IN REVERSE COALESCE((SELECT max(jsonb_array_length(path)) FROM nodes),0)..0 LOOP
        UPDATE nodes n SET routing_text=left(n.title || ' ' || left(n.content,160) || ' ' ||
            COALESCE((SELECT string_agg(c.routing_text,' ' ORDER BY c.ordinal) FROM nodes c
                WHERE (c.tenant,c.namespace,c.document_id,c.revision,c.parent_id)=
                      (n.tenant,n.namespace,n.document_id,n.revision,n.id)),''),1200)
        WHERE jsonb_array_length(n.path)=depth;
    END LOOP;
END $$;
CREATE INDEX nodes_identity ON nodes(tenant,namespace,id);
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE nodes FORCE ROW LEVEL SECURITY;
