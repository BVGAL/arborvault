ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
CREATE POLICY scoped_documents ON documents
    USING (tenant = current_setting('arborvault.tenant', true)
        AND namespace = current_setting('arborvault.namespace', true))
    WITH CHECK (tenant = current_setting('arborvault.tenant', true)
        AND namespace = current_setting('arborvault.namespace', true));
ALTER TABLE document_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_revisions FORCE ROW LEVEL SECURITY;
CREATE POLICY scoped_revisions ON document_revisions
    USING (tenant = current_setting('arborvault.tenant', true)
        AND namespace = current_setting('arborvault.namespace', true))
    WITH CHECK (tenant = current_setting('arborvault.tenant', true)
        AND namespace = current_setting('arborvault.namespace', true));
ALTER TABLE nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE nodes FORCE ROW LEVEL SECURITY;
CREATE POLICY scoped_nodes ON nodes
    USING (tenant = current_setting('arborvault.tenant', true)
        AND namespace = current_setting('arborvault.namespace', true))
    WITH CHECK (tenant = current_setting('arborvault.tenant', true)
        AND namespace = current_setting('arborvault.namespace', true));
