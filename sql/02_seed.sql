-- Categorias base (SUNAFIL). El corpus con embeddings se carga desde src/ingest.py.

INSERT INTO categorias (codigo, descripcion) VALUES ('DENUNCIA_LABORAL',    'Denuncia de un trabajador por incumplimiento laboral');
INSERT INTO categorias (codigo, descripcion) VALUES ('SOLICITUD_INSPECCION','Solicitud formal de inspeccion de trabajo');
INSERT INTO categorias (codigo, descripcion) VALUES ('DESCARGO_EMPLEADOR',  'Descargo o alegato presentado por el empleador');
INSERT INTO categorias (codigo, descripcion) VALUES ('RECURSO_APELACION',   'Recurso de apelacion o reconsideracion contra una resolucion');
INSERT INTO categorias (codigo, descripcion) VALUES ('ESCRITO_SUBSANACION', 'Escrito de subsanacion de observaciones');
INSERT INTO categorias (codigo, descripcion) VALUES ('CONSULTA_GENERAL',    'Consulta general o solicitud de informacion');

COMMIT;
