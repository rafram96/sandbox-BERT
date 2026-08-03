-- Esquema del sandbox. Lo ejecuta src/bootstrap.py.

BEGIN EXECUTE IMMEDIATE 'DROP TABLE clasificaciones CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE kb_documentos CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE documentos CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;
/
BEGIN EXECUTE IMMEDIATE 'DROP TABLE categorias CASCADE CONSTRAINTS';
EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;
/

CREATE TABLE categorias (
  id          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  codigo      VARCHAR2(40)  NOT NULL UNIQUE,
  descripcion VARCHAR2(400) NOT NULL
);

CREATE TABLE documentos (
  id          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  expediente  VARCHAR2(60),
  texto       CLOB NOT NULL,
  fuente      VARCHAR2(120) DEFAULT 'mesa_de_partes',
  created_at  TIMESTAMP DEFAULT SYSTIMESTAMP
);

-- corpus etiquetado: centroides del clasificador + vector store del RAG
CREATE TABLE kb_documentos (
  id           NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  texto        CLOB NOT NULL,
  categoria_id NUMBER NOT NULL REFERENCES categorias(id),
  embedding    VECTOR(768, FLOAT32),
  created_at   TIMESTAMP DEFAULT SYSTIMESTAMP
);

CREATE TABLE clasificaciones (
  id                 NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  documento_id       NUMBER NOT NULL REFERENCES documentos(id),
  categoria_base     VARCHAR2(40),
  score_confianza    NUMBER,
  ruta               VARCHAR2(10) NOT NULL,
  escalo_llm         NUMBER(1) DEFAULT 0 NOT NULL,
  top_k_json         CLOB,
  etiqueta_final     VARCHAR2(40),
  revision_humana    NUMBER(1) DEFAULT 0 NOT NULL,
  created_at         TIMESTAMP DEFAULT SYSTIMESTAMP,
  CONSTRAINT chk_ruta CHECK (ruta IN ('rapida','llm'))
);

-- indice vectorial; si el entorno no lo permite, cae a busqueda exacta
BEGIN
  EXECUTE IMMEDIATE
    'CREATE VECTOR INDEX kb_emb_idx ON kb_documentos (embedding) '
    || 'ORGANIZATION INMEMORY NEIGHBOR GRAPH '
    || 'DISTANCE COSINE WITH TARGET ACCURACY 95';
EXCEPTION WHEN OTHERS THEN
  DBMS_OUTPUT.PUT_LINE('sin indice vectorial, se usa busqueda exacta: ' || SQLERRM);
END;
/
