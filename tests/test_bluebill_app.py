#!/usr/bin/env python3
"""
Unit Tests for BlueBill App v4.0
Tests for SmartDoc and Fiscal Classifier endpoints
"""

import pytest
import json
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from io import BytesIO
import numpy as np

# Import the main application
# Note: You may need to adjust the import path based on your project structure
from bluebill_app import app, persistence_manager, documents, faiss_indexes, global_chunk_metadata, \
    document_fingerprints

# Test client
client = TestClient(app)


# ===============================================================================
# FIXTURES AND SETUP
# ===============================================================================

@pytest.fixture(scope="function")
def clean_database():
    """Create a clean test database for each test"""
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()

    # Patch the database path
    with patch('bluebill_app.DATABASE_PATH', temp_db.name):
        # Initialize clean database
        test_persistence = type(persistence_manager)(temp_db.name)

        # Clear global variables
        documents.clear()
        faiss_indexes.clear()
        global_chunk_metadata.clear()
        document_fingerprints.clear()

        yield test_persistence

    # Cleanup
    os.unlink(temp_db.name)


@pytest.fixture
def mock_vllm():
    """Mock vLLM client for testing"""
    with patch('bluebill_app.vllm_client') as mock_client:
        mock_client.generate.return_value = "Mocked response from vLLM"
        mock_client.base_url = "http://mock-vllm:8000"
        mock_client.model_name = "mock-model"
        mock_client.max_tokens = 300
        mock_client.temperature = 0.3
        yield mock_client


@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer for testing"""
    with patch('bluebill_app.embedding_model') as mock_model:
        # Mock embeddings as random vectors
        mock_model.encode.return_value = np.random.rand(5, 384).astype(np.float32)
        yield mock_model


@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing uploads"""
    # Create a simple text file (we'll mock PDF processing)
    content = b"Sample PDF content for testing document upload functionality."
    file_obj = BytesIO(content)
    file_obj.name = "test_document.pdf"
    return file_obj


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing uploads"""
    content = b"Sample text content for testing document upload functionality. This is a longer text to test chunking."
    file_obj = BytesIO(content)
    file_obj.name = "test_document.txt"
    return file_obj


@pytest.fixture
def sample_factura_data():
    """Sample invoice data for fiscal classifier tests"""
    return {
        "identificacion": {
            "numero_factura": "F001-2024",
            "fecha_emision": "2024-01-15"
        },
        "conceptos": [
            {
                "descripcion": "Servicios de consultoría",
                "importe": 1000.0,
                "iva": 210.0
            }
        ],
        "receptor": {
            "nombre": "Test Company S.L.",
            "nif": "B12345678"
        },
        "emisor": {
            "nombre": "Consultora Test S.L.",
            "nif": "B87654321"
        },
        "importes": {
            "base_imponible": 1000.0,
            "iva": 210.0,
            "total": 1210.0
        }
    }


# ===============================================================================
# SYSTEM ENDPOINTS TESTS
# ===============================================================================

class TestSystemEndpoints:
    """Test system-level endpoints"""

    def test_root_endpoint(self):
        """Test root information endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "Unified API Server"
        assert data["version"] == "4.0.0"
        assert "available_modules" in data
        assert "persistence" in data

    def test_health_check(self, mock_vllm):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert data["version"] == "4.0.0"
        assert "components" in data

    def test_capabilities_endpoint(self):
        """Test capabilities endpoint"""
        response = client.get("/capabilities")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == "4.0.0"
        assert "available_modules" in data
        assert "endpoints" in data
        assert "system" in data["endpoints"]


# ===============================================================================
# SMARTDOC ENDPOINTS TESTS
# ===============================================================================

class TestSmartDocEndpoints:
    """Test SmartDoc document management endpoints"""

    @patch('bluebill_app.extract_text')
    @patch('bluebill_app.chunk_text')
    @patch('bluebill_app.create_embeddings')
    def test_upload_document_success(self, mock_embeddings, mock_chunk, mock_extract,
                                     mock_vllm, mock_embedding_model, clean_database):
        """Test successful document upload"""
        # Setup mocks
        mock_extract.return_value = "Sample document content for testing"
        mock_chunk.return_value = ["Sample document", "content for testing"]
        mock_embeddings.return_value = np.random.rand(2, 384).astype(np.float32)

        # Create test file
        test_file = ("test.txt", b"Sample content", "text/plain")

        response = client.post(
            "/smartdoc/upload",
            files={"file": test_file}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] is not None
        assert "successfully" in data["message"]
        assert data["persistence_status"] in ["saved", "error"]

    def test_upload_document_no_content(self, mock_vllm, mock_embedding_model):
        """Test upload with no extractable content"""
        with patch('bluebill_app.extract_text', return_value=""):
            test_file = ("empty.txt", b"", "text/plain")

            response = client.post(
                "/smartdoc/upload",
                files={"file": test_file}
            )

            assert response.status_code == 400
            assert "No extractable text" in response.json()["detail"]

    def test_list_documents_empty(self, clean_database):
        """Test listing documents when none exist"""
        response = client.get("/smartdoc/documents")
        assert response.status_code == 200
        assert response.json() == []

    @patch('bluebill_app.documents', {1: {
        "filename": "test.txt",
        "text": "Sample content",
        "chunks": ["Sample", "content"],
        "upload_timestamp": "2024-01-01T00:00:00"
    }})
    def test_list_documents_with_data(self):
        """Test listing documents with existing data"""
        response = client.get("/smartdoc/documents")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["doc_id"] == 1
        assert data[0]["filename"] == "test.txt"

    def test_document_summary_not_found(self, mock_vllm):
        """Test summary for non-existent document"""
        response = client.get("/smartdoc/document/999/summary")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch('bluebill_app.documents', {1: {
        "filename": "test.txt",
        "text": "Sample content for summarization testing",
        "chunks": ["Sample content", "for summarization testing"]
    }})
    def test_document_summary_success(self, mock_vllm):
        """Test successful document summary"""
        mock_vllm.generate.return_value = "This is a test summary"

        response = client.get("/smartdoc/document/1/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["doc_id"] == 1
        assert data["summary"] == "This is a test summary"

    def test_query_document_not_found(self, mock_vllm, mock_embedding_model):
        """Test query for non-existent document"""
        query_data = {"query": "What is this about?"}
        response = client.post("/smartdoc/document/999/query", json=query_data)
        assert response.status_code == 404

    @patch('bluebill_app.documents', {1: {
        "filename": "test.txt",
        "text": "Sample content",
        "chunks": ["Sample content for testing queries"]
    }})
    @patch('bluebill_app.faiss_indexes')
    def test_query_document_success(self, mock_indexes, mock_vllm, mock_embedding_model):
        """Test successful document query"""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.search.return_value = (np.array([[0.1]]), np.array([[0]]))
        mock_indexes.__getitem__.return_value = mock_index
        mock_indexes.__contains__.return_value = True

        mock_vllm.generate.return_value = "This is the answer"

        query_data = {"query": "What is this about?"}
        response = client.post("/smartdoc/document/1/query", json=query_data)
        assert response.status_code == 200

        data = response.json()
        assert data["doc_id"] == 1
        assert data["query"] == "What is this about?"
        assert data["answer"] == "This is the answer"

    def test_query_all_no_documents(self, mock_vllm, mock_embedding_model):
        """Test global query with no documents"""
        query_data = {"query": "What is this about?"}
        response = client.post("/smartdoc/query_all", json=query_data)
        assert response.status_code == 404
        assert "No documents found" in response.json()["detail"]

    def test_persistence_stats(self):
        """Test persistence statistics endpoint"""
        response = client.get("/smartdoc/persistence_stats")
        assert response.status_code == 200

        data = response.json()
        assert "database_path" in data
        assert "total_documents" in data
        assert "database_size_mb" in data

    def test_create_backup(self):
        """Test manual backup creation"""
        response = client.post("/smartdoc/backup")
        assert response.status_code == 200

        data = response.json()
        assert "backup completed" in data["message"]
        assert "timestamp" in data

    def test_delete_document_not_found(self):
        """Test deleting non-existent document"""
        response = client.delete("/smartdoc/document/999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @patch('bluebill_app.documents', {1: {
        "filename": "test.txt",
        "text": "Sample content",
        "fingerprint": "sample_fingerprint"
    }})
    @patch('bluebill_app.faiss_indexes', {1: Mock()})
    @patch('bluebill_app.document_fingerprints', {"sample_fingerprint": 1})
    def test_delete_document_success(self, clean_database):
        """Test successful document deletion"""
        with patch('bluebill_app.remove_document_from_global_index'):
            response = client.delete("/smartdoc/document/1")
            assert response.status_code == 200

            data = response.json()
            assert "deleted successfully" in data["message"]
            assert data["doc_id"] == 1


# ===============================================================================
# FISCAL CLASSIFIER ENDPOINTS TESTS
# ===============================================================================

class TestFiscalClassifierEndpoints:
    """Test Fiscal Classifier endpoints"""

    @patch('bluebill_app.AVAILABLE_MODULES', {"fiscal_classifier": True})
    @patch('bluebill_app.fiscal_classifier')
    def test_validate_factura_success(self, mock_classifier, sample_factura_data):
        """Test successful invoice validation"""
        mock_classifier.validate_factura_input.return_value = None  # No exception = valid

        response = client.post("/fiscal/validate", json=sample_factura_data)
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is True
        assert data["conceptos_count"] == 1
        assert "identificacion" in data["sections_present"]

    @patch('bluebill_app.AVAILABLE_MODULES', {"fiscal_classifier": True})
    @patch('bluebill_app.fiscal_classifier')
    def test_validate_factura_invalid(self, mock_classifier, sample_factura_data):
        """Test invoice validation with invalid data"""
        mock_classifier.validate_factura_input.side_effect = ValueError("Invalid invoice structure")

        response = client.post("/fiscal/validate", json=sample_factura_data)
        assert response.status_code == 200

        data = response.json()
        assert data["valid"] is False
        assert "Invalid invoice structure" in data["error"]

    def test_validate_factura_module_unavailable(self, sample_factura_data):
        """Test validation when fiscal classifier is unavailable"""
        with patch('bluebill_app.AVAILABLE_MODULES', {"fiscal_classifier": False}):
            response = client.post("/fiscal/validate", json=sample_factura_data)
            assert response.status_code == 503
            assert "not available" in response.json()["detail"]

    @patch('bluebill_app.AVAILABLE_MODULES', {"fiscal_classifier": True})
    @patch('bluebill_app.fiscal_classifier')
    @patch('bluebill_app.AEATConsultaScraper')
    @patch('bluebill_app.EnhancedFiscalClassifier')
    def test_classify_fiscal_expense_success(self, mock_enhanced_classifier, mock_scraper,
                                             mock_fiscal_classifier, sample_factura_data):
        """Test successful fiscal expense classification"""
        # Mock scraper results
        mock_scraper_instance = Mock()
        mock_scraper_instance.search_comprehensive.return_value = ["mock_result"]
        mock_scraper.return_value = mock_scraper_instance

        # Mock classifier results
        mock_classifier_instance = Mock()
        mock_classifier_instance.classify_expense_with_precedents.return_value = {
            "clasificacion": {"codigo_principal": "G001"},
            "deducibilidad": {"porcentaje": 100},
            "oportunidades_fiscales": [],
            "alertas_cumplimiento": [],
            "analisis_relacion_comercial": {},
            "consultas_vinculantes_aplicables": [],
            "confidence_score": 0.95
        }
        mock_enhanced_classifier.return_value = mock_classifier_instance

        response = client.post("/fiscal/classify", json=sample_factura_data)
        assert response.status_code == 200

        data = response.json()
        assert "clasificacion" in data
        assert data["clasificacion"]["codigo_principal"] == "G001"
        assert data["confidence_score"] == 0.95

    @patch('bluebill_app.AVAILABLE_MODULES', {"fiscal_classifier": True})
    @patch('bluebill_app.fiscal_classifier')
    def test_load_consultas_success(self, mock_classifier):
        """Test successful loading of external consultas"""
        mock_classifier.load_consultas_from_data.return_value = True

        consultas_data = [
            {"id": 1, "titulo": "Test consulta", "contenido": "Test content"}
        ]

        response = client.post("/fiscal/load_consultas", json=consultas_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert data["consultas_loaded"] == 1

    @patch('bluebill_app.AVAILABLE_MODULES', {"fiscal_classifier": True})
    @patch('bluebill_app.fiscal_classifier')
    def test_load_consultas_failure(self, mock_classifier):
        """Test failed loading of external consultas"""
        mock_classifier.load_consultas_from_data.return_value = False

        consultas_data = [{"id": 1, "titulo": "Test"}]

        response = client.post("/fiscal/load_consultas", json=consultas_data)
        assert response.status_code == 500
        assert "Failed to load" in response.json()["detail"]


# ===============================================================================
# INTEGRATION TESTS
# ===============================================================================

class TestIntegrationScenarios:
    """Test complete workflows and integration scenarios"""

    @patch('bluebill_app.extract_text')
    @patch('bluebill_app.chunk_text')
    @patch('bluebill_app.create_embeddings')
    def test_complete_document_workflow(self, mock_embeddings, mock_chunk, mock_extract,
                                        mock_vllm, mock_embedding_model, clean_database):
        """Test complete document upload, query, and delete workflow"""
        # Setup mocks
        mock_extract.return_value = "Sample document content for complete workflow testing"
        mock_chunk.return_value = ["Sample document content", "for complete workflow testing"]
        mock_embeddings.return_value = np.random.rand(2, 384).astype(np.float32)

        # 1. Upload document
        test_file = ("workflow_test.txt", b"Sample content", "text/plain")
        upload_response = client.post("/smartdoc/upload", files={"file": test_file})
        assert upload_response.status_code == 200
        doc_id = upload_response.json()["doc_id"]

        # 2. List documents
        list_response = client.get("/smartdoc/documents")
        assert len(list_response.json()) == 1

        # 3. Get summary
        with patch('bluebill_app.documents', {doc_id: {
            "filename": "workflow_test.txt",
            "text": "Sample document content for complete workflow testing",
            "chunks": ["Sample document content", "for complete workflow testing"]
        }}):
            summary_response = client.get(f"/smartdoc/document/{doc_id}/summary")
            assert summary_response.status_code == 200

        # 4. Query document
        with patch('bluebill_app.documents', {doc_id: {
            "filename": "workflow_test.txt",
            "text": "Sample content",
            "chunks": ["Sample content for testing"]
        }}), patch('bluebill_app.faiss_indexes') as mock_indexes:
            mock_index = Mock()
            mock_index.search.return_value = (np.array([[0.1]]), np.array([[0]]))
            mock_indexes.__getitem__.return_value = mock_index
            mock_indexes.__contains__.return_value = True

            query_data = {"query": "What is this about?"}
            query_response = client.post(f"/smartdoc/document/{doc_id}/query", json=query_data)
            assert query_response.status_code == 200

        # 5. Delete document
        with patch('bluebill_app.documents', {doc_id: {
            "filename": "workflow_test.txt",
            "fingerprint": "test_fingerprint"
        }}), patch('bluebill_app.remove_document_from_global_index'):
            delete_response = client.delete(f"/smartdoc/document/{doc_id}")
            assert delete_response.status_code == 200

    def test_duplicate_document_handling(self, mock_vllm, mock_embedding_model, clean_database):
        """Test duplicate document detection and replacement"""
        with patch('bluebill_app.extract_text') as mock_extract, \
                patch('bluebill_app.chunk_text') as mock_chunk, \
                patch('bluebill_app.create_embeddings') as mock_embeddings:
            # Setup mocks
            mock_extract.return_value = "Duplicate document content"
            mock_chunk.return_value = ["Duplicate document", "content"]
            mock_embeddings.return_value = np.random.rand(2, 384).astype(np.float32)

            # Upload first document
            test_file1 = ("doc1.txt", b"Same content", "text/plain")
            response1 = client.post("/smartdoc/upload", files={"file": test_file1})
            assert response1.status_code == 200

            # Upload duplicate document
            test_file2 = ("doc2.txt", b"Same content", "text/plain")
            response2 = client.post("/smartdoc/upload", files={"file": test_file2})
            assert response2.status_code == 200

            # Should indicate duplicate was replaced
            assert "duplicate_info" in response2.json()


# ===============================================================================
# ERROR HANDLING TESTS
# ===============================================================================

class TestErrorHandling:
    """Test error scenarios and edge cases"""

    def test_invalid_file_upload(self):
        """Test upload with invalid file"""
        response = client.post("/smartdoc/upload", files={"file": ("", b"", "")})
        # This might return 422 for validation error or 400 for processing error
        assert response.status_code in [400, 422]

    def test_malformed_query_request(self):
        """Test query with malformed request body"""
        response = client.post("/smartdoc/document/1/query", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error

    def test_malformed_factura_request(self):
        """Test fiscal classification with malformed data"""
        with patch('bluebill_app.AVAILABLE_MODULES', {"fiscal_classifier": True}):
            response = client.post("/fiscal/classify", json={"invalid": "structure"})
            # Could be validation error or business logic error
            assert response.status_code in [400, 422]

    def test_endpoints_when_modules_disabled(self):
        """Test endpoint behavior when modules are disabled"""
        with patch('bluebill_app.AVAILABLE_MODULES', {"smartdoc": False, "fiscal_classifier": False}):
            # SmartDoc endpoints should return 503
            response = client.post("/smartdoc/upload", files={"file": ("test.txt", b"content", "text/plain")})
            assert response.status_code == 503

            # Fiscal endpoints should return 503
            response = client.post("/fiscal/validate", json={})
            assert response.status_code == 503


# ===============================================================================
# PERFORMANCE AND LOAD TESTS
# ===============================================================================

class TestPerformance:
    """Basic performance and load tests"""

    def test_multiple_document_uploads(self, mock_vllm, mock_embedding_model, clean_database):
        """Test handling multiple document uploads"""
        with patch('bluebill_app.extract_text') as mock_extract, \
                patch('bluebill_app.chunk_text') as mock_chunk, \
                patch('bluebill_app.create_embeddings') as mock_embeddings:
            mock_extract.return_value = "Test document content"
            mock_chunk.return_value = ["Test document", "content"]
            mock_embeddings.return_value = np.random.rand(2, 384).astype(np.float32)

            # Upload multiple documents
            doc_ids = []
            for i in range(5):
                test_file = (f"doc{i}.txt", f"Content {i}".encode(), "text/plain")
                response = client.post("/smartdoc/upload", files={"file": test_file})
                assert response.status_code == 200
                doc_ids.append(response.json()["doc_id"])

            # Verify all documents are listed
            list_response = client.get("/smartdoc/documents")
            assert len(list_response.json()) == 5

    def test_system_endpoints_response_time(self):
        """Test that system endpoints respond quickly"""
        import time

        # Test root endpoint
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond in under 1 second

        # Test health endpoint
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 2.0  # Health check might take a bit longer


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])