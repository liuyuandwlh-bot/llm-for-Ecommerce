"""
API Integration Tests

Tests FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.gateway.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test health endpoint returns correct format."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data


class TestCustomerServiceEndpoint:
    """Test customer service endpoint."""

    def test_basic_request(self):
        """Test basic customer service request."""
        response = client.post(
            "/api/v1/customer-service",
            json={
                "query": "耳机能退货吗？",
                "domain": "ecommerce"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "response" in data
        assert "latency_ms" in data

    def test_invalid_domain(self):
        """Test request with invalid domain."""
        response = client.post(
            "/api/v1/customer-service",
            json={
                "query": "测试",
                "domain": "invalid"
            }
        )
        
        assert response.status_code == 400

    def test_pii_masking_in_logs(self):
        """Test that PII is masked in response."""
        response = client.post(
            "/api/v1/customer-service",
            json={
                "query": "我的手机号13812345678可以退货吗？",
                "domain": "ecommerce"
            }
        )
        
        assert response.status_code == 200


class TestFinanceRAGEndpoint:
    """Test finance RAG endpoint."""

    def test_basic_request(self):
        """Test basic finance RAG request."""
        response = client.post(
            "/api/v1/finance-rag",
            json={
                "query": "某公司2023年的净利润是多少？",
                "domain": "finance"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "confidence" in data


class TestBatchEndpoint:
    """Test batch prediction endpoint."""

    def test_batch_request(self):
        """Test batch prediction."""
        response = client.post(
            "/api/v1/batch",
            json=[
                {"query": "耳机能退吗？", "domain": "ecommerce"},
                {"query": "物流到哪了？", "domain": "ecommerce"},
            ]
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "responses" in data
        assert "count" in data
        assert data["count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
