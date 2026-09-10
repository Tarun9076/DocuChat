import io


def test_index_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_status_before_upload(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.get_json() == {"document_loaded": False, "filename": None, "chunks": 0}


def test_upload_then_chat_then_reset(client):
    data = {"file": (io.BytesIO(b"Tesla was founded in 2003 by Martin Eberhard and Marc Tarpenning."), "tesla.txt")}
    upload_response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert upload_response.status_code == 200
    body = upload_response.get_json()
    assert body["filename"] == "tesla.txt"
    assert body["chunks"] >= 1

    status_response = client.get("/api/status")
    assert status_response.get_json()["document_loaded"] is True

    chat_response = client.post("/api/chat", json={"message": "When was Tesla founded?"})
    assert chat_response.status_code == 200
    assert b"2003" in chat_response.data

    reset_response = client.post("/api/reset")
    assert reset_response.status_code == 200
    assert reset_response.get_json() == {"ok": True}


def test_upload_rejects_unsupported_extension(client):
    data = {"file": (io.BytesIO(b"not a real doc"), "malware.exe")}
    response = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_chat_without_upload_is_rejected(client):
    response = client.post("/api/chat", json={"message": "Anything?"})
    assert response.status_code == 400
